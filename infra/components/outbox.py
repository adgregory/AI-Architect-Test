"""Production target for job dispatch: transactional outbox → Debezium Server → Kinesis → dispatcher Lambda.

Debezium reads the `outbox` table through a logical replication slot (RDS parameter group
enables it), routes events with the outbox EventRouter and publishes to Kinesis. The Lambda
starts the Temporal workflow idempotently (workflow_id = job_id), reports partial batch
failures, bisects on error, and sends exhausted records to an SQS dead-letter queue.

Gated by the `outboxEnabled` config (requires the app's outbox write mode — the local build
starts workflows directly; see ADR 0001).
"""

from __future__ import annotations

import json

import pulumi
import pulumi_aws as aws

from components.data import Database, FileSystem, Registry
from components.fargate_service import FargateService, ServiceSpec
from components.network import Network, SecurityGroups
from components.platform import Cluster
from config import StackConfig

DEBEZIUM_IMAGE = "quay.io/debezium/server:3.2"
# EventRouter publishes to "outbox.event.<aggregatetype>"; the Kinesis sink uses it as the stream name.
STREAM_NAME = "outbox.event.job"


class Outbox(pulumi.ComponentResource):
    def __init__(
        self,
        cfg: StackConfig,
        network: Network,
        sgs: SecurityGroups,
        cluster: Cluster,
        registry: Registry,
        db: Database,
        efs: FileSystem,
        opts: pulumi.ResourceOptions | None = None,
    ):
        super().__init__("pdfx:events:Outbox", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)

        self.stream = aws.kinesis.Stream(
            f"{cfg.name}-outbox",
            name=f"{cfg.name}.{STREAM_NAME}",
            shard_count=cfg.kinesis_shards,
            retention_period=24,
            encryption_type="KMS",
            kms_key_id="alias/aws/kinesis",
            stream_mode_details=aws.kinesis.StreamStreamModeDetailsArgs(stream_mode="PROVISIONED"),
            tags=cfg.tags,
            opts=child,
        )

        self.debezium = FargateService(
            ServiceSpec(
                name=f"{cfg.name}-debezium",
                image=DEBEZIUM_IMAGE,
                size=cfg.debezium,
                environment={
                    "DEBEZIUM_SOURCE_CONNECTOR_CLASS": "io.debezium.connector.postgresql.PostgresConnector",
                    "DEBEZIUM_SOURCE_PLUGIN_NAME": "pgoutput",
                    "DEBEZIUM_SOURCE_DATABASE_HOSTNAME": db.instance.address,
                    "DEBEZIUM_SOURCE_DATABASE_PORT": "5432",
                    "DEBEZIUM_SOURCE_DATABASE_DBNAME": "app",
                    "DEBEZIUM_SOURCE_TABLE_INCLUDE_LIST": "public.outbox",
                    "DEBEZIUM_SOURCE_PUBLICATION_AUTOCREATE_MODE": "filtered",
                    "DEBEZIUM_SOURCE_SLOT_NAME": "outbox_slot",
                    "DEBEZIUM_SOURCE_TOPIC_PREFIX": cfg.name,
                    # Offsets on EFS: durable across task restarts (an ephemeral file would replay or skip).
                    "DEBEZIUM_SOURCE_OFFSET_STORAGE_FILE_FILENAME": "/debezium/data/offsets.dat",
                    "DEBEZIUM_SOURCE_OFFSET_FLUSH_INTERVAL_MS": "1000",
                    "DEBEZIUM_TRANSFORMS": "outbox",
                    "DEBEZIUM_TRANSFORMS_OUTBOX_TYPE": "io.debezium.transforms.outbox.EventRouter",
                    "DEBEZIUM_TRANSFORMS_OUTBOX_ROUTE_TOPIC_REPLACEMENT": f"{cfg.name}.outbox.event.${{routedByValue}}",
                    "DEBEZIUM_SINK_TYPE": "kinesis",
                    "DEBEZIUM_SINK_KINESIS_REGION": cfg.region,
                },
                secrets={
                    "DEBEZIUM_SOURCE_DATABASE_USER": (db.secret_arn, "username"),
                    "DEBEZIUM_SOURCE_DATABASE_PASSWORD": (db.secret_arn, "password"),
                },
                efs_volume=efs.volume("debezium", "/debezium/data"),
                task_policy_statements=[
                    {
                        "Effect": "Allow",
                        "Action": ["kinesis:PutRecord", "kinesis:PutRecords", "kinesis:DescribeStream"],
                        "Resource": self.stream.arn,
                    },
                    *efs.client_statements("debezium"),
                ],
            ),
            cluster_arn=cluster.cluster.arn,
            subnet_ids=network.app_subnet_ids,
            security_group_id=sgs.debezium.id,
            assign_public_ip=network.assign_public_ip,
            namespace_id=None,
            region=cfg.region,
            log_retention_days=cfg.log_retention_days,
            tags=cfg.tags,
            opts=child,
        )

        self.dlq = aws.sqs.Queue(
            f"{cfg.name}-dispatch-dlq",
            message_retention_seconds=14 * 24 * 3600,
            sqs_managed_sse_enabled=True,
            tags=cfg.tags,
            opts=child,
        )

        role = aws.iam.Role(
            f"{cfg.name}-dispatcher-role",
            assume_role_policy=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": "sts:AssumeRole",
                            "Principal": {"Service": "lambda.amazonaws.com"},
                        }
                    ],
                }
            ),
            tags=cfg.tags,
            opts=child,
        )
        for name, arn in (("vpc", "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"),):
            aws.iam.RolePolicyAttachment(f"{cfg.name}-dispatcher-{name}", role=role.name, policy_arn=arn, opts=child)
        aws.iam.RolePolicy(
            f"{cfg.name}-dispatcher-policy",
            role=role.id,
            policy=pulumi.Output.json_dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Resource": self.stream.arn,
                            "Action": [
                                "kinesis:GetRecords",
                                "kinesis:GetShardIterator",
                                "kinesis:DescribeStream",
                                "kinesis:DescribeStreamSummary",
                                "kinesis:ListShards",
                            ],
                        },
                        {"Effect": "Allow", "Action": "kinesis:ListStreams", "Resource": "*"},
                        {"Effect": "Allow", "Action": "sqs:SendMessage", "Resource": self.dlq.arn},
                    ],
                }
            ),
            opts=child,
        )

        self.function = aws.lambda_.Function(
            f"{cfg.name}-dispatcher",
            package_type="Image",
            image_uri=registry.image("dispatcher", cfg.image_tag),
            role=role.arn,
            timeout=30,
            memory_size=256,
            architectures=["arm64"],
            environment=aws.lambda_.FunctionEnvironmentArgs(
                variables={
                    "TEMPORAL_ADDRESS": pulumi.Output.concat(cluster.dns("temporal"), ":7233"),
                    "LOG_JSON": "true",
                }
            ),
            vpc_config=aws.lambda_.FunctionVpcConfigArgs(
                subnet_ids=network.app_subnet_ids, security_group_ids=[sgs.dispatcher.id]
            ),
            tags=cfg.tags,
            opts=child,
        )
        aws.cloudwatch.LogGroup(
            f"{cfg.name}-dispatcher-logs",
            name=self.function.name.apply(lambda n: f"/aws/lambda/{n}"),
            retention_in_days=cfg.log_retention_days,
            tags=cfg.tags,
            opts=child,
        )

        self.mapping = aws.lambda_.EventSourceMapping(
            f"{cfg.name}-dispatcher-esm",
            event_source_arn=self.stream.arn,
            function_name=self.function.arn,
            starting_position="TRIM_HORIZON",
            batch_size=50,
            maximum_batching_window_in_seconds=1,
            function_response_types=["ReportBatchItemFailures"],  # only failed records are retried
            bisect_batch_on_function_error=True,
            maximum_retry_attempts=5,
            maximum_record_age_in_seconds=3600,
            parallelization_factor=2,
            destination_config=aws.lambda_.EventSourceMappingDestinationConfigArgs(
                on_failure=aws.lambda_.EventSourceMappingDestinationConfigOnFailureArgs(destination_arn=self.dlq.arn)
            ),
            opts=child,
        )
        self.register_outputs({"stream": self.stream.name, "dlq": self.dlq.url})
