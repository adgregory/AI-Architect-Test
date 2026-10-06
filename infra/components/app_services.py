"""Application workloads: API, CPU workers, IO workers, migration task, and the AgentCore agent."""

from __future__ import annotations

import json

import pulumi
import pulumi_aws as aws

from components.data import ArtifactsBucket, Database, Registry
from components.fargate_service import FargateService, ServiceSpec
from components.network import Network, SecurityGroups
from components.platform import Cluster, LoadBalancer
from config import StackConfig


class Agent(pulumi.ComponentResource):
    """Strands agent on AgentCore Runtime with a Bedrock model. The execution role may invoke
    only the configured foundation model; the runtime runs inside the VPC to reach Qdrant."""

    def __init__(
        self,
        cfg: StackConfig,
        network: Network,
        sgs: SecurityGroups,
        cluster: Cluster,
        registry: Registry,
        opts: pulumi.ResourceOptions | None = None,
    ):
        super().__init__("pdfx:compute:Agent", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)
        model_arn = f"arn:aws:bedrock:{cfg.region}::foundation-model/{cfg.bedrock_model_id}"

        self.role = aws.iam.Role(
            f"{cfg.name}-agent-role",
            assume_role_policy=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": "sts:AssumeRole",
                            "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                        }
                    ],
                }
            ),
            tags=cfg.tags,
            opts=child,
        )
        aws.iam.RolePolicy(
            f"{cfg.name}-agent-policy",
            role=self.role.id,
            policy=pulumi.Output.json_dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "InvokeConfiguredModelOnly",
                            "Effect": "Allow",
                            "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                            "Resource": model_arn,
                        },
                        {
                            "Sid": "PullAgentImage",
                            "Effect": "Allow",
                            "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
                            "Resource": registry.repos["agent"].arn,
                        },
                        {"Sid": "EcrAuth", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
                        {
                            "Sid": "Logs",
                            "Effect": "Allow",
                            "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                            "Resource": f"arn:aws:logs:{cfg.region}:*:log-group:/aws/bedrock-agentcore/*",
                        },
                    ],
                }
            ),
            opts=child,
        )

        self.runtime = aws.bedrock.AgentcoreAgentRuntime(
            f"{cfg.name}-agent",
            agent_runtime_name=cfg.name.replace("-", "_") + "_agent",
            description="PDF documents Q&A agent (Strands, retrieve-then-generate, semantic cache)",
            role_arn=self.role.arn,
            agent_runtime_artifact=aws.bedrock.AgentcoreAgentRuntimeAgentRuntimeArtifactArgs(
                container_configuration=aws.bedrock.AgentcoreAgentRuntimeAgentRuntimeArtifactContainerConfigurationArgs(
                    container_uri=registry.image("agent", cfg.image_tag)
                )
            ),
            network_configuration=aws.bedrock.AgentcoreAgentRuntimeNetworkConfigurationArgs(
                network_mode="VPC",
                network_mode_config=aws.bedrock.AgentcoreAgentRuntimeNetworkConfigurationNetworkModeConfigArgs(
                    security_groups=[sgs.agent.id], subnets=network.app_subnet_ids
                ),
            ),
            protocol_configuration=aws.bedrock.AgentcoreAgentRuntimeProtocolConfigurationArgs(server_protocol="HTTP"),
            environment_variables={
                "LLM_PROVIDER": "bedrock",
                "BEDROCK_MODEL_ID": cfg.bedrock_model_id,
                "AWS_REGION": cfg.region,
                "QDRANT_HOST": cluster.dns("qdrant"),
                "LOG_JSON": "true",
            },
            tags=cfg.tags,
            opts=child,
        )
        self.register_outputs({"runtime_arn": self.runtime.agent_runtime_arn})


class AppServices(pulumi.ComponentResource):
    def __init__(
        self,
        cfg: StackConfig,
        network: Network,
        sgs: SecurityGroups,
        cluster: Cluster,
        alb: LoadBalancer,
        registry: Registry,
        db: Database,
        bucket: ArtifactsBucket,
        agent: Agent,
        opts: pulumi.ResourceOptions | None = None,
    ):
        super().__init__("pdfx:compute:AppServices", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)
        image = registry.image("app", cfg.image_tag)
        common = dict(
            cluster_arn=cluster.cluster.arn,
            subnet_ids=network.app_subnet_ids,
            assign_public_ip=network.assign_public_ip,
            namespace_id=cluster.namespace.id,
            region=cfg.region,
            log_retention_days=cfg.log_retention_days,
            tags=cfg.tags,
        )
        env = {
            "LOG_JSON": "true",
            "DB_HOST": db.instance.address,
            "DB_NAME": "app",
            "TEMPORAL_ADDRESS": pulumi.Output.concat(cluster.dns("temporal"), ":7233"),
            "QDRANT_HOST": cluster.dns("qdrant"),
            "STORAGE_BACKEND": "s3",
            "STORAGE_BUCKET": bucket.bucket.bucket,
            "ANSWER_BACKEND": "agentcore",
            "AGENT_RUNTIME_ARN": agent.runtime.agent_runtime_arn,
            "AWS_REGION": cfg.region,
        }
        db_secrets = {"DB_USER": (db.secret_arn, "username"), "DB_PASSWORD": (db.secret_arn, "password")}
        s3 = bucket.rw_statements()

        self.api = FargateService(
            ServiceSpec(
                name=f"{cfg.name}-api",
                image=image,
                size=cfg.api,
                port=8000,
                environment=env,
                secrets=db_secrets,
                target_group_arn=alb.api_tg.arn,
                task_policy_statements=[
                    *s3,
                    {
                        "Effect": "Allow",
                        "Action": "bedrock-agentcore:InvokeAgentRuntime",
                        "Resource": [
                            agent.runtime.agent_runtime_arn,
                            pulumi.Output.concat(agent.runtime.agent_runtime_arn, "/*"),
                        ],
                    },
                ],
                health_check_command=[
                    "CMD-SHELL",
                    'python -c "import urllib.request,sys; '
                    "sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status!=200)\"",
                ],
            ),
            security_group_id=sgs.api.id,
            **common,
            opts=child,
        )

        # CPU workers: models only. No DB credentials and no DB network path (see SecurityGroups).
        self.worker_cpu = FargateService(
            ServiceSpec(
                name=f"{cfg.name}-worker-cpu",
                image=image,
                size=cfg.worker_cpu,
                command=["python", "-m", "app.workers.cpu"],
                environment={**env, "CPU_WORKER_CONCURRENCY": str(cfg.worker_cpu.concurrency)},
                task_policy_statements=s3,
            ),
            security_group_id=sgs.worker_cpu.id,
            **common,
            opts=child,
        )

        # IO workers: the only worker-side owner of a database pool; also host the workflows.
        self.worker_io = FargateService(
            ServiceSpec(
                name=f"{cfg.name}-worker-io",
                image=image,
                size=cfg.worker_io,
                command=["python", "-m", "app.workers.io"],
                environment=env,
                secrets=db_secrets,
                task_policy_statements=s3,
            ),
            security_group_id=sgs.worker_io.id,
            **common,
            opts=child,
        )

        # One-off: `aws ecs run-task` from CI before rolling out a release.
        self.migrate = FargateService(
            ServiceSpec(
                name=f"{cfg.name}-migrate",
                image=image,
                size=cfg.worker_io,
                run_as_service=False,
                command=["alembic", "upgrade", "head"],
                environment=env,
                secrets=db_secrets,
            ),
            security_group_id=sgs.worker_io.id,
            **common,
            opts=child,
        )

        self.register_outputs({})
