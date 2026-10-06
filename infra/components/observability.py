"""Alarms for the failure modes that matter (ADR 0001), routed to one SNS topic."""

from __future__ import annotations

import pulumi
import pulumi_aws as aws

from components.data import Database
from components.outbox import Outbox
from components.platform import LoadBalancer
from config import StackConfig


class Alarms(pulumi.ComponentResource):
    def __init__(
        self,
        cfg: StackConfig,
        db: Database,
        alb: LoadBalancer,
        outbox: Outbox | None,
        opts: pulumi.ResourceOptions | None = None,
    ):
        super().__init__("pdfx:ops:Alarms", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)
        self.topic = aws.sns.Topic(f"{cfg.name}-alarms", tags=cfg.tags, opts=child)

        def alarm(
            name: str,
            namespace: str,
            metric: str,
            dims,
            threshold: float,
            comparison: str,
            stat: str = "Average",
            periods: int = 3,
            description: str = "",
        ):
            return aws.cloudwatch.MetricAlarm(
                f"{cfg.name}-{name}",
                namespace=namespace,
                metric_name=metric,
                dimensions=dims,
                statistic=stat,
                period=60,
                evaluation_periods=periods,
                threshold=threshold,
                comparison_operator=comparison,
                treat_missing_data="notBreaching",
                alarm_description=description,
                alarm_actions=[self.topic.arn],
                ok_actions=[self.topic.arn],
                tags=cfg.tags,
                opts=child,
            )

        rds = {"DBInstanceIdentifier": db.instance.identifier}
        alarm(
            "rds-free-storage",
            "AWS/RDS",
            "FreeStorageSpace",
            rds,
            5 * 1024**3,
            "LessThanThreshold",
            description="Disk filling up (check replication-slot WAL retention first).",
        )
        alarm(
            "rds-connections",
            "AWS/RDS",
            "DatabaseConnections",
            rds,
            160,
            "GreaterThanThreshold",
            description="Approaching max_connections (200): check pool sizes vs replica counts.",
        )
        alarm("rds-cpu", "AWS/RDS", "CPUUtilization", rds, 80, "GreaterThanThreshold")
        alb_dims = {"LoadBalancer": alb.alb.arn_suffix}
        alarm(
            "alb-5xx",
            "AWS/ApplicationELB",
            "HTTPCode_Target_5XX_Count",
            alb_dims,
            10,
            "GreaterThanThreshold",
            stat="Sum",
            description="API returning server errors (incl. 503 load shedding).",
        )

        if outbox is not None:
            alarm(
                "replication-slot-lag",
                "AWS/RDS",
                "OldestReplicationSlotLag",
                rds,
                1024**3,
                "GreaterThanThreshold",
                description="Debezium is stalled: its slot is retaining WAL (risk: disk full).",
            )
            alarm(
                "dispatcher-errors",
                "AWS/Lambda",
                "Errors",
                {"FunctionName": outbox.function.name},
                1,
                "GreaterThanOrEqualToThreshold",
                stat="Sum",
                periods=1,
            )
            alarm(
                "dispatch-dlq",
                "AWS/SQS",
                "ApproximateNumberOfMessagesVisible",
                {"QueueName": outbox.dlq.name},
                1,
                "GreaterThanOrEqualToThreshold",
                stat="Maximum",
                periods=1,
                description="Jobs whose workflow could not be started: inspect and redrive.",
            )
            alarm(
                "outbox-iterator-age",
                "AWS/Kinesis",
                "GetRecords.IteratorAgeMilliseconds",
                {"StreamName": outbox.stream.name},
                60_000,
                "GreaterThanThreshold",
                stat="Maximum",
            )
        self.register_outputs({"topic": self.topic.arn})
