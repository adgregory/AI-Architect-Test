"""Shared platform: ECS cluster + Cloud Map namespace, load balancer, Temporal (+UI), Qdrant."""

from __future__ import annotations

import pulumi
import pulumi_aws as aws

from components.data import Database, FileSystem
from components.fargate_service import FargateService, ServiceSpec
from components.network import Network, SecurityGroups
from config import StackConfig

TEMPORAL_IMAGE = "temporalio/server:1.28.1"
TEMPORAL_ADMIN_IMAGE = "temporalio/admin-tools:1.28.1-tctl-1.18.2-cli-1.3.0"
TEMPORAL_UI_IMAGE = "temporalio/ui:2.40.1"
QDRANT_IMAGE = "qdrant/qdrant:v1.15.4"


class Cluster(pulumi.ComponentResource):
    def __init__(self, cfg: StackConfig, network: Network, opts: pulumi.ResourceOptions | None = None):
        super().__init__("pdfx:compute:Cluster", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)
        self.cluster = aws.ecs.Cluster(
            f"{cfg.name}-cluster",
            name=cfg.name,
            settings=[aws.ecs.ClusterSettingArgs(name="containerInsights", value="enabled")],
            tags=cfg.tags,
            opts=child,
        )
        # Internal service discovery: <service>.<name>.local (gRPC to Temporal never touches the ALB).
        self.namespace = aws.servicediscovery.PrivateDnsNamespace(
            f"{cfg.name}-ns", name=f"{cfg.name}.local", vpc=network.vpc.id, tags=cfg.tags, opts=child
        )
        self.register_outputs({"cluster_arn": self.cluster.arn})

    def dns(self, service: str) -> pulumi.Output[str]:
        return pulumi.Output.concat(service, ".", self.namespace.name)


class LoadBalancer(pulumi.ComponentResource):
    """Public API listener (HTTPS when a certificate is configured) and an admin-only
    listener for the Temporal UI on 8233 (restricted by security group to adminCidrs)."""

    def __init__(
        self, cfg: StackConfig, network: Network, sgs: SecurityGroups, opts: pulumi.ResourceOptions | None = None
    ):
        super().__init__("pdfx:compute:LoadBalancer", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)
        self.alb = aws.lb.LoadBalancer(
            f"{cfg.name}-alb",
            name=f"{cfg.short}-alb",
            load_balancer_type="application",
            subnets=network.public_subnet_ids,
            security_groups=[sgs.alb.id],
            drop_invalid_header_fields=True,
            idle_timeout=120,  # SSE streams
            tags=cfg.tags,
            opts=child,
        )

        def target_group(name: str, port: int, health_path: str) -> aws.lb.TargetGroup:
            return aws.lb.TargetGroup(
                f"{cfg.name}-{name}-tg",
                name=f"{cfg.short}-{name}",
                port=port,
                protocol="HTTP",
                target_type="ip",
                vpc_id=network.vpc.id,
                deregistration_delay=30,
                health_check=aws.lb.TargetGroupHealthCheckArgs(
                    path=health_path, matcher="200", interval=15, healthy_threshold=2, unhealthy_threshold=3
                ),
                tags=cfg.tags,
                opts=child,
            )

        self.api_tg = target_group("api", 8000, "/health")
        self.ui_tg = target_group("ui", 8080, "/")
        if cfg.certificate_arn:
            aws.lb.Listener(
                f"{cfg.name}-https",
                load_balancer_arn=self.alb.arn,
                port=443,
                protocol="HTTPS",
                ssl_policy="ELBSecurityPolicy-TLS13-1-2-2021-06",
                certificate_arn=cfg.certificate_arn,
                default_actions=[aws.lb.ListenerDefaultActionArgs(type="forward", target_group_arn=self.api_tg.arn)],
                opts=child,
            )
        else:
            aws.lb.Listener(
                f"{cfg.name}-http",
                load_balancer_arn=self.alb.arn,
                port=80,
                protocol="HTTP",
                default_actions=[aws.lb.ListenerDefaultActionArgs(type="forward", target_group_arn=self.api_tg.arn)],
                opts=child,
            )
        aws.lb.Listener(
            f"{cfg.name}-ui",
            load_balancer_arn=self.alb.arn,
            port=8233,
            protocol="HTTP",
            default_actions=[aws.lb.ListenerDefaultActionArgs(type="forward", target_group_arn=self.ui_tg.arn)],
            opts=child,
        )
        self.register_outputs({"dns_name": self.alb.dns_name})


class Platform(pulumi.ComponentResource):
    """Temporal server (one replica; persistence in RDS) + UI, and Qdrant (data on EFS)."""

    def __init__(
        self,
        cfg: StackConfig,
        network: Network,
        sgs: SecurityGroups,
        cluster: Cluster,
        alb: LoadBalancer,
        db: Database,
        efs: FileSystem,
        opts: pulumi.ResourceOptions | None = None,
    ):
        super().__init__("pdfx:compute:Platform", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)
        common = dict(
            cluster_arn=cluster.cluster.arn,
            subnet_ids=network.app_subnet_ids,
            assign_public_ip=network.assign_public_ip,
            namespace_id=cluster.namespace.id,
            region=cfg.region,
            log_retention_days=cfg.log_retention_days,
            tags=cfg.tags,
        )
        db_env = {
            "DB": "postgres12",
            "DB_PORT": "5432",
            "POSTGRES_SEEDS": db.instance.address,
            "DBNAME": "temporal",
            "VISIBILITY_DBNAME": "temporal_visibility",
            "SQL_MAX_CONNS": "20",
        }
        db_secrets = {"POSTGRES_USER": (db.secret_arn, "username"), "POSTGRES_PWD": (db.secret_arn, "password")}

        self.temporal = FargateService(
            ServiceSpec(
                name=f"{cfg.name}-temporal",
                image=TEMPORAL_IMAGE,
                size=cfg.temporal,
                port=7233,
                environment=db_env,
                secrets=db_secrets,
                discovery_name="temporal",
            ),
            security_group_id=sgs.temporal.id,
            **common,
            opts=child,
        )

        # One-off schema setup/upgrade (run from CI before deploying a new Temporal version).
        self.temporal_schema_task = FargateService(
            ServiceSpec(
                name=f"{cfg.name}-temporal-schema",
                image=TEMPORAL_ADMIN_IMAGE,
                size=cfg.temporal,
                run_as_service=False,
                command=[
                    "sh",
                    "-c",
                    "temporal-sql-tool --plugin postgres12 --ep $POSTGRES_SEEDS -u $POSTGRES_USER "
                    "--pw $POSTGRES_PWD --db temporal create-database || true; "
                    "temporal-sql-tool --plugin postgres12 --ep $POSTGRES_SEEDS -u $POSTGRES_USER "
                    "--pw $POSTGRES_PWD --db temporal update-schema -d "
                    "/etc/temporal/schema/postgresql/v12/temporal/versioned",
                ],
                environment=db_env,
                secrets=db_secrets,
            ),
            security_group_id=sgs.temporal.id,
            **common,
            opts=child,
        )

        self.temporal_ui = FargateService(
            ServiceSpec(
                name=f"{cfg.name}-temporal-ui",
                image=TEMPORAL_UI_IMAGE,
                size=cfg.temporal.model_copy(update={"cpu": 256, "memory": 512}),
                port=8080,
                environment={"TEMPORAL_ADDRESS": pulumi.Output.concat(cluster.dns("temporal"), ":7233")},
                target_group_arn=alb.ui_tg.arn,
            ),
            security_group_id=sgs.temporal_ui.id,
            **common,
            opts=child,
        )

        self.qdrant = FargateService(
            ServiceSpec(
                name=f"{cfg.name}-qdrant",
                image=QDRANT_IMAGE,
                size=cfg.qdrant,
                port=6333,
                discovery_name="qdrant",
                efs_volume=efs.volume("qdrant", "/qdrant/storage"),
                task_policy_statements=efs.client_statements("qdrant"),
            ),
            security_group_id=sgs.qdrant.id,
            **common,
            opts=child,
        )

        self.register_outputs({})
