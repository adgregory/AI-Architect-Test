"""Reusable Fargate service: task definition, roles, log group, security group wiring,
optional Cloud Map service discovery and optional load-balancer registration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pulumi
import pulumi_aws as aws

from config import Size


@dataclass
class ServiceSpec:
    name: str
    image: pulumi.Input[str]
    size: Size
    port: int | None = None
    command: list[str] | None = None
    environment: dict[str, pulumi.Input[str]] = field(default_factory=dict)
    # env var name -> (Secrets Manager secret ARN, JSON key or None)
    secrets: dict[str, tuple[pulumi.Input[str], str | None]] = field(default_factory=dict)
    discovery_name: str | None = None  # Cloud Map name, e.g. "temporal" -> temporal.<ns>
    target_group_arn: pulumi.Input[str] | None = None
    task_policy_statements: list[dict] = field(default_factory=list)
    efs_volume: dict | None = None  # {"file_system_id", "access_point_id", "container_path"}
    health_check_command: list[str] | None = None
    run_as_service: bool = True  # False: task definition only (one-off tasks run by CI)


class FargateService(pulumi.ComponentResource):
    def __init__(
        self,
        spec: ServiceSpec,
        *,
        cluster_arn: pulumi.Input[str],
        subnet_ids: pulumi.Input[list[str]],
        security_group_id: pulumi.Input[str],
        assign_public_ip: bool,
        namespace_id: pulumi.Input[str] | None,
        region: str,
        log_retention_days: int,
        tags: dict[str, str],
        opts: pulumi.ResourceOptions | None = None,
    ):
        super().__init__("pdfx:ecs:FargateService", spec.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)

        self.log_group = aws.cloudwatch.LogGroup(
            f"{spec.name}-logs", name=f"/ecs/{spec.name}", retention_in_days=log_retention_days, tags=tags, opts=child
        )

        assume = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}
                ],
            }
        )
        # Execution role: pull images, write logs, read the injected secrets.
        self.execution_role = aws.iam.Role(f"{spec.name}-exec", assume_role_policy=assume, tags=tags, opts=child)
        aws.iam.RolePolicyAttachment(
            f"{spec.name}-exec-policy",
            role=self.execution_role.name,
            policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
            opts=child,
        )
        if spec.secrets:
            secret_arns = [arn for arn, _ in spec.secrets.values()]
            aws.iam.RolePolicy(
                f"{spec.name}-exec-secrets",
                role=self.execution_role.id,
                policy=pulumi.Output.json_dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {"Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"], "Resource": secret_arns}
                        ],
                    }
                ),
                opts=child,
            )
        # Task role: what the application itself may do (least privilege, per service).
        self.task_role = aws.iam.Role(f"{spec.name}-task", assume_role_policy=assume, tags=tags, opts=child)
        if spec.task_policy_statements:
            aws.iam.RolePolicy(
                f"{spec.name}-task-policy",
                role=self.task_role.id,
                policy=pulumi.Output.json_dumps({"Version": "2012-10-17", "Statement": spec.task_policy_statements}),
                opts=child,
            )

        container: dict = {
            "name": spec.name,
            "image": spec.image,
            "essential": True,
            "environment": [{"name": k, "value": v} for k, v in spec.environment.items()],
            "secrets": [
                {"name": k, "valueFrom": pulumi.Output.concat(arn, f":{key}::") if key else arn}
                for k, (arn, key) in spec.secrets.items()
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": self.log_group.name,
                    "awslogs-region": region,
                    "awslogs-stream-prefix": spec.name,
                },
            },
        }
        if spec.command:
            container["command"] = spec.command
        if spec.port:
            container["portMappings"] = [{"containerPort": spec.port, "protocol": "tcp"}]
        if spec.health_check_command:
            container["healthCheck"] = {
                "command": spec.health_check_command,
                "interval": 15,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 60,
            }
        volumes = None
        if spec.efs_volume:
            container["mountPoints"] = [{"sourceVolume": "data", "containerPath": spec.efs_volume["container_path"]}]
            volumes = [
                aws.ecs.TaskDefinitionVolumeArgs(
                    name="data",
                    efs_volume_configuration=aws.ecs.TaskDefinitionVolumeEfsVolumeConfigurationArgs(
                        file_system_id=spec.efs_volume["file_system_id"],
                        transit_encryption="ENABLED",
                        authorization_config=aws.ecs.TaskDefinitionVolumeEfsVolumeConfigurationAuthorizationConfigArgs(
                            access_point_id=spec.efs_volume["access_point_id"], iam="ENABLED"
                        ),
                    ),
                )
            ]

        self.task_definition = aws.ecs.TaskDefinition(
            f"{spec.name}-task-def",
            family=spec.name,
            cpu=str(spec.size.cpu),
            memory=str(spec.size.memory),
            network_mode="awsvpc",
            requires_compatibilities=["FARGATE"],
            runtime_platform=aws.ecs.TaskDefinitionRuntimePlatformArgs(
                operating_system_family="LINUX", cpu_architecture="ARM64"
            ),
            execution_role_arn=self.execution_role.arn,
            task_role_arn=self.task_role.arn,
            container_definitions=pulumi.Output.json_dumps([container]),
            volumes=volumes,
            tags=tags,
            opts=child,
        )

        if not spec.run_as_service:
            self.service = None
            self.register_outputs({"task_definition_arn": self.task_definition.arn})
            return

        registries = None
        if spec.discovery_name and namespace_id is not None:
            self.discovery = aws.servicediscovery.Service(
                f"{spec.name}-discovery",
                name=spec.discovery_name,
                dns_config=aws.servicediscovery.ServiceDnsConfigArgs(
                    namespace_id=namespace_id,
                    routing_policy="MULTIVALUE",
                    dns_records=[aws.servicediscovery.ServiceDnsConfigDnsRecordArgs(ttl=10, type="A")],
                ),
                health_check_custom_config=aws.servicediscovery.ServiceHealthCheckCustomConfigArgs(),
                tags=tags,
                opts=child,
            )
            registries = aws.ecs.ServiceServiceRegistriesArgs(registry_arn=self.discovery.arn)

        load_balancers = None
        if spec.target_group_arn is not None and spec.port:
            load_balancers = [
                aws.ecs.ServiceLoadBalancerArgs(
                    target_group_arn=spec.target_group_arn, container_name=spec.name, container_port=spec.port
                )
            ]

        self.service = aws.ecs.Service(
            f"{spec.name}-service",
            name=spec.name,
            cluster=cluster_arn,
            task_definition=self.task_definition.arn,
            desired_count=spec.size.desired_count,
            launch_type="FARGATE",
            network_configuration=aws.ecs.ServiceNetworkConfigurationArgs(
                subnets=subnet_ids, security_groups=[security_group_id], assign_public_ip=assign_public_ip
            ),
            service_registries=registries,
            load_balancers=load_balancers,
            deployment_circuit_breaker=aws.ecs.ServiceDeploymentCircuitBreakerArgs(enable=True, rollback=True),
            enable_execute_command=False,
            propagate_tags="SERVICE",
            tags=tags,
            opts=child,
        )

        self.register_outputs({"service_name": self.service.name, "task_role_arn": self.task_role.arn})
