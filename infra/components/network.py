"""VPC, subnets, routing and security groups.

`natGateway: false` (dev): tasks run in public subnets with public IPs — no NAT cost.
`natGateway: true` (staging): tasks run in private subnets behind one NAT gateway.
No data-source lookups (AZs come from config), so previews work without AWS credentials.
"""

from __future__ import annotations

import ipaddress

import pulumi
import pulumi_aws as aws

from config import StackConfig


class Network(pulumi.ComponentResource):
    def __init__(self, cfg: StackConfig, opts: pulumi.ResourceOptions | None = None):
        super().__init__("pdfx:network:Network", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)
        tags = cfg.tags
        subnets = list(ipaddress.ip_network(cfg.vpc_cidr).subnets(new_prefix=20))

        self.vpc = aws.ec2.Vpc(
            f"{cfg.name}-vpc",
            cidr_block=cfg.vpc_cidr,
            enable_dns_hostnames=True,
            enable_dns_support=True,
            tags={**tags, "Name": f"{cfg.name}-vpc"},
            opts=child,
        )
        igw = aws.ec2.InternetGateway(f"{cfg.name}-igw", vpc_id=self.vpc.id, tags=tags, opts=child)

        public_rt = aws.ec2.RouteTable(
            f"{cfg.name}-public-rt",
            vpc_id=self.vpc.id,
            tags=tags,
            routes=[aws.ec2.RouteTableRouteArgs(cidr_block="0.0.0.0/0", gateway_id=igw.id)],
            opts=child,
        )
        self.public_subnets, self.private_subnets = [], []
        for i, az in enumerate(cfg.availability_zones):
            pub = aws.ec2.Subnet(
                f"{cfg.name}-public-{az}",
                vpc_id=self.vpc.id,
                cidr_block=str(subnets[i]),
                availability_zone=az,
                map_public_ip_on_launch=False,
                tags={**tags, "Name": f"{cfg.name}-public-{az}", "tier": "public"},
                opts=child,
            )
            aws.ec2.RouteTableAssociation(
                f"{cfg.name}-public-rta-{az}", subnet_id=pub.id, route_table_id=public_rt.id, opts=child
            )
            priv = aws.ec2.Subnet(
                f"{cfg.name}-private-{az}",
                vpc_id=self.vpc.id,
                cidr_block=str(subnets[i + len(cfg.availability_zones)]),
                availability_zone=az,
                tags={**tags, "Name": f"{cfg.name}-private-{az}", "tier": "private"},
                opts=child,
            )
            self.public_subnets.append(pub)
            self.private_subnets.append(priv)

        route_tables = [public_rt.id]
        if cfg.nat_gateway:
            eip = aws.ec2.Eip(f"{cfg.name}-nat-eip", domain="vpc", tags=tags, opts=child)
            nat = aws.ec2.NatGateway(
                f"{cfg.name}-nat", allocation_id=eip.id, subnet_id=self.public_subnets[0].id, tags=tags, opts=child
            )
            private_rt = aws.ec2.RouteTable(
                f"{cfg.name}-private-rt",
                vpc_id=self.vpc.id,
                tags=tags,
                routes=[aws.ec2.RouteTableRouteArgs(cidr_block="0.0.0.0/0", nat_gateway_id=nat.id)],
                opts=child,
            )
            for az, priv in zip(cfg.availability_zones, self.private_subnets, strict=True):
                aws.ec2.RouteTableAssociation(
                    f"{cfg.name}-private-rta-{az}", subnet_id=priv.id, route_table_id=private_rt.id, opts=child
                )
            route_tables.append(private_rt.id)

        # Free gateway endpoint: S3 traffic (job artefacts) stays off the NAT / internet.
        aws.ec2.VpcEndpoint(
            f"{cfg.name}-s3-endpoint",
            vpc_id=self.vpc.id,
            service_name=f"com.amazonaws.{cfg.region}.s3",
            vpc_endpoint_type="Gateway",
            route_table_ids=route_tables,
            tags=tags,
            opts=child,
        )

        # Where workloads run, and whether they need a public IP to reach ECR / Bedrock / the internet.
        self.app_subnet_ids = [s.id for s in (self.private_subnets if cfg.nat_gateway else self.public_subnets)]
        self.public_subnet_ids = [s.id for s in self.public_subnets]
        self.assign_public_ip = not cfg.nat_gateway

        self.register_outputs({"vpc_id": self.vpc.id})


class SecurityGroups(pulumi.ComponentResource):
    """Least-privilege network paths. Every rule names its source group; nothing is open
    except the public API listener and the admin-only Temporal UI port."""

    def __init__(self, cfg: StackConfig, vpc_id: pulumi.Input[str], opts: pulumi.ResourceOptions | None = None):
        super().__init__("pdfx:network:SecurityGroups", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)

        def group(name: str, description: str) -> aws.ec2.SecurityGroup:
            sg = aws.ec2.SecurityGroup(
                f"{cfg.name}-{name}-sg",
                vpc_id=vpc_id,
                description=description,
                tags={**cfg.tags, "Name": f"{cfg.name}-{name}"},
                opts=child,
            )
            aws.vpc.SecurityGroupEgressRule(
                f"{cfg.name}-{name}-egress",
                security_group_id=sg.id,
                ip_protocol="-1",
                cidr_ipv4="0.0.0.0/0",
                opts=child,
            )
            return sg

        self.alb = group("alb", "Public load balancer")
        self.api = group("api", "API tasks")
        self.worker_cpu = group("worker-cpu", "CPU workers (no database access)")
        self.worker_io = group("worker-io", "IO workers (sole worker-side database owner)")
        self.temporal = group("temporal", "Temporal server")
        self.temporal_ui = group("temporal-ui", "Temporal UI")
        self.qdrant = group("qdrant", "Qdrant vector store")
        self.efs = group("efs", "EFS for Qdrant and Debezium state")
        self.agent = group("agent", "AgentCore runtime ENIs")
        self.rds = group("rds", "Postgres")
        self.debezium = group("debezium", "Debezium Server (outbox CDC)")
        self.dispatcher = group("dispatcher", "Dispatcher Lambda (Kinesis -> Temporal)")

        def allow(name: str, target, port: int, source=None, cidrs: list[str] | None = None):
            if cidrs:
                for i, cidr in enumerate(cidrs):
                    aws.vpc.SecurityGroupIngressRule(
                        f"{cfg.name}-{name}-{i}",
                        security_group_id=target.id,
                        ip_protocol="tcp",
                        from_port=port,
                        to_port=port,
                        cidr_ipv4=cidr,
                        opts=child,
                    )
            else:
                aws.vpc.SecurityGroupIngressRule(
                    f"{cfg.name}-{name}",
                    security_group_id=target.id,
                    ip_protocol="tcp",
                    from_port=port,
                    to_port=port,
                    referenced_security_group_id=source.id,
                    opts=child,
                )

        api_port = 443 if cfg.certificate_arn else 80
        allow("alb-from-internet", self.alb, api_port, cidrs=["0.0.0.0/0"])
        allow("alb-ui-from-admins", self.alb, 8233, cidrs=cfg.admin_cidrs)
        allow("api-from-alb", self.api, 8000, self.alb)
        allow("ui-from-alb", self.temporal_ui, 8080, self.alb)
        for src in ("api", "worker_cpu", "worker_io", "temporal_ui", "dispatcher"):
            allow(f"temporal-from-{src}", self.temporal, 7233, getattr(self, src))
        for src in ("api", "worker_io", "agent"):
            allow(f"qdrant-from-{src}", self.qdrant, 6333, getattr(self, src))
        for src in ("qdrant", "debezium"):
            allow(f"efs-from-{src}", self.efs, 2049, getattr(self, src))
        # Postgres: API (jobs + LISTEN), IO workers, Temporal, Debezium. CPU workers: never.
        for src in ("api", "worker_io", "temporal", "debezium"):
            allow(f"rds-from-{src}", self.rds, 5432, getattr(self, src))

        self.register_outputs({})
