"""PDF name extractor — AWS infrastructure.

INFRA_OFFLINE=1 pulumi preview --stack dev     # plan without an AWS account (see README)
"""

import pulumi
import pulumi_aws as aws

import config
from components.app_services import Agent, AppServices
from components.data import ArtifactsBucket, Database, FileSystem, Registry
from components.network import Network, SecurityGroups
from components.observability import Alarms
from components.outbox import Outbox
from components.platform import Cluster, LoadBalancer, Platform

cfg = config.load()

# Not a credential: a placeholder that lets the provider plan without an AWS account.
OFFLINE_PLACEHOLDER = "offline"  # pragma: allowlist secret

# Explicit provider. Offline mode plans with dummy credentials and skips every call that needs a
# real account (credential validation, account-id lookup, metadata API) — the program uses no
# data sources, so a full preview needs no AWS access.
provider = aws.Provider(
    "aws",
    region=cfg.region,
    default_tags=aws.ProviderDefaultTagsArgs(tags=cfg.tags),
    **(
        dict(
            access_key=OFFLINE_PLACEHOLDER,
            secret_key=OFFLINE_PLACEHOLDER,
            skip_credentials_validation=True,
            skip_requesting_account_id=True,
            skip_metadata_api_check=True,
            skip_region_validation=True,
        )
        if cfg.offline
        else {}
    ),
)
opts = pulumi.ResourceOptions(providers={"aws": provider})

network = Network(cfg, opts=opts)
sgs = SecurityGroups(cfg, network.vpc.id, opts=opts)
registry = Registry(cfg, opts=opts)
bucket = ArtifactsBucket(cfg, opts=opts)
db = Database(cfg, [s.id for s in network.private_subnets], sgs.rds.id, opts=opts)  # never internet-facing
efs = FileSystem(cfg, network.app_subnet_ids, sgs.efs.id, opts=opts)
cluster = Cluster(cfg, network, opts=opts)
alb = LoadBalancer(cfg, network, sgs, opts=opts)
platform = Platform(cfg, network, sgs, cluster, alb, db, efs, opts=opts)
agent = Agent(cfg, network, sgs, cluster, registry, opts=opts)
apps = AppServices(cfg, network, sgs, cluster, alb, registry, db, bucket, agent, opts=opts)
outbox = Outbox(cfg, network, sgs, cluster, registry, db, efs, opts=opts) if cfg.outbox_enabled else None
alarms = Alarms(cfg, db, alb, outbox, opts=opts)

pulumi.export("api_url", alb.alb.dns_name.apply(lambda d: f"http{'s' if cfg.certificate_arn else ''}://{d}"))
pulumi.export("temporal_ui_url", alb.alb.dns_name.apply(lambda d: f"http://{d}:8233"))
pulumi.export("ecr_repositories", {k: r.repository_url for k, r in registry.repos.items()})
pulumi.export("artifacts_bucket", bucket.bucket.bucket)
pulumi.export("db_endpoint", db.instance.address)
pulumi.export("agent_runtime_arn", agent.runtime.agent_runtime_arn)
pulumi.export("alarm_topic", alarms.topic.arn)
if outbox is not None:
    pulumi.export("outbox_stream", outbox.stream.name)
    pulumi.export("dispatch_dlq", outbox.dlq.url)
