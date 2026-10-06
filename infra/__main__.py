"""PDF name extractor — AWS infrastructure.

INFRA_OFFLINE=1 pulumi preview --stack dev     # plan without an AWS account (see README)
"""

import pulumi
from stack import build

import config

cfg = config.load()
r = build(cfg)
alb, registry, bucket, db, agent, alarms, outbox = (
    r["alb"],
    r["registry"],
    r["bucket"],
    r["db"],
    r["agent"],
    r["alarms"],
    r["outbox"],
)

pulumi.export("api_url", alb.alb.dns_name.apply(lambda d: f"http{'s' if cfg.certificate_arn else ''}://{d}"))
pulumi.export("temporal_ui_url", alb.alb.dns_name.apply(lambda d: f"http://{d}:8233"))
pulumi.export("ecr_repositories", {name: repo.repository_url for name, repo in registry.repos.items()})
pulumi.export("artifacts_bucket", bucket.bucket.bucket)
pulumi.export("db_endpoint", db.instance.address)
pulumi.export("agent_runtime_arn", agent.runtime.agent_runtime_arn)
pulumi.export("alarm_topic", alarms.topic.arn)
if outbox is not None:
    pulumi.export("outbox_stream", outbox.stream.name)
    pulumi.export("dispatch_dlq", outbox.dlq.url)
