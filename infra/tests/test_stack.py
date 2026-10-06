"""Policy-level assertions on the planned infrastructure (Pulumi mocks, no AWS access)."""

import json

import pulumi
import pytest
import rules
from pydantic import ValidationError

from tests.conftest import stack_config

SG_INGRESS = "aws:vpc/securityGroupIngressRule:SecurityGroupIngressRule"


def build_stack(stack: str):
    from stack import build

    cfg = stack_config(stack)
    return cfg, build(cfg)


def settle(result: dict):
    """An output that resolves once every component has registered its resources."""
    return pulumi.Output.all(result["alarms"].topic.arn, result["apps"].api.service.name)


@pulumi.runtime.test
def test_cpu_workers_have_no_database_path(mocks):
    _, result = build_stack("dev")

    def check(_):
        rds_rules = [r for r in mocks.of_type(SG_INGRESS) if r.name.startswith("pdf-name-extractor-dev-rds-from")]
        sources = sorted(r.name.removeprefix("pdf-name-extractor-dev-rds-from-") for r in rds_rules)
        assert sources == ["api", "debezium", "temporal", "worker_io"]
        assert all(r.inputs["fromPort"] == 5432 for r in rds_rules)

    return settle(result).apply(check)


@pulumi.runtime.test
def test_only_the_public_listener_is_open_to_the_internet(mocks):
    cfg, result = build_stack("dev")

    def check(_):
        open_rules = [r for r in mocks.of_type(SG_INGRESS) if r.inputs.get("cidrIpv4") == "0.0.0.0/0"]
        assert [(r.name, int(r.inputs["fromPort"])) for r in open_rules] == [
            ("pdf-name-extractor-dev-alb-from-internet-0", 80)
        ]
        ui_rules = [r for r in mocks.of_type(SG_INGRESS) if r.name.startswith("pdf-name-extractor-dev-alb-ui-from")]
        assert {r.inputs["cidrIpv4"] for r in ui_rules} == set(cfg.admin_cidrs)

    return settle(result).apply(check)


@pulumi.runtime.test
def test_artifacts_bucket_is_private_and_tls_only(mocks):
    _, result = build_stack("dev")

    def check(_):
        [pab] = mocks.of_type("aws:s3/bucketPublicAccessBlock:BucketPublicAccessBlock")
        assert all(
            pab.inputs[k] for k in ("blockPublicAcls", "blockPublicPolicy", "ignorePublicAcls", "restrictPublicBuckets")
        )
        [policy] = mocks.of_type("aws:s3/bucketPolicy:BucketPolicy")
        statement = json.loads(policy.inputs["policy"])["Statement"][0]
        assert statement["Effect"] == "Deny" and statement["Condition"] == {"Bool": {"aws:SecureTransport": "false"}}
        assert mocks.of_type("aws:s3/bucketServerSideEncryptionConfiguration:BucketServerSideEncryptionConfiguration")

    return settle(result).apply(check)


@pytest.mark.parametrize("stack, protected, multi_az", [("dev", False, False), ("staging", True, True)])
@pulumi.runtime.test
def test_database_is_private_encrypted_and_managed(mocks, stack, protected, multi_az):
    _, result = build_stack(stack)

    def check(_):
        [db] = mocks.of_type("aws:rds/instance:Instance")
        assert db.inputs["storageEncrypted"] and not db.inputs["publiclyAccessible"]
        assert db.inputs["manageMasterUserPassword"] and "password" not in db.inputs
        assert db.inputs["deletionProtection"] is protected and db.inputs["multiAz"] is multi_az
        [params] = mocks.of_type("aws:rds/parameterGroup:ParameterGroup")
        names = {p["name"]: p["value"] for p in params.inputs["parameters"]}
        assert names["rds.logical_replication"] == "1" and int(names["max_slot_wal_keep_size"]) > 0

    return settle(result).apply(check)


@pulumi.runtime.test
def test_agent_may_invoke_only_the_configured_model(mocks):
    cfg, result = build_stack("dev")

    def check(_):
        [policy] = [r for r in mocks.of_type("aws:iam/rolePolicy:RolePolicy") if r.name.endswith("-agent-policy")]
        statements = {s["Sid"]: s for s in json.loads(policy.inputs["policy"])["Statement"]}
        invoke = statements["InvokeConfiguredModelOnly"]
        assert invoke["Resource"] == f"arn:aws:bedrock:{cfg.region}::foundation-model/{cfg.bedrock_model_id}"
        assert set(invoke["Action"]) == {"bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"}
        [runtime] = mocks.of_type("aws:bedrock/agentcoreAgentRuntime:AgentcoreAgentRuntime")
        assert runtime.inputs["networkConfiguration"]["networkMode"] == "VPC"
        assert runtime.inputs["environmentVariables"]["LLM_PROVIDER"] == "bedrock"

    return settle(result).apply(check)


@pytest.mark.parametrize("stack, enabled", [("dev", False), ("staging", True)])
@pulumi.runtime.test
def test_outbox_dispatch_is_resilient_when_enabled(mocks, stack, enabled):
    _, result = build_stack(stack)

    def check(_):
        mappings = mocks.of_type("aws:lambda/eventSourceMapping:EventSourceMapping")
        assert bool(mappings) is enabled
        if enabled:
            [esm] = mappings
            assert esm.inputs["functionResponseTypes"] == ["ReportBatchItemFailures"]
            assert esm.inputs["bisectBatchOnFunctionError"] is True
            assert esm.inputs["maximumRetryAttempts"] == 5
            assert "onFailure" in esm.inputs["destinationConfig"]
            assert [
                a.name
                for a in mocks.of_type("aws:cloudwatch/metricAlarm:MetricAlarm")
                if a.name.endswith("replication-slot-lag")
            ]

    return settle(result).apply(check)


@pytest.mark.parametrize("stack, nat", [("dev", 0), ("staging", 1)])
@pulumi.runtime.test
def test_nat_only_where_configured(mocks, stack, nat):
    _, result = build_stack(stack)
    return settle(result).apply(
        lambda _: (
            len(mocks.of_type("aws:ec2/natGateway:NatGateway")) == nat or pytest.fail("unexpected NAT gateway count")
        )
    )


@pulumi.runtime.test
def test_worker_sizes_come_from_stack_config(mocks):
    cfg, result = build_stack("staging")

    def check(_):
        tasks = {r.inputs["family"]: r.inputs for r in mocks.of_type("aws:ecs/taskDefinition:TaskDefinition")}
        cpu_task = tasks["pdf-name-extractor-staging-worker-cpu"]
        assert (cpu_task["cpu"], cpu_task["memory"]) == (str(cfg.worker_cpu.cpu), str(cfg.worker_cpu.memory))
        env = {e["name"]: e["value"] for e in json.loads(cpu_task["containerDefinitions"])[0]["environment"]}
        assert env["CPU_WORKER_CONCURRENCY"] == str(cfg.worker_cpu.concurrency)
        assert "DB_PASSWORD" not in cpu_task["containerDefinitions"]  # no DB credentials for CPU workers

    return settle(result).apply(check)


def test_config_rejects_open_admin_cidr_and_invalid_sizes():
    from config import StackConfig

    base = stack_config("dev").model_dump(by_alias=True)
    with pytest.raises(ValidationError, match="adminCidrs"):
        StackConfig.model_validate({**base, "adminCidrs": ["0.0.0.0/0"]})
    with pytest.raises(ValidationError, match="Fargate CPU"):
        StackConfig.model_validate({**base, "workerCpu": {"cpu": 3000, "memory": 4096}})


@pytest.mark.parametrize("stack", ["dev", "staging"])
@pulumi.runtime.test
def test_every_resource_passes_the_policy_rules(mocks, stack):
    """The CrossGuard rules, evaluated on fully known values (a preview can't see IAM documents
    whose ARNs don't exist yet; the mocks can)."""
    checks = [
        rules._s3_no_public_access,
        rules._rds_encrypted_private,
        rules._no_open_ingress,
        rules._ecr_scan_on_push,
        rules._iam_least_privilege,
        rules._kinesis_partial_batch_failures,
    ]
    _, result = build_stack(stack)

    def check(_):
        violations = []
        for resource in mocks.resources:
            args = type("Args", (), {"resource_type": resource.typ, "props": resource.inputs})()
            for rule in checks:
                rule(args, lambda msg, r=resource: violations.append(f"{r.name}: {msg}"))
        assert violations == []
        assert len([r for r in mocks.resources if r.typ == "aws:iam/rolePolicy:RolePolicy"]) >= 8

    return settle(result).apply(check)


@pytest.mark.parametrize(
    "typ, props, expected",
    [
        (
            "aws:iam/rolePolicy:RolePolicy",
            {
                "policy": json.dumps(
                    {"Statement": [{"Effect": "Allow", "Action": "s3:*", "Resource": "arn:aws:s3:::b/*"}]}
                )
            },
            "Wildcard IAM action",
        ),
        (
            "aws:iam/rolePolicy:RolePolicy",
            {"policy": json.dumps({"Statement": [{"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "*"}]})},
            "Resource '*'",
        ),
        ("aws:rds/instance:Instance", {"storageEncrypted": True, "publiclyAccessible": True}, "publicly accessible"),
        (
            "aws:vpc/securityGroupIngressRule:SecurityGroupIngressRule",
            {"cidrIpv4": "0.0.0.0/0", "fromPort": 5432},
            "only allowed on 80/443",
        ),
        ("aws:s3/bucketPublicAccessBlock:BucketPublicAccessBlock", {"blockPublicAcls": True}, "blockPublicPolicy"),
        ("aws:lambda/eventSourceMapping:EventSourceMapping", {"functionResponseTypes": []}, "partial batch failures"),
    ],
)
def test_rules_flag_violations(typ, props, expected):
    messages = []
    args = type("Args", (), {"resource_type": typ, "props": props})()
    for rule in (
        rules._s3_no_public_access,
        rules._rds_encrypted_private,
        rules._no_open_ingress,
        rules._iam_least_privilege,
        rules._kinesis_partial_batch_failures,
    ):
        rule(args, messages.append)
    assert any(expected in m for m in messages), messages
