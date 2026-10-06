"""Pulumi mock runtime: programs run fully offline and every resource's inputs are recorded."""

import sys
from pathlib import Path

import pulumi
import pytest
import yaml

INFRA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFRA))
sys.path.insert(0, str(INFRA / "policies"))  # policy rules are plain functions, shared with the pack


class RecordingMocks(pulumi.runtime.Mocks):
    def __init__(self):
        self.resources: list[pulumi.runtime.MockResourceArgs] = []

    def new_resource(self, args):
        self.resources.append(args)
        outputs = {
            **args.inputs,
            "arn": f"arn:aws:mock:us-east-1:000000000000:{args.name}",
            "name": args.inputs.get("name", args.name),
            "id": f"{args.name}-id",
        }
        if args.typ == "aws:rds/instance:Instance":
            outputs["masterUserSecrets"] = [{"secretArn": f"arn:aws:secretsmanager:us-east-1:0:secret:{args.name}"}]
            outputs["address"] = f"{args.name}.mock.rds.amazonaws.com"
        if args.typ == "aws:ecr/repository:Repository":
            outputs["repositoryUrl"] = f"000000000000.dkr.ecr.us-east-1.amazonaws.com/{args.inputs['name']}"
        if args.typ == "aws:bedrock/agentcoreAgentRuntime:AgentcoreAgentRuntime":
            outputs["agentRuntimeArn"] = f"arn:aws:bedrock-agentcore:us-east-1:0:runtime/{args.name}"
        return [f"{args.name}-id", outputs]

    def call(self, args):
        return {}

    def of_type(self, typ: str) -> list:
        return [r for r in self.resources if r.typ == typ]


def stack_config(stack: str):
    from config import StackConfig

    raw = yaml.safe_load((INFRA / f"Pulumi.{stack}.yaml").read_text())["config"]
    values = {k.split(":", 1)[1]: v for k, v in raw.items() if k.startswith("pdf-name-extractor:")}
    values["region"] = raw["aws:region"]
    return StackConfig.model_validate(values)


@pytest.fixture
def mocks():
    m = RecordingMocks()
    pulumi.runtime.set_mocks(m, project="pdf-name-extractor", stack="test", preview=False)
    return m
