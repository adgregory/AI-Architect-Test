"""Typed stack configuration (Pulumi.<stack>.yaml), validated with pydantic."""

from __future__ import annotations

import json
import os

import pulumi
from pydantic import BaseModel, Field, field_validator

PROJECT = "pdf-name-extractor"


class Size(BaseModel):
    cpu: int = Field(description="Fargate CPU units (256 = 0.25 vCPU)")
    memory: int = Field(description="MiB")
    desired_count: int = Field(1, ge=0, alias="desiredCount")
    concurrency: int = Field(1, ge=1)

    @field_validator("cpu")
    @classmethod
    def fargate_cpu(cls, v: int) -> int:
        if v not in (256, 512, 1024, 2048, 4096, 8192, 16384):
            raise ValueError(f"{v} is not a Fargate CPU size")
        return v


class RdsConfig(BaseModel):
    instance_class: str = Field(alias="instanceClass")
    allocated_storage_gb: int = Field(alias="allocatedStorageGb", ge=20)
    multi_az: bool = Field(alias="multiAz")
    max_slot_wal_keep_size_mb: int = Field(alias="maxSlotWalKeepSizeMb", ge=256)


class StackConfig(BaseModel):
    env: str
    region: str
    availability_zones: list[str] = Field(alias="availabilityZones", min_length=2)
    vpc_cidr: str = Field(alias="vpcCidr")
    nat_gateway: bool = Field(alias="natGateway")
    admin_cidrs: list[str] = Field(alias="adminCidrs")
    certificate_arn: str = Field("", alias="certificateArn")
    image_tag: str = Field(alias="imageTag")
    rds: RdsConfig
    api: Size
    worker_cpu: Size = Field(alias="workerCpu")
    worker_io: Size = Field(alias="workerIo")
    temporal: Size
    qdrant: Size
    debezium: Size
    bedrock_model_id: str = Field(alias="bedrockModelId")
    kinesis_shards: int = Field(alias="kinesisShards", ge=1)
    log_retention_days: int = Field(alias="logRetentionDays")
    outbox_enabled: bool = Field(False, alias="outboxEnabled")
    offline: bool = False

    @field_validator("admin_cidrs")
    @classmethod
    def no_open_admin(cls, v: list[str]) -> list[str]:
        if "0.0.0.0/0" in v:
            raise ValueError("adminCidrs must not include 0.0.0.0/0 (the Temporal UI has no auth)")
        return v

    @property
    def name(self) -> str:
        return f"{PROJECT}-{self.env}"

    @property
    def short(self) -> str:
        """For resources with tight name limits (ALB / target group: 32 chars)."""
        return f"pdfx-{self.env}"

    @property
    def tags(self) -> dict[str, str]:
        return {"project": PROJECT, "env": self.env, "managed-by": "pulumi"}


def load() -> StackConfig:
    cfg = pulumi.Config(PROJECT)
    keys = [
        "env",
        "availabilityZones",
        "vpcCidr",
        "natGateway",
        "adminCidrs",
        "certificateArn",
        "imageTag",
        "rds",
        "api",
        "workerCpu",
        "workerIo",
        "temporal",
        "qdrant",
        "debezium",
        "bedrockModelId",
        "kinesisShards",
        "logRetentionDays",
        "outboxEnabled",
    ]
    raw: dict = {}
    for key in keys:
        value = cfg.get(key)
        if value is None:
            continue
        try:  # objects, lists, numbers and booleans are stored as JSON; plain strings are not
            raw[key] = json.loads(value)
        except json.JSONDecodeError:
            raw[key] = value
    raw["region"] = pulumi.Config("aws").require("region")
    # INFRA_OFFLINE=1: plan without AWS credentials (dummy keys, no account/metadata calls).
    raw["offline"] = os.environ.get("INFRA_OFFLINE") == "1"
    return StackConfig.model_validate({k: v for k, v in raw.items() if v is not None})
