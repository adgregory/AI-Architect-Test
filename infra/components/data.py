"""Stateful resources: ECR repositories, the artefacts bucket, Postgres, and EFS."""

from __future__ import annotations

import json

import pulumi
import pulumi_aws as aws

from config import StackConfig

IMAGES = ("app", "agent", "dispatcher")  # api + workers share the app image


class Registry(pulumi.ComponentResource):
    def __init__(self, cfg: StackConfig, opts: pulumi.ResourceOptions | None = None):
        super().__init__("pdfx:data:Registry", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)
        self.repos: dict[str, aws.ecr.Repository] = {}
        for image in IMAGES:
            repo = aws.ecr.Repository(
                f"{cfg.name}-{image}",
                name=f"{cfg.name}/{image}",
                image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(scan_on_push=True),
                encryption_configurations=[aws.ecr.RepositoryEncryptionConfigurationArgs(encryption_type="AES256")],
                image_tag_mutability="MUTABLE" if cfg.env == "dev" else "IMMUTABLE",
                force_delete=cfg.env == "dev",
                tags=cfg.tags,
                opts=child,
            )
            aws.ecr.LifecyclePolicy(
                f"{cfg.name}-{image}-lifecycle",
                repository=repo.name,
                policy=json.dumps(
                    {
                        "rules": [
                            {
                                "rulePriority": 1,
                                "description": "keep the last 20 images",
                                "selection": {"tagStatus": "any", "countType": "imageCountMoreThan", "countNumber": 20},
                                "action": {"type": "expire"},
                            }
                        ]
                    }
                ),
                opts=child,
            )
            self.repos[image] = repo
        self.register_outputs({k: v.repository_url for k, v in self.repos.items()})

    def image(self, name: str, tag: str) -> pulumi.Output[str]:
        return pulumi.Output.concat(self.repos[name].repository_url, ":", tag)


class ArtifactsBucket(pulumi.ComponentResource):
    """Uploaded PDFs, per-page OCR results and extraction results (keys under jobs/<id>/)."""

    def __init__(self, cfg: StackConfig, opts: pulumi.ResourceOptions | None = None):
        super().__init__("pdfx:data:ArtifactsBucket", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)
        self.bucket = aws.s3.Bucket(f"{cfg.name}-artifacts", force_destroy=cfg.env == "dev", tags=cfg.tags, opts=child)
        aws.s3.BucketPublicAccessBlock(
            f"{cfg.name}-artifacts-pab",
            bucket=self.bucket.id,
            block_public_acls=True,
            block_public_policy=True,
            ignore_public_acls=True,
            restrict_public_buckets=True,
            opts=child,
        )
        aws.s3.BucketServerSideEncryptionConfiguration(
            f"{cfg.name}-artifacts-sse",
            bucket=self.bucket.id,
            rules=[
                aws.s3.BucketServerSideEncryptionConfigurationRuleArgs(
                    apply_server_side_encryption_by_default=aws.s3.BucketServerSideEncryptionConfigurationRuleApplyServerSideEncryptionByDefaultArgs(
                        sse_algorithm="AES256"
                    )
                )
            ],
            opts=child,
        )
        aws.s3.BucketVersioning(
            f"{cfg.name}-artifacts-versioning",
            bucket=self.bucket.id,
            versioning_configuration=aws.s3.BucketVersioningVersioningConfigurationArgs(status="Enabled"),
            opts=child,
        )
        aws.s3.BucketLifecycleConfiguration(
            f"{cfg.name}-artifacts-lifecycle",
            bucket=self.bucket.id,
            rules=[
                aws.s3.BucketLifecycleConfigurationRuleArgs(
                    id="expire-job-artifacts",
                    status="Enabled",
                    filter=aws.s3.BucketLifecycleConfigurationRuleFilterArgs(prefix="jobs/"),
                    expiration=aws.s3.BucketLifecycleConfigurationRuleExpirationArgs(days=30),
                    noncurrent_version_expiration=aws.s3.BucketLifecycleConfigurationRuleNoncurrentVersionExpirationArgs(
                        noncurrent_days=7
                    ),
                    abort_incomplete_multipart_upload=aws.s3.BucketLifecycleConfigurationRuleAbortIncompleteMultipartUploadArgs(
                        days_after_initiation=1
                    ),
                ),
            ],
            opts=child,
        )
        aws.s3.BucketPolicy(
            f"{cfg.name}-artifacts-tls-only",
            bucket=self.bucket.id,
            policy=self.bucket.arn.apply(
                lambda arn: json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Sid": "DenyInsecureTransport",
                                "Effect": "Deny",
                                "Principal": "*",
                                "Action": "s3:*",
                                "Resource": [arn, f"{arn}/*"],
                                "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                            }
                        ],
                    }
                )
            ),
            opts=child,
        )
        self.register_outputs({"bucket": self.bucket.bucket})

    def rw_statements(self) -> list[dict]:
        return [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": pulumi.Output.concat(self.bucket.arn, "/jobs/*"),
            },
            {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": self.bucket.arn},
        ]


class Database(pulumi.ComponentResource):
    """Postgres for jobs and Temporal persistence. Master password managed by RDS in Secrets
    Manager. Logical replication is enabled for the Debezium outbox target, with a WAL cap so a
    stalled replication slot can't fill the disk."""

    def __init__(self, cfg: StackConfig, subnet_ids, security_group_id, opts: pulumi.ResourceOptions | None = None):
        super().__init__("pdfx:data:Database", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)
        subnet_group = aws.rds.SubnetGroup(f"{cfg.name}-db-subnets", subnet_ids=subnet_ids, tags=cfg.tags, opts=child)
        params = aws.rds.ParameterGroup(
            f"{cfg.name}-pg16",
            family="postgres16",
            tags=cfg.tags,
            parameters=[
                aws.rds.ParameterGroupParameterArgs(
                    name="rds.logical_replication", value="1", apply_method="pending-reboot"
                ),
                aws.rds.ParameterGroupParameterArgs(
                    name="max_slot_wal_keep_size", value=str(cfg.rds.max_slot_wal_keep_size_mb)
                ),
                aws.rds.ParameterGroupParameterArgs(name="max_connections", value="200", apply_method="pending-reboot"),
                aws.rds.ParameterGroupParameterArgs(name="idle_in_transaction_session_timeout", value="30000"),
            ],
            opts=child,
        )
        production_like = cfg.env != "dev"
        self.instance = aws.rds.Instance(
            f"{cfg.name}-db",
            identifier=f"{cfg.name}-db",
            engine="postgres",
            engine_version="16",
            instance_class=cfg.rds.instance_class,
            allocated_storage=cfg.rds.allocated_storage_gb,
            max_allocated_storage=cfg.rds.allocated_storage_gb * 4,
            storage_type="gp3",
            storage_encrypted=True,
            db_name="app",
            username="app",
            manage_master_user_password=True,
            db_subnet_group_name=subnet_group.name,
            vpc_security_group_ids=[security_group_id],
            parameter_group_name=params.name,
            publicly_accessible=False,
            multi_az=cfg.rds.multi_az,
            backup_retention_period=7 if production_like else 1,
            deletion_protection=production_like,
            skip_final_snapshot=not production_like,
            final_snapshot_identifier=f"{cfg.name}-db-final" if production_like else None,
            performance_insights_enabled=True,
            auto_minor_version_upgrade=True,
            copy_tags_to_snapshot=True,
            tags=cfg.tags,
            opts=child,
        )
        self.secret_arn = self.instance.master_user_secrets.apply(lambda s: s[0]["secret_arn"])
        self.register_outputs({"endpoint": self.instance.address})


class FileSystem(pulumi.ComponentResource):
    """Encrypted EFS with one access point per stateful service (Qdrant data, Debezium offsets)."""

    def __init__(self, cfg: StackConfig, subnet_ids, security_group_id, opts: pulumi.ResourceOptions | None = None):
        super().__init__("pdfx:data:FileSystem", cfg.name, None, opts)
        child = pulumi.ResourceOptions(parent=self)
        self.fs = aws.efs.FileSystem(
            f"{cfg.name}-efs", encrypted=True, tags={**cfg.tags, "Name": f"{cfg.name}-efs"}, opts=child
        )
        for i, subnet_id in enumerate(subnet_ids):
            aws.efs.MountTarget(
                f"{cfg.name}-efs-mt-{i}",
                file_system_id=self.fs.id,
                subnet_id=subnet_id,
                security_groups=[security_group_id],
                opts=child,
            )
        self.access_points = {
            name: aws.efs.AccessPoint(
                f"{cfg.name}-efs-{name}",
                file_system_id=self.fs.id,
                posix_user=aws.efs.AccessPointPosixUserArgs(uid=1000, gid=1000),
                root_directory=aws.efs.AccessPointRootDirectoryArgs(
                    path=f"/{name}",
                    creation_info=aws.efs.AccessPointRootDirectoryCreationInfoArgs(
                        owner_uid=1000, owner_gid=1000, permissions="750"
                    ),
                ),
                tags=cfg.tags,
                opts=child,
            )
            for name in ("qdrant", "debezium")
        }
        self.register_outputs({"file_system_id": self.fs.id})

    def volume(self, name: str, container_path: str) -> dict:
        return {
            "file_system_id": self.fs.id,
            "access_point_id": self.access_points[name].id,
            "container_path": container_path,
        }

    def client_statements(self, name: str) -> list[dict]:
        return [
            {
                "Effect": "Allow",
                "Action": ["elasticfilesystem:ClientMount", "elasticfilesystem:ClientWrite"],
                "Resource": self.fs.arn,
                "Condition": {"StringEquals": {"elasticfilesystem:AccessPointArn": self.access_points[name].arn}},
            }
        ]
