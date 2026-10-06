"""CrossGuard policies, enforced (mandatory) on every preview/update of the stack."""

from pulumi_policy import EnforcementLevel, PolicyPack, ResourceValidationPolicy
from rules import (
    _ecr_scan_on_push,
    _iam_least_privilege,
    _kinesis_partial_batch_failures,
    _no_open_ingress,
    _rds_encrypted_private,
    _s3_no_public_access,
)

PolicyPack(
    name="pdf-name-extractor-guardrails",
    enforcement_level=EnforcementLevel.MANDATORY,
    policies=[
        ResourceValidationPolicy(name=name, description=description, validate=fn)
        for name, description, fn in [
            ("s3-no-public-access", "S3 buckets block all public access", _s3_no_public_access),
            ("rds-encrypted-private", "RDS is encrypted, private and uses managed passwords", _rds_encrypted_private),
            ("no-open-ingress", "Internet ingress only on 80/443", _no_open_ingress),
            ("ecr-scan-on-push", "Container images are scanned on push", _ecr_scan_on_push),
            (
                "iam-least-privilege",
                "No wildcard IAM actions; '*' resources only where AWS requires",
                _iam_least_privilege,
            ),
            (
                "kinesis-partial-batch-failures",
                "Stream consumers report partial failures and have a DLQ",
                _kinesis_partial_batch_failures,
            ),
        ]
    ],
)
