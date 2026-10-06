"""Policy rules as plain functions: used by the CrossGuard pack (preview/up) and by the unit tests
(which can evaluate values that are still unknown during a preview, such as IAM documents with ARNs).

Each rule takes an object with `.resource_type` and `.props`, and a `report(message)` callable.
"""

import json

# AWS actions that only accept "*" as the resource.
RESOURCE_STAR_ALLOWED = {"ecr:GetAuthorizationToken", "kinesis:ListStreams"}


def _s3_no_public_access(args, report):
    if args.resource_type != "aws:s3/bucketPublicAccessBlock:BucketPublicAccessBlock":
        return
    for key in ("blockPublicAcls", "blockPublicPolicy", "ignorePublicAcls", "restrictPublicBuckets"):
        if args.props.get(key) is not True:
            report(f"S3 public access block must set {key}=true")


def _rds_encrypted_private(args, report):
    if args.resource_type != "aws:rds/instance:Instance":
        return
    if not args.props.get("storageEncrypted"):
        report("RDS storage must be encrypted")
    if args.props.get("publiclyAccessible"):
        report("RDS must not be publicly accessible")
    if args.props.get("password"):
        report("Use manageMasterUserPassword (Secrets Manager) instead of a literal password")


def _no_open_ingress(args, report):
    if args.resource_type != "aws:vpc/securityGroupIngressRule:SecurityGroupIngressRule":
        return
    if args.props.get("cidrIpv4") == "0.0.0.0/0" and int(args.props.get("fromPort", -1)) not in (80, 443):
        report(f"0.0.0.0/0 ingress is only allowed on 80/443 (got port {args.props.get('fromPort')})")


def _ecr_scan_on_push(args, report):
    if args.resource_type != "aws:ecr/repository:Repository":
        return
    if not (args.props.get("imageScanningConfiguration") or {}).get("scanOnPush"):
        report("ECR repositories must scan images on push")


def _iam_least_privilege(args, report):
    if args.resource_type != "aws:iam/rolePolicy:RolePolicy" or not isinstance(args.props.get("policy"), str):
        return
    for statement in json.loads(args.props["policy"]).get("Statement", []):
        if statement.get("Effect") != "Allow":
            continue
        actions = statement.get("Action", [])
        actions = [actions] if isinstance(actions, str) else actions
        resources = statement.get("Resource", [])
        resources = [resources] if isinstance(resources, str) else resources
        if any(a == "*" or a.endswith(":*") for a in actions):
            report(f"Wildcard IAM action is not allowed: {actions}")
        if "*" in resources and not set(actions) <= RESOURCE_STAR_ALLOWED:
            report(f"Resource '*' is only allowed for {sorted(RESOURCE_STAR_ALLOWED)}; got {actions}")


def _kinesis_partial_batch_failures(args, report):
    if args.resource_type != "aws:lambda/eventSourceMapping:EventSourceMapping":
        return
    if "ReportBatchItemFailures" not in (args.props.get("functionResponseTypes") or []):
        report("Stream consumers must report partial batch failures (ReportBatchItemFailures)")
    if not (args.props.get("destinationConfig") or {}).get("onFailure"):
        report("Stream consumers need an on-failure destination (DLQ)")
