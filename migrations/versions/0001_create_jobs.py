"""Create the jobs table.

Revision ID: 0001
Revises:
Create Date: 2026-10-06
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE TYPE job_status AS ENUM ('queued', 'running', 'succeeded', 'failed')")
    op.execute("""
        CREATE TABLE jobs (
            id           uuid PRIMARY KEY,
            status       job_status  NOT NULL DEFAULT 'queued',
            filename     text        NOT NULL,
            input_key    text        NOT NULL,
            query_names  jsonb       NOT NULL DEFAULT '[]'::jsonb,
            page_count   integer,
            result       jsonb,
            error        text,
            attempts     integer     NOT NULL DEFAULT 0,
            created_at   timestamptz NOT NULL DEFAULT now(),
            updated_at   timestamptz NOT NULL DEFAULT now(),
            started_at   timestamptz,
            finished_at  timestamptz
        )
    """)
    # The reconciler scans for stale queued jobs.
    op.execute("CREATE INDEX jobs_status_created_at_idx ON jobs (status, created_at)")


def downgrade() -> None:
    op.execute("DROP TABLE jobs")
    op.execute("DROP TYPE job_status")
