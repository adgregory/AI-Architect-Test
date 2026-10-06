"""Alembic environment: plain SQL migrations (no ORM models); URL from application Settings."""

from alembic import context
from sqlalchemy import create_engine, pool

from app.core.config import get_settings


def sqlalchemy_url() -> str:
    url = get_settings().database_url.get_secret_value()
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


def run_migrations_offline() -> None:
    context.configure(url=sqlalchemy_url(), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(sqlalchemy_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
