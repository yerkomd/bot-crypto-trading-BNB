import os
import time
import threading
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


def _env_required(name: str) -> str:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return str(value).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    minconn: int = 1
    maxconn: int = 8
    connect_timeout_s: int = 5


def load_postgres_config_from_env() -> PostgresConfig:
    if not _env_bool("USE_DATABASE", False):
        raise RuntimeError("USE_DATABASE must be true. CSV/file persistence is disabled.")

    return PostgresConfig(
        host=_env_required("POSTGRES_HOST"),
        port=_env_int("POSTGRES_PORT", 5432),
        dbname=_env_required("POSTGRES_DB"),
        user=_env_required("POSTGRES_USER"),
        password=_env_required("POSTGRES_PASSWORD"),
        minconn=_env_int("POSTGRES_POOL_MIN", 1),
        maxconn=_env_int("POSTGRES_POOL_MAX", 8),
        connect_timeout_s=_env_int("POSTGRES_CONNECT_TIMEOUT", 5),
    )


class PostgresDatabase:
    """Thread-safe psycopg2 pool with reconnect + explicit transactions.

    Design goals:
    - Never crash the bot due to DB outages (callers can opt-in to swallow errors).
    - Explicit commit/rollback.
    - One pool shared by all bot threads.
    """

    def __init__(self, cfg: PostgresConfig):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._pool = None

    def initialize(self) -> None:
        self._ensure_pool()

    def close(self) -> None:
        with self._lock:
            pool = self._pool
            self._pool = None
        if pool is not None:
            try:
                pool.closeall()
            except Exception:
                pass

    def _ensure_pool(self) -> None:
        with self._lock:
            if self._pool is not None:
                return
            self._pool = self._create_pool()

    def _reset_pool(self) -> None:
        with self._lock:
            old = self._pool
            self._pool = self._create_pool()
        if old is not None:
            try:
                old.closeall()
            except Exception:
                pass

    def _create_pool(self):
        try:
            import psycopg2
            from psycopg2.pool import ThreadedConnectionPool
        except Exception as e:
            raise RuntimeError(
                "psycopg2 is required for PostgreSQL storage. Install psycopg2-binary."
            ) from e

        dsn = (
            f"host={self._cfg.host} port={self._cfg.port} dbname={self._cfg.dbname} "
            f"user={self._cfg.user} password={self._cfg.password} connect_timeout={self._cfg.connect_timeout_s} "
            "keepalives=1 keepalives_idle=30 keepalives_interval=10 keepalives_count=5"
        )

        logger.info(
            "Initializing PostgreSQL pool host=%s port=%s db=%s user=%s min=%s max=%s",
            self._cfg.host,
            self._cfg.port,
            self._cfg.dbname,
            self._cfg.user,
            self._cfg.minconn,
            self._cfg.maxconn,
        )
        return ThreadedConnectionPool(self._cfg.minconn, self._cfg.maxconn, dsn=dsn)

    def run(
        self,
        fn: Callable[[Any], T],
        *,
        retries: int = 1,
        swallow: bool = True,
        backoff_base: float = 1.7,
    ) -> Optional[T]:
        """Runs `fn(cursor)` inside an explicit transaction.

        - Retries on OperationalError (reconnects pool).
        - Rolls back on any exception.
        - When swallow=True, returns None on final failure.
        """
        self._ensure_pool()

        try:
            import psycopg2
            from psycopg2 import OperationalError
            from psycopg2.extras import RealDictCursor
        except Exception as e:
            raise RuntimeError("psycopg2 missing") from e

        attempt = 0
        while True:
            attempt += 1
            conn = None
            try:
                conn = self._pool.getconn()  # type: ignore[union-attr]
                conn.autocommit = False
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    result = fn(cur)
                conn.commit()
                return result
            except OperationalError as oe:
                if conn is not None:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    try:
                        conn.close()
                    except Exception:
                        pass
                # Replace pool and retry
                if attempt <= (retries + 1):
                    sleep_s = backoff_base ** (attempt - 1)
                    logger.warning(
                        "PostgreSQL OperationalError (attempt %s/%s): %s. Reconnecting in %.1fs",
                        attempt,
                        retries + 1,
                        oe,
                        sleep_s,
                    )
                    try:
                        self._reset_pool()
                    except Exception as re:
                        logger.error("Failed to reset PostgreSQL pool: %s", re)
                    time.sleep(sleep_s)
                    continue
                logger.error("PostgreSQL OperationalError: %s", oe)
                if swallow:
                    return None
                raise
            except Exception as e:
                if conn is not None:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                logger.exception("PostgreSQL error in transaction: %s", e)
                if swallow:
                    return None
                raise
            finally:
                if conn is not None:
                    try:
                        # If conn was closed, putconn will raise; ignore.
                        self._pool.putconn(conn)  # type: ignore[union-attr]
                    except Exception:
                        pass


_DB_SINGLETON: Optional[PostgresDatabase] = None


def init_db_from_env() -> PostgresDatabase:
    global _DB_SINGLETON
    if _DB_SINGLETON is None:
        cfg = load_postgres_config_from_env()
        _DB_SINGLETON = PostgresDatabase(cfg)
        _DB_SINGLETON.initialize()
    return _DB_SINGLETON


def get_db() -> PostgresDatabase:
    if _DB_SINGLETON is None:
        raise RuntimeError("Database not initialized. Call init_db_from_env() at startup.")
    return _DB_SINGLETON
