"""src/common/db.py

Единая точка подключения к PostgreSQL для всего проекта.

Зачем нужен этот файл:
- На соревновании вам выдадут другие параметры подключения (host/port/db/user/password).
  Чтобы НЕ менять код во всех модулях, параметры лежат в `.env`, а здесь мы их читаем.
- Все остальные скрипты (ингест, фичи, кластеризация, дашборд) импортируют `engine`
  или используют функции `get_engine()/get_connection()`.

Что вы меняете на соревновании:
- Почти всегда ТОЛЬКО файл `.env` в корне проекта (PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD,
  иногда PG_SCHEMA). Код в этом файле менять не нужно.

Важно:
- `.env` должен лежать в КОРНЕ проекта (рядом с requirements.txt), как у вас сейчас.
- Не хардкодьте логины/пароли в коде.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection


# Загружаем переменные окружения из файла `.env`.
# Если `.env` отсутствует, load_dotenv() просто ничего не сделает.
# На соревновании вы подставите выданные доступы именно в `.env`.
load_dotenv()


@dataclass(frozen=True)
class DBConfig:
    """Конфигурация подключения к PostgreSQL.

    Источник: переменные окружения (обычно из `.env`).

    Обязательные:
    - PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD

    Опциональные:
    - PG_SCHEMA (по умолчанию public)
    - PG_SSLMODE (обычно не нужен локально; иногда нужен на удалённом сервере)
    """

    host: str
    port: int
    db: str
    user: str
    password: str
    schema: str = "public"
    sslmode: Optional[str] = None

    @staticmethod
    def from_env() -> "DBConfig":
        """Считывает конфиг из переменных окружения."""
        host = os.getenv("PG_HOST")
        port = os.getenv("PG_PORT")
        db = os.getenv("PG_DB")
        user = os.getenv("PG_USER")
        password = os.getenv("PG_PASSWORD")

        # Схема иногда на соревнованиях не public (может совпадать с логином).
        schema = os.getenv("PG_SCHEMA", "public")

        # Иногда на удалённых серверах могут попросить SSL.
        sslmode = os.getenv("PG_SSLMODE")

        missing = [
            name
            for name, value in {
                "PG_HOST": host,
                "PG_PORT": port,
                "PG_DB": db,
                "PG_USER": user,
                "PG_PASSWORD": password,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Не найдены переменные окружения для подключения к БД: "
                + ", ".join(missing)
                + ".\nПроверьте файл .env в корне проекта."
            )

        return DBConfig(
            host=str(host),
            port=int(port),
            db=str(db),
            user=str(user),
            password=str(password),
            schema=str(schema),
            sslmode=str(sslmode) if sslmode else None,
        )

    def sqlalchemy_url(self) -> str:
        """Формирует SQLAlchemy URL для psycopg2."""
        # Пример: postgresql+psycopg2://user:pass@host:port/db
        # ВАЖНО: пароль может содержать спецсимволы; для соревнований обычно простой.
        base = (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )
        # Если понадобится SSL (редко), можно добавить параметр.
        if self.sslmode:
            return base + f"?sslmode={self.sslmode}"
        return base


# Кешируем Engine один раз на весь процесс.
_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Возвращает singleton Engine.

    Engine создаётся один раз. Используется во всех модулях проекта.
    """
    global _engine
    if _engine is None:
        cfg = DBConfig.from_env()

        # pool_pre_ping=True помогает избегать ошибок "connection already closed"
        # при долгой работе дашборда/скрипта.
        _engine = create_engine(
            cfg.sqlalchemy_url(),
            pool_pre_ping=True,
            future=True,
        )
    return _engine


# Удобный алиас, чтобы в других модулях можно было писать:
# from src.common.db import engine
engine: Engine = get_engine()


def get_connection() -> Connection:
    """Открывает и возвращает Connection.

    Используйте так:
        with get_connection() as conn:
            ...

    Внутри with соединение корректно закрывается.
    """
    return engine.connect()


def set_search_path(conn: Connection) -> None:
    """Устанавливает search_path на схему из конфигурации.

    Это полезно, если на соревновании таблицы нужно создавать/читать не из public.
    """
    cfg = DBConfig.from_env()
    conn.execute(text(f"SET search_path TO {cfg.schema};"))


def test_connection(verbose: bool = True) -> None:
    """Проверяет, что соединение с БД работает.

    Что проверяем:
    - SELECT 1
    - (опционально) установка search_path

    Если всё ок — ошибок не будет.
    """
    with get_connection() as conn:
        conn.execute(text("SELECT 1;"))
        # На всякий случай сразу выставим схему (не ломает public).
        set_search_path(conn)

    if verbose:
        cfg = DBConfig.from_env()
        print(
            "OK: connected to PostgreSQL | "
            f"host={cfg.host} port={cfg.port} db={cfg.db} schema={cfg.schema}"
        )


def run_sql_file(path: str, verbose: bool = True) -> None:
    """Выполняет SQL-файл (например, схемы из папки sql/).

    Зачем:
    - Инициализация таблиц одной командой.
    - На соревновании удобно: python -c "from ... import run_sql_file; run_sql_file('sql/001_schema.sql')"

    Примечание:
    - Этот метод читает файл целиком и выполняет как один SQL-скрипт.
    - Ваши SQL-файлы лучше писать идемпотентно (CREATE TABLE IF NOT EXISTS).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SQL-файл не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        sql = f.read()

    with get_connection() as conn:
        set_search_path(conn)
        conn.execute(text(sql))
        conn.commit()

    if verbose:
        print(f"OK: executed SQL file: {path}")
