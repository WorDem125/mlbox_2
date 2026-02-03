

"""src/init_db.py

Инициализация БД (создание таблиц, индексов и view) одной командой.

Зачем нужен этот файл:
- На соревнованиях/в проекте вы хотите быстро развернуть структуру БД.
- SQL-логика хранится в папке `sql/` (отдельно от Python), а этот скрипт
  просто последовательно выполняет SQL-файлы.

Как использовать локально:
1) Заполните `.env` (PG_HOST/PG_PORT/PG_DB/PG_USER/PG_PASSWORD/PG_SCHEMA)
2) Запустите:
   python -m src.init_db
   или
   python src/init_db.py

Как использовать на соревнованиях:
- Обычно вы меняете ТОЛЬКО `.env` (выданные логин/пароль/host/db/schema).
- Код менять не нужно.

Что может потребоваться изменить на соревнованиях:
- Если права запрещают CREATE INDEX — можно пропустить `002_indexes.sql`.
- Если права запрещают CREATE VIEW — можно пропустить `003_views.sql`.
- Если эксперты требуют иной порядок/добавочные SQL — добавляете файл в `sql/`
  и включаете его в список ниже.

Примечание:
- Скрипт не "ломает" существующую БД: все SQL написаны идемпотентно
  (CREATE ... IF NOT EXISTS / CREATE OR REPLACE VIEW).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.common.db import run_sql_file, test_connection


def _project_root() -> Path:
    """Определяет корень проекта.

    Мы считаем, что этот файл лежит в `src/`, а корень проекта — на уровень выше.
    Это нужно, чтобы корректно находить папку `sql/` независимо от того,
    откуда запущен скрипт.
    """
    return Path(__file__).resolve().parents[1]


def _sql_dir() -> Path:
    """Возвращает путь до папки sql/ в корне проекта."""
    return _project_root() / "sql"


def init_db(
    run_schema: bool = True,
    run_indexes: bool = True,
    run_views: bool = True,
) -> None:
    """Выполняет SQL-скрипты инициализации.

    Параметры позволяют гибко отключать индексы/вьюхи,
    если на соревнованиях нет прав на их создание.
    """

    # 1) Проверяем, что подключение работает и schema/search_path корректны.
    test_connection(verbose=True)

    sql_dir = _sql_dir()
    if not sql_dir.exists():
        raise FileNotFoundError(
            f"Не найдена папка sql/: {sql_dir}\n"
            "Проверьте структуру проекта: в корне должна быть папка sql/."
        )

    # 2) Список SQL-файлов в нужном порядке.
    # Важно: сначала таблицы, затем индексы, затем view.
    steps: list[tuple[str, bool]] = [
        ("001_schema.sql", run_schema),
        ("002_indexes.sql", run_indexes),
        ("003_views.sql", run_views),
    ]

    # 3) Выполняем выбранные шаги.
    for filename, enabled in steps:
        path = sql_dir / filename
        if not enabled:
            print(f"SKIP: {filename} (отключено параметром)")
            continue

        if not path.exists():
            # На соревнованиях иногда удобно временно удалить/не иметь файл —
            # тогда просто пропускаем с предупреждением.
            print(f"WARN: SQL-файл не найден, пропускаю: {path}")
            continue

        print(f"RUN: {path}")
        run_sql_file(str(path), verbose=True)

    print("DONE: database initialized")


def _parse_args(argv: list[str]) -> dict:
    """Очень простой парсер аргументов.

    Поддерживаем:
    --no-indexes  (пропустить 002_indexes.sql)
    --no-views    (пропустить 003_views.sql)

    Зачем:
    - На соревнованиях может не быть прав на CREATE INDEX/VIEW.
    - Тогда вы запускаете: python -m src.init_db --no-indexes
    """
    args = {
        "run_schema": True,
        "run_indexes": True,
        "run_views": True,
    }

    for a in argv:
        if a == "--no-indexes":
            args["run_indexes"] = False
        elif a == "--no-views":
            args["run_views"] = False
        elif a in ("-h", "--help"):
            print(
                "Использование:\n"
                "  python -m src.init_db [--no-indexes] [--no-views]\n\n"
                "Опции:\n"
                "  --no-indexes   пропустить создание индексов\n"
                "  --no-views     пропустить создание представлений (views)\n"
            )
            raise SystemExit(0)

    return args


if __name__ == "__main__":
    # Позволяем запускать и как `python src/init_db.py`, и как `python -m src.init_db`.
    # На некоторых системах при прямом запуске может понадобиться добавить корень
    # проекта в PYTHONPATH. Подстрахуемся.
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parsed = _parse_args(sys.argv[1:])
    init_db(**parsed)

    # python -m src.init_db