"""Validate the cleaned Instamart exports and SQLite database."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"


def main() -> None:
    customers = pd.read_csv(CLEAN_DIR / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders.csv", parse_dates=["order_ts"])

    checks = {
        "six standardized cities": customers["city"].nunique() == 6,
        "standardized gender values": customers["gender"].isin(
            ["female", "male", "unknown"]
        ).all(),
        "all customer ages present": customers["age"].notna().all(),
        "customer ages within 10-100": customers["age"].between(10, 100).all(),
        "signup dates parsed": customers["signup_date"].notna().all(),
        "one row per cleaned order ID": orders["order_id"].is_unique,
        "no pending orders": not orders["status"].eq("pending").any(),
        "order amounts all present": orders["order_amount"].notna().all(),
        "completed delivery times all present": orders.loc[
            orders["status"].eq("completed"), "delivery_minutes"
        ].notna().all(),
        "cancelled delivery times remain missing": orders.loc[
            orders["status"].eq("cancelled"), "delivery_minutes"
        ].isna().all(),
        "all order timestamps parsed": orders["order_ts"].notna().all(),
    }

    with sqlite3.connect(DB_PATH) as con:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        checks["three expected database tables"] = {
            "customers",
            "products",
            "orders",
        }.issubset(tables)
        if checks["three expected database tables"]:
            db_counts = {
                table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("customers", "products", "orders")
            }
            checks["CSV and SQLite row counts agree"] = db_counts == {
                "customers": len(customers),
                "products": len(products),
                "orders": len(orders),
            }
            timestamp_values = [
                row[0]
                for row in con.execute(
                    "SELECT order_ts FROM orders WHERE order_ts IS NOT NULL LIMIT 20"
                )
            ]
            checks["SQLite timestamps use second precision text"] = all(
                re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", value)
                for value in timestamp_values
            )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("Validation failed: " + ", ".join(failed))
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
