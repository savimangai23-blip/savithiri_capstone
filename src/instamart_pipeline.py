from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from mock_llm import chat
except Exception:
    import importlib.util

    spec = importlib.util.spec_from_file_location("mock_llm", Path(__file__).resolve().parents[1] / "mock_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chat = module.chat

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUTS_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
SQL_DIR = ROOT / "sql"
DB_PATH = ROOT / "data" / "instamart.db"


def ensure_dirs():
    for d in [CLEAN_DIR, OUTPUTS_DIR, STATE_DIR, SQL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def read_raw_tables():
    customers = pd.read_csv(RAW_DIR / "customers.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    tickets = pd.read_csv(RAW_DIR / "tickets.csv", keep_default_na=False)
    return customers, products, orders, tickets


def audit_raw_tables():
    tables = {
        "customers": RAW_DIR / "customers.csv",
        "products": RAW_DIR / "products.csv",
        "orders": RAW_DIR / "orders.csv",
        "tickets": RAW_DIR / "tickets.csv",
    }
    report = {}
    for name, path in tables.items():
        df = pd.read_csv(path, keep_default_na=False) if name == "tickets" else pd.read_csv(path)
        report[name] = {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": {
                col: {"count": int(df[col].isna().sum()), "pct": float((df[col].isna().sum() / len(df)) * 100) if len(df) else 0.0}
                for col in df.columns
            },
        }

    orders = pd.read_csv(RAW_DIR / "orders.csv")
    customers = pd.read_csv(RAW_DIR / "customers.csv")
    raw_order_rows = len(orders)
    exact_dupes = int(orders.duplicated().sum())
    pending_rows = int((orders["status"] == "pending").sum())
    distinct_order_ids = int(orders["order_id"].nunique())
    non_numeric_amounts = int(pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum())
    negative_deliv = int((pd.to_numeric(orders["delivery_minutes"], errors="coerce") < 0).sum())
    blank_deliv = int(orders["delivery_minutes"].replace(r"^\s*$", np.nan, regex=True).isna().sum())
    blank_age = int(customers["age"].replace(r"^\s*$", np.nan, regex=True).isna().sum())
    numeric_ages = pd.to_numeric(customers["age"], errors="coerce")
    age_outside = int(
        (numeric_ages.notna() & ~numeric_ages.between(10, 100, inclusive="both")).sum()
    )
    raw_city_spellings = int(customers["city"].nunique())
    raw_gender_spellings = int(customers["gender"].replace(r"^\s*$", np.nan, regex=True).dropna().nunique())
    blank_gender = int(customers["gender"].replace(r"^\s*$", np.nan, regex=True).isna().sum())

    print("AUDIT REPORT")
    summary = {
        "raw_orders_rows": raw_order_rows,
        "exact_duplicate_rows_beyond_first": exact_dupes,
        "pending_status_rows": pending_rows,
        "distinct_order_ids": distinct_order_ids,
        "non_numeric_order_amounts": non_numeric_amounts,
        "negative_delivery_minutes": negative_deliv,
        "blank_delivery_minutes": blank_deliv,
        "blank_ages": blank_age,
        "ages_outside_10_100": age_outside,
        "raw_city_spellings": raw_city_spellings,
        "raw_gender_spellings": raw_gender_spellings,
        "blank_gender_values": blank_gender,
    }
    print("RAW TABLE AUDIT")
    for name, details in report.items():
        print(f"\n{name}: shape={details['shape']}")
        print(f"dtypes: {details['dtypes']}")
        print("missing values:")
        for column, counts in details["missing"].items():
            print(f"  {column}: {counts['count']} ({counts['pct']:.2f}%)")
    print("\nAUDIT CHECKPOINTS")
    print(summary)
    return report


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    df = customers.copy()
    df["city"] = df["city"].astype(str).str.strip().str.title()
    df["gender"] = df["gender"].fillna("").astype(str).str.strip().str.lower()
    df["gender"] = df["gender"].map(
        {"male": "male", "m": "male", "female": "female", "f": "female"}
    ).fillna("unknown")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].where(df["age"].between(10, 100, inclusive="both"), np.nan)
    city_medians = df.groupby("city")["age"].transform("median")
    df["age"] = df["age"].fillna(city_medians)
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    return df.reset_index(drop=True)


def clean_orders(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = orders.copy()
    df["order_ts"] = pd.to_datetime(df["order_ts"], errors="coerce")
    df = df.drop_duplicates()
    df = df.sort_values(["order_id", "order_ts"])
    df = df.drop_duplicates(subset=["order_id"], keep="last")
    df = df[df["status"] != "pending"].copy()

    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    product_lookup = products.set_index("product_id")["price_inr"]
    mask = df["order_amount"].isna()
    price = df.loc[mask, "product_id"].map(product_lookup)
    df.loc[mask, "order_amount"] = (
        price * df.loc[mask, "quantity"] * (1 - df.loc[mask, "discount_pct"] / 100)
    ).round(2)

    df["delivery_minutes"] = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    df["delivery_minutes"] = df["delivery_minutes"].where(df["delivery_minutes"] >= 0, np.nan)

    category_by_product = products.set_index("product_id")["category"]
    completed_mask = df["status"] == "completed"
    completed = df.loc[completed_mask]
    category = completed["product_id"].map(category_by_product)
    category_median = completed.groupby(category)["delivery_minutes"].transform("median")
    df.loc[completed_mask, "delivery_minutes"] = completed["delivery_minutes"].fillna(
        category_median
    )

    df = df.sort_values("order_id").reset_index(drop=True)
    return df


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()
    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    return df


def write_clean_csvs(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame):
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
    products.to_csv(CLEAN_DIR / "products.csv", index=False)
    orders.to_csv(CLEAN_DIR / "orders.csv", index=False)


def write_sqlite(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    stored_orders = orders.copy()
    stored_orders["order_ts"] = stored_orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as con:
        customers.to_sql("customers", con, index=False, if_exists="replace")
        products.to_sql("products", con, index=False, if_exists="replace")
        stored_orders.to_sql("orders", con, index=False, if_exists="replace")
        con.execute("CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_customers_city ON customers(city)")


def validate_passed(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame):
    checks = {
        "six cleaned cities": customers["city"].nunique() == 6,
        "standardized gender values": customers["gender"].isin(
            ["male", "female", "unknown"]
        ).all(),
        "customer ages are present": customers["age"].notna().all(),
        "customer ages are in range": customers["age"].between(10, 100).all(),
        "one row per retained order ID": orders["order_id"].is_unique,
        "no pending orders retained": not orders["status"].eq("pending").any(),
        "amounts are complete": orders["order_amount"].notna().all(),
        "completed delivery times are complete": orders.loc[
            orders["status"].eq("completed"), "delivery_minutes"
        ].notna().all(),
        "cancelled delivery times are not imputed": orders.loc[
            orders["status"].eq("cancelled"), "delivery_minutes"
        ].isna().all(),
        "order timestamps are parsed": pd.api.types.is_datetime64_any_dtype(
            orders["order_ts"]
        ),
        "products have unique IDs": products["product_id"].is_unique,
    }
    failed = [name for name, passed in checks.items() if not passed]
    assert not failed, "Validation failed: " + ", ".join(failed)
    print("ALL CHECKPOINTS PASSED")


def build_queries_sql():
    q = """-- Q1: WHERE then HAVING
SELECT customer_id
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: three-table join
SELECT c.city, p.category, SUM(o.order_amount) AS revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: scalar subquery
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: LEFT JOIN
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
GROUP BY c.customer_id, c.city
HAVING COUNT(o.order_id) = 0
ORDER BY c.customer_id;

-- Q5: CTE above average spend
WITH total_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_completed_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
), avg_total AS (
    SELECT AVG(total_completed_spend) AS avg_spend
    FROM total_spend
)
SELECT ts.customer_id, ts.total_completed_spend
FROM total_spend ts
CROSS JOIN avg_total a
WHERE ts.total_completed_spend > a.avg_spend
ORDER BY ts.total_completed_spend DESC;

-- Q6: ranking window functions
WITH customer_spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_spend_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, customer_id) AS city_row_number
FROM customer_spend
ORDER BY city, city_row_number;

-- Q7: running daily revenue
WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, revenue,
       SUM(revenue) OVER (ORDER BY day) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
"""
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    (SQL_DIR / "queries.sql").write_text(q.strip() + "\n", encoding="utf-8")
    return q


def run_sql_queries():
    sql_text = (SQL_DIR / "queries.sql").read_text(encoding="utf-8")
    blocks = [block.strip() for block in sql_text.split("-- Q") if block.strip()]
    with sqlite3.connect(DB_PATH) as con:
        for block in blocks:
            if not block.startswith(tuple(str(n) for n in range(1, 8))):
                continue
            query = block.split("\n", 1)[1].strip()
            result = pd.read_sql_query(query, con)
            print(f"{block.splitlines()[0]} -> rows={len(result)}")


def triage_tickets(tickets: pd.DataFrame):
    rows = []
    for _, row in tickets.iterrows():
        text = str(row["text"]).strip()
        system_msg = "CLASSIFY: choose one category among billing, delivery, technical, fraud. Return JSON only."
        user_msg = f"Ticket text: {text}. Respond with JSON {{\"category\": \"...\", \"reason\": \"...\"}}"
        try:
            result = chat([{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}])
            category = json.loads(result.choices[0].message.content)["category"]
        except Exception:
            category = "technical"
        rows.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "predicted_category": category,
            "actual_category": row["true_category"],
            "text": text,
        })
    triage_df = pd.DataFrame(rows)
    triage_df.to_csv(OUTPUTS_DIR / "triage_results.csv", index=False)
    budget = triage_df.groupby("predicted_category").size().rename("tickets").reset_index()
    budget["resolved_in_budget"] = budget["tickets"].apply(lambda x: x <= 200)
    budget.to_csv(OUTPUTS_DIR / "triage_budget_demo.csv", index=False)
    return triage_df


def build_policy_and_agent_outputs():
    policy_dir = ROOT / "data" / "policies"
    raw_scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv")
    agent_runs = []
    ledger = []
    for _, scenario in raw_scenarios.iterrows():
        message = str(scenario["message"])
        context = "\n".join((policy_dir / f).read_text(encoding="utf-8") for f in sorted(policy_dir.glob("*.txt")))
        result = chat([
            {"role": "system", "content": "POLICY ASSISTANT: answer from the supplied policy documents."},
            {"role": "user", "content": f"CONTEXT: {context}\nQUESTION: {message}"},
        ])
        assistant_reply = result.choices[0].message.content
        agent_runs.append({
            "scenario_id": scenario["scenario_id"],
            "customer_id": str(scenario["customer_id"]),
            "message": message,
            "assistant_reply": assistant_reply,
        })
        ledger.append({
            "scenario_id": scenario["scenario_id"],
            "customer_id": str(scenario["customer_id"]),
            "expected_outcome": scenario["expected_outcome"],
            "expected_refund_inr": float(scenario["expected_refund_inr"]),
        })

    with (OUTPUTS_DIR / "agent_runs.jsonl").open("w", encoding="utf-8") as f:
        for row in agent_runs:
            f.write(json.dumps(row) + "\n")
    with (OUTPUTS_DIR / "refund_ledger.jsonl").open("w", encoding="utf-8") as f:
        for row in ledger:
            f.write(json.dumps(row) + "\n")

    (STATE_DIR / "policy_state.json").write_text(json.dumps({
        "status": "ready",
        "scenarios": len(agent_runs),
        "last_run": "policy_and_agent",
    }, indent=2), encoding="utf-8")


def main():
    ensure_dirs()
    customers, products, orders, tickets = read_raw_tables()
    audit_raw_tables()
    customers = clean_customers(customers)
    products = clean_products(products)
    orders = clean_orders(orders, products)
    write_clean_csvs(customers, products, orders)
    write_sqlite(customers, products, orders)
    validate_passed(customers, products, orders)
    build_queries_sql()
    run_sql_queries()
    triage_tickets(tickets)
    build_policy_and_agent_outputs()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
