"""Run the seven SQL analyses and pandas cross-checks for Part A."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"


def read_queries() -> list[tuple[str, str]]:
    query_blocks = SQL_PATH.read_text(encoding="utf-8").split("-- Q")[1:]
    return [
        (block.splitlines()[0].strip(), block.split("\n", 1)[1].strip())
        for block in query_blocks
        if block.strip()
    ]


def main() -> None:
    with sqlite3.connect(DB_PATH) as con:
        results = {
            title: pd.read_sql_query(query, con)
            for title, query in read_queries()
        }
        completed = pd.read_sql_query(
            """
            SELECT o.order_amount, o.discount_pct, o.quantity, o.delivery_minutes
            FROM orders o
            WHERE o.status = 'completed'
            """,
            con,
        )

    print("SQL QUERY RESULTS")
    for title, result in results.items():
        print(f"{title}: {len(result)} rows")

    q2 = results["2: three-table join"]
    print(
        "Q2 revenue total: INR "
        f"{q2['revenue'].sum():,.2f}; largest city-category cell: "
        f"{q2.loc[q2['revenue'].idxmax(), 'city']} / "
        f"{q2.loc[q2['revenue'].idxmax(), 'category']} = "
        f"INR {q2['revenue'].max():,.2f}"
    )
    q5 = results["5: CTE above average spend"]
    with sqlite3.connect(DB_PATH) as con:
        spend = pd.read_sql_query(
            """
            SELECT customer_id, SUM(order_amount) AS total_spend
            FROM orders
            WHERE status = 'completed'
            GROUP BY customer_id
            """,
            con,
        )
    print(
        f"Q5 above-average customers: {len(q5)}; "
        f"average customer spend: INR {spend['total_spend'].mean():,.2f}"
    )
    q6 = results["6: ranking window functions"]
    print("Q6 top spender per city:")
    print(
        q6.loc[q6["city_row_number"].eq(1), ["city", "customer_id", "total_spend"]]
        .to_string(index=False)
    )
    print(
        "Q6 rows where RANK differs from ROW_NUMBER: "
        f"{int(q6['spend_rank'].ne(q6['city_row_number']).sum())}"
    )
    q7 = results["7: running daily revenue"]
    print(
        f"Q7 daily rows: {len(q7)}; last cumulative revenue: "
        f"INR {q7['cumulative_revenue'].iloc[-1]:,.2f}"
    )

    with sqlite3.connect(DB_PATH) as con:
        pivot_data = pd.read_sql_query(
            """
            SELECT c.city, p.category, o.order_amount
            FROM orders o
            JOIN customers c ON c.customer_id = o.customer_id
            JOIN products p ON p.product_id = o.product_id
            WHERE o.status = 'completed'
            """,
            con,
        )
        city_spend = pd.read_sql_query(
            """
            SELECT c.city, o.order_amount
            FROM orders o
            JOIN customers c ON c.customer_id = o.customer_id
            WHERE o.status = 'completed'
            """,
            con,
        )
    pivot = pd.pivot_table(
        pivot_data,
        index="city",
        columns="category",
        values="order_amount",
        aggfunc="sum",
        fill_value=0,
        margins=True,
    )
    print("\nPANDAS Q2 PIVOT WITH MARGINS")
    print(pivot.to_string())
    print(f"Pivot grand total: INR {pivot.loc['All', 'All']:,.2f}")
    by_city = city_spend.groupby("city").agg(
        sum=("order_amount", "sum"),
        mean=("order_amount", "mean"),
        count=("order_amount", "count"),
    )
    print("\nCOMPLETED SPEND BY CITY")
    print(by_city.to_string(float_format=lambda value: f"{value:,.2f}"))

    values = completed["order_amount"]
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    print("\nCOMPLETED ORDER AMOUNT DISTRIBUTION")
    print(f"count: {len(values)}")
    print(f"mean: INR {values.mean():,.2f}")
    print(f"median: INR {median:,.2f}")
    print(f"skewness: {values.skew():.4f}")
    print(
        f"Q1: INR {q1:,.2f}; Q2: INR {median:,.2f}; Q3: INR {q3:,.2f}"
    )
    print(f"IQR: INR {iqr:,.2f}")
    print(f"lower 1.5xIQR fence: INR {lower_fence:,.2f}")
    print(f"upper 1.5xIQR fence: INR {upper_fence:,.2f}")
    print(f"outliers below lower fence: {int((values < lower_fence).sum())}")
    print(f"outliers above upper fence: {int((values > upper_fence).sum())}")

    correlations = completed.corr(numeric_only=True)
    pairs = correlations.where(
        pd.DataFrame(
            [[i != j for j in correlations.columns] for i in correlations.columns],
            index=correlations.index,
            columns=correlations.columns,
        )
    ).stack()
    pair = max(
        (
            (abs(value), left, right, value)
            for (left, right), value in pairs.items()
            if left < right
        ),
        key=lambda item: item[0],
    )
    print("\nCOMPLETED-ORDER CORRELATION MATRIX")
    print(correlations.to_string(float_format=lambda value: f"{value:.4f}"))
    print(
        f"Largest absolute correlation: {pair[1]} / {pair[2]} = {pair[3]:.4f}"
    )


if __name__ == "__main__":
    main()
