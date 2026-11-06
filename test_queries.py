# import sqlite3
# import pandas as pd

# # Connect to the SQLite database
# conn = sqlite3.connect("data/instamart.db")

# # Read the SQL file
# with open("sql/queries.sql", "r", encoding="utf-8") as f:
#     sql_script = f.read()

# # Split the script by semicolons into individual queries
# queries = [q.strip() for q in sql_script.split(";") if q.strip()]

# # Execute each query and print results
# for i, query in enumerate(queries, 1):
#     print(f"\n--- Running Query {i} ---")
#     try:
#         df = pd.read_sql(query, conn)
#         print(df.head(10)) # Print first 10 rows of result
#     except Exception as e:
#         print(f"Error in Query {i}: {e}")

# conn.close()



import sqlite3
import pandas as pd

# Load cleaned datasets
customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")
products = pd.read_csv("data/clean/products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Connect to SQLite
conn = sqlite3.connect("data/instamart.db")

# Read and execute all seven SQL queries
with open("sql/queries.sql", "r", encoding="utf-8") as f:
    sql_script = f.read()

queries = [q.strip() for q in sql_script.split(";") if q.strip()]

results = {}

for i, query in enumerate(queries, 1):
    print(f"\n--- Running Query {i} ---")
    df = pd.read_sql(query, conn)
    results[i] = df
    print(df.head(10))

# Compare SQL results against pandas calculations
def check_match(name, sql_result, pandas_result):
    sql_result = sql_result.reset_index(drop=True)
    pandas_result = pandas_result.reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            sql_result,
            pandas_result,
            check_dtype=False,
            check_exact=False,
            atol=0.01,
            rtol=0
        )
        print(f"PASS: {name} SQL and pandas results match.")
    except AssertionError as e:
        print(f"FAIL: {name} results do not match.")
        raise e


print("\n--- Pandas Cross-Checks ---")

# Cross-check 1: Revenue by city
completed = orders[orders["status"] == "completed"]

city_data = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id"
)

pandas_q1 = (
    city_data.groupby("city")
    .agg(
        total_completed_orders=("order_id", "count"),
        total_revenue_inr=("order_amount", "sum"),
        avg_order_value_inr=("order_amount", "mean")
    )
    .reset_index()
)

pandas_q1["total_revenue_inr"] = pandas_q1["total_revenue_inr"].round(2)
pandas_q1["avg_order_value_inr"] = pandas_q1["avg_order_value_inr"].round(2)
pandas_q1 = pandas_q1.sort_values(
    "total_revenue_inr", ascending=False
).reset_index(drop=True)

check_match("City revenue", results[1], pandas_q1)

# Cross-check 2: Average delivery time by city
delivery_data = completed[
    completed["delivery_minutes"] > 0
].merge(
    customers[["customer_id", "city"]],
    on="customer_id"
)

pandas_q2 = (
    delivery_data.groupby("city")
    .agg(
        avg_delivery_minutes=("delivery_minutes", "mean"),
        delivered_orders_count=("order_id", "count")
    )
    .reset_index()
)

pandas_q2["avg_delivery_minutes"] = (
    pandas_q2["avg_delivery_minutes"].round(1)
)

pandas_q2 = pandas_q2.sort_values(
    "avg_delivery_minutes"
).reset_index(drop=True)

check_match("Delivery time", results[2], pandas_q2)

# Cross-check 3: Monthly order volume and revenue
completed = completed.copy()
completed["order_month"] = completed["order_ts"].dt.strftime("%Y-%m")

pandas_q4 = (
    completed.groupby("order_month")
    .agg(
        order_count=("order_id", "count"),
        monthly_revenue_inr=("order_amount", "sum")
    )
    .reset_index()
)

pandas_q4["monthly_revenue_inr"] = (
    pandas_q4["monthly_revenue_inr"].round(2)
)

pandas_q4 = pandas_q4.sort_values(
    "order_month"
).reset_index(drop=True)

check_match("Monthly revenue", results[4], pandas_q4)

# Cross-check 4: Order status distribution
pandas_q6 = (
    orders.groupby("status")
    .size()
    .reset_index(name="total_orders")
)

pandas_q6["percentage"] = (
    pandas_q6["total_orders"] * 100 / len(orders)
).round(2)

pandas_q6 = pandas_q6.sort_values(
    "total_orders", ascending=False
).reset_index(drop=True)

check_match("Order status distribution", results[6], pandas_q6)

conn.close()

print("\nAll seven SQL queries executed.")
print("All four pandas cross-checks passed.")
