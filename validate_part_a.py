
import os
import sqlite3
import pandas as pd

# --------------------------------------------------
# 1. Load cleaned files
# --------------------------------------------------

customers_path = "data/clean/customers.csv"
products_path = "data/clean/products.csv"
orders_path = "data/clean/orders.csv"
database_path = "data/instamart.db"
queries_path = "sql/queries.sql"

checks_passed = 0
checks_failed = 0


def check(condition, description):
    global checks_passed, checks_failed

    if condition:
        checks_passed += 1
        print(f"PASS: {description}")
    else:
        checks_failed += 1
        print(f"FAIL: {description}")


# --------------------------------------------------
# 2. Check required files
# --------------------------------------------------

print("\n--- Part A Validation ---")

check(os.path.exists(customers_path), "Clean customers CSV exists")
check(os.path.exists(products_path), "Clean products CSV exists")
check(os.path.exists(orders_path), "Clean orders CSV exists")
check(os.path.exists(database_path), "SQLite database exists")
check(os.path.exists(queries_path), "SQL queries file exists")

if checks_failed:
    print("\nValidation stopped: required files are missing.")
    raise SystemExit(1)


# --------------------------------------------------
# 3. Read cleaned CSV files
# --------------------------------------------------

customers = pd.read_csv(customers_path)
products = pd.read_csv(products_path)
orders = pd.read_csv(orders_path)


# --------------------------------------------------
# 4. Customer validations
# --------------------------------------------------

print("\n--- Customer Checks ---")

check(len(customers) == 1200, "Exactly 1200 customers")
check(customers["customer_id"].is_unique, "No duplicate customer IDs")
check(customers["age"].notna().all(), "No missing customer ages")
check(
    customers["age"].between(18, 60).all(),
    "All customer ages are between 18 and 60"
)
check(
    set(customers["city"].dropna().unique())
    == {"Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune"},
    "Customers belong to the six expected cities"
)
check(
    set(customers["gender"].dropna().unique())
    == {"male", "female", "unknown"},
    "Gender values are standardized"
)
check(
    pd.to_datetime(customers["signup_date"], errors="coerce").notna().all(),
    "All signup dates are valid"
)


# --------------------------------------------------
# 5. Product validations
# --------------------------------------------------

print("\n--- Product Checks ---")

check(len(products) == 40, "Exactly 40 products")
check(products["product_id"].is_unique, "No duplicate product IDs")
check(
    products[["product_id", "category", "price_inr"]].notna().all().all(),
    "Product ID, category and price have no missing values"
)
check(
    pd.to_numeric(products["price_inr"], errors="coerce").gt(0).all(),
    "All product prices are positive numbers"
)


# --------------------------------------------------
# 6. Order validations
# --------------------------------------------------

print("\n--- Order Checks ---")

check(len(orders) == 12309, "Exactly 12309 cleaned orders")
check(orders["order_id"].is_unique, "No duplicate order IDs")
check(
    set(orders["status"].dropna().unique()) == {"completed", "cancelled"},
    "Only completed and cancelled orders remain"
)
check(
    (orders["status"] == "completed").sum() == 11110,
    "Exactly 11110 completed orders"
)
check(
    (orders["status"] == "cancelled").sum() == 1199,
    "Exactly 1199 cancelled orders"
)
check(
    orders["order_amount"].notna().all(),
    "No missing order amounts"
)
check(
    pd.to_numeric(orders["order_amount"], errors="coerce").ge(0).all(),
    "Order amounts are numeric and non-negative"
)
check(
    orders.loc[orders["status"] == "completed", "delivery_minutes"].notna().all(),
    "Completed orders have no missing delivery times"
)
check(
    orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().all(),
    "Cancelled orders have missing delivery times"
)
check(
    pd.to_datetime(orders["order_ts"], errors="coerce").notna().all(),
    "All order timestamps are valid"
)


# --------------------------------------------------
# 7. SQLite validations
# --------------------------------------------------

print("\n--- SQLite Checks ---")

conn = sqlite3.connect(database_path)

try:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        conn
    )["name"].tolist()

    check(
        {"customers", "products", "orders"}.issubset(set(tables)),
        "SQLite contains customers, products and orders tables"
    )

    for table, expected_rows in [
        ("customers", len(customers)),
        ("products", len(products)),
        ("orders", len(orders))
    ]:
        actual_rows = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table}",
            conn
        )["total"].iloc[0]

        check(
            actual_rows == expected_rows,
            f"SQLite {table} table has {expected_rows} rows"
        )

finally:
    conn.close()


# --------------------------------------------------
# 8. SQL query count
# --------------------------------------------------

print("\n--- SQL Query Check ---")

with open(queries_path, "r", encoding="utf-8") as file:
    sql_script = file.read()

queries = [q.strip() for q in sql_script.split(";") if q.strip()]

check(len(queries) == 7, "SQL file contains seven queries")


# --------------------------------------------------
# 9. Final result
# --------------------------------------------------

print("\n--- Final Result ---")
print(f"Checks passed: {checks_passed}")
print(f"Checks failed: {checks_failed}")

if checks_failed == 0:
    print("\nALL CHECKPOINTS PASSED")
else:
    print("\nVALIDATION FAILED")
    raise SystemExit(1)
