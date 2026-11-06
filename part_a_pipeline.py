import os
import sqlite3
import pandas as pd
import numpy as np

# Ensure directories exist
os.makedirs("data/clean", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("--- 1. Auditing and Cleaning Customers ---")
cust_df = pd.read_csv("data/raw/customers.csv")

# Clean City: remove whitespace and title case
cust_df["city"] = cust_df["city"].str.strip().str.title()

# Clean Gender: map variations to male, female, unknown (blank -> unknown)
def clean_gender(g):
    if pd.isna(g) or str(g).strip() == "":
        return "unknown"
    g_lower = str(g).strip().lower()
    if g_lower in ["m", "male"]:
        return "male"
    elif g_lower in ["f", "female"]:
        return "female"
    return "unknown"

cust_df["gender"] = cust_df["gender"].apply(clean_gender)

# Clean Age: Treat ages outside 10-100 as missing, then fill with median age of customer's city
cust_df["age"] = pd.to_numeric(cust_df["age"], errors="coerce")
cust_df.loc[(cust_df["age"] < 10) | (cust_df["age"] > 100), "age"] = np.nan
city_medians = cust_df.groupby("city")["age"].transform("median")
cust_df["age"] = cust_df["age"].fillna(city_medians).astype(int)

# Convert signup_date to datetime
cust_df["signup_date"] = pd.to_datetime(cust_df["signup_date"])

# Save clean customers
cust_df.to_csv("data/clean/customers.csv", index=False)
print("Customers cleaned and saved. City counts:")
print(cust_df["city"].value_counts())
print("Gender counts:", cust_df["gender"].value_counts().to_dict())

print("\n--- 2. Auditing and Cleaning Orders ---")
orders_df = pd.read_csv("data/raw/orders.csv")

# 1. Convert order_ts to datetime
orders_df["order_ts"] = pd.to_datetime(orders_df["order_ts"])

# 2. Remove exact duplicate rows
orders_df = orders_df.drop_duplicates()

# 3. Handle partial/status duplicates: keep record with latest order_ts per order_id
orders_df = orders_df.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last")

# 4. Convert order_amount to numeric, unparseable -> NaN
orders_df["order_amount"] = pd.to_numeric(orders_df["order_amount"], errors="coerce")

# 5. Fill missing order_amount using product price * quantity and discounting if available
products_df = pd.read_csv("data/raw/products.csv")
orders_df = orders_df.merge(products_df[["product_id", "price_inr"]], on="product_id", how="left")

missing_amt_mask = orders_df["order_amount"].isna()
calculated_amount = orders_df["price_inr"] * orders_df["quantity"] * (1 - orders_df["discount_pct"] / 100.0)
orders_df.loc[missing_amt_mask, "order_amount"] = calculated_amount.loc[missing_amt_mask].round(2)
orders_df = orders_df.drop(columns=["price_inr"])

# 6. Convert delivery_minutes to numeric
orders_df["delivery_minutes"] = pd.to_numeric(
    orders_df["delivery_minutes"], errors="coerce"
)

# Negative delivery times are invalid
orders_df.loc[
    orders_df["delivery_minutes"] < 0,
    "delivery_minutes"
] = np.nan

# Map product category to each order without changing row order
product_categories = products_df.set_index("product_id")["category"]
orders_category = orders_df["product_id"].map(product_categories)

# Identify completed orders
completed_mask = orders_df["status"] == "completed"

# Calculate category-wise median using completed orders only
category_medians = (
    orders_df.loc[completed_mask]
    .groupby(orders_category[completed_mask])["delivery_minutes"]
    .transform("median")
)

# Fill missing delivery times only for completed orders
orders_df.loc[completed_mask, "delivery_minutes"] = (
    orders_df.loc[completed_mask, "delivery_minutes"]
    .fillna(category_medians)
)

# Cancelled orders remain NaN
print("\nDelivery time validation:")
print("Completed orders:", (orders_df["status"] == "completed").sum())
print("Cancelled orders:", (orders_df["status"] == "cancelled").sum())

print(
    "Missing delivery time in completed orders:",
    orders_df.loc[
        orders_df["status"] == "completed",
        "delivery_minutes"
    ].isna().sum()
)

print(
    "Missing delivery time in cancelled orders:",
    orders_df.loc[
        orders_df["status"] == "cancelled",
        "delivery_minutes"
    ].isna().sum()
)


# Save clean orders and products
products_df.to_csv("data/clean/products.csv", index=False)
orders_df.to_csv("data/clean/orders.csv", index=False)
print(f"Orders cleaned and saved. Total clean rows: {len(orders_df)}")


# Calculate values dynamically
total_orders = len(orders_df)

completed_count = (orders_df["status"] == "completed").sum()
cancelled_count = (orders_df["status"] == "cancelled").sum()

pending_count = (orders_df["status"] == "pending").sum()

amount_total = orders_df.loc[
    orders_df["status"] == "completed",
    "order_amount"
].sum()

# General data-quality validations
assert total_orders == completed_count + cancelled_count + pending_count

assert orders_df["order_amount"].notna().all()

assert orders_df.loc[
    orders_df["status"] == "completed",
    "delivery_minutes"
].notna().all()

assert orders_df.loc[
    orders_df["status"] == "cancelled",
    "delivery_minutes"
].isna().all()

print("Total orders:", total_orders)
print("Completed orders:", completed_count)
print("Cancelled orders:", cancelled_count)
print("Pending orders:", pending_count)
print("Completed order amount:", round(amount_total, 2))

print("\n--- 3. Storing into SQLite Database ---")
db_path = "data/instamart.db"
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
cust_df.to_sql("customers", conn, index=False, if_exists="replace")
products_df.to_sql("products", conn, index=False, if_exists="replace")
orders_df.to_sql("orders", conn, index=False, if_exists="replace")
conn.commit()
conn.close()
print(f"SQLite database created successfully at {db_path}!")
