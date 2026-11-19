
import pandas as pd
import numpy as np

pd.set_option("display.width", 150)
pd.set_option("display.float_format", "{:,.2f}".format)

# 1. LOAD AND INSPECT
df = pd.read_csv("/22_restaurant_orders.csv")
print("1. Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.dtypes, "\n")

# 2. MISSING VALUES AND DUPLICATE ORDER_ID
print("2. Missing values:\n", df.isna().sum(), "\n")
print("Duplicate Order_ID records:", df["Order_ID"].duplicated().sum())
print("Fully identical duplicate rows:", df.duplicated().sum(), "\n")
df = df.drop_duplicates(subset="Order_ID", keep="first").reset_index(drop=True)
print("Shape after removing duplicates:", df.shape, "\n")

# 3. CLEAN MISSING QUANTITY / PRICE
# Method: fill with the median of the same dish (median resists extreme values);
# Quantity is rounded because items are whole numbers.
df["Quantity"] = df["Quantity"].fillna(
    df.groupby("Dish_Name")["Quantity"].transform("median")).round().astype(int)
df["Unit_Price"] = df["Unit_Price"].fillna(
    df.groupby("Dish_Name")["Unit_Price"].transform("median"))
print("3. Missing Quantity/Unit_Price after cleaning:",
      df[["Quantity", "Unit_Price"]].isna().sum().sum(), "\n")

# 4. DATETIME, HOUR, WEEKDAY
# The file has Order_Date and Order_Hour (no single Order_DateTime column), so combine them.
df["Order_DateTime"] = pd.to_datetime(df["Order_Date"]) + pd.to_timedelta(df["Order_Hour"], unit="h")
df["Hour"] = df["Order_DateTime"].dt.hour
df["Weekday"] = df["Order_DateTime"].dt.day_name()

# 6. ITEM REVENUE (also used as the bill, since each Order_ID is one row)
df["Item_Revenue"] = df["Quantity"] * df["Unit_Price"]

# 5. TOTALS
print("5. Total orders :", df["Order_ID"].nunique())
print("   Total items  :", df["Quantity"].sum())
print(f"   Total revenue: {df['Item_Revenue'].sum():,.2f}\n")

# 7. TOP 10 MOST FREQUENTLY ORDERED DISHES
popular = df["Dish_Name"].value_counts().head(10)
print("7. Most frequently ordered dishes (by number of orders):\n", popular, "\n")

# 8. CATEGORIES BY REVENUE
category_revenue = df.groupby("Category")["Item_Revenue"].sum().sort_values(ascending=False)
print("8. Category revenue ranking:\n", category_revenue, "\n")

# 9. AVERAGE BILL BY ORDER TYPE
avg_bill = df.groupby("Delivery_Mode")["Item_Revenue"].mean().sort_values(ascending=False)
print("9. Average bill by Delivery_Mode:\n", avg_bill, "\n")

# 10. PEAK HOURS AND BUSIEST DAY
hour_counts = df["Hour"].value_counts().sort_index()
weekday_counts = df["Weekday"].value_counts()
print("10. Orders per hour:\n", hour_counts)
print("Peak hours (top 3):", hour_counts.sort_values(ascending=False).head(3).index.tolist())
print("Busiest weekday:", weekday_counts.idxmax(), f"({weekday_counts.max()} orders)\n")

# 11. LIST AND TUPLE
top_dishes = popular.index.tolist()
meal_categories = tuple(sorted(df["Category"].unique()))
print("11. Top dishes (list):", top_dishes)
print("    Meal categories (tuple):", meal_categories, "\n")

# 12. SET OF CUSTOMERS + BILL FUNCTION
# The dataset has no Customer_ID column, so unique restaurants are used as the closest identifier.
if "Customer_ID" in df.columns:
    unique_ids = set(df["Customer_ID"])
    print("12. Unique customer IDs:", len(unique_ids))
else:
    unique_ids = set(df["Restaurant_Name"])
    print("12. No Customer_ID column in this file; unique restaurants:", unique_ids)


def total_bill(order_id, data=df):
    """Total bill (Quantity x Unit_Price summed over the order's rows) for an Order_ID."""
    rows = data[data["Order_ID"] == order_id]
    if rows.empty:
        return f"Order '{order_id}' not found"
    return round(rows["Item_Revenue"].sum(), 2)


print("    total_bill('FO000000') =", total_bill("FO000000"))
print("    total_bill('XX') =", total_bill("XX"), "\n")

# 13. NUMPY STATS
bills = df["Item_Revenue"].to_numpy()
print("13. Bill value (NumPy)")
print(f"Mean  : {np.mean(bills):,.2f}")
print(f"Median: {np.median(bills):,.2f}")
print(f"Max   : {np.max(bills):,.2f}\n")

# 14. IQR OUTLIERS + ORDER-TIME CATEGORIES
q1, q3 = np.percentile(bills, [25, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["Item_Revenue"] > upper) | (df["Item_Revenue"] < lower)] \
    .sort_values("Item_Revenue", ascending=False)
print("14. IQR method")
print(f"Q1={q1:,.2f}  Q3={q3:,.2f}  IQR={iqr:,.2f}  Upper fence={upper:,.2f}")
print("Unusually large bills:", len(outliers))
if len(outliers):
    print(outliers[["Order_ID", "Dish_Name", "Quantity", "Unit_Price", "Item_Revenue"]]
          .to_string(index=False))


def time_category(hour):
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"


df["Time_Category"] = df["Hour"].apply(time_category)
print("\nOrders by time category:\n",
      df["Time_Category"].value_counts().reindex(["Morning", "Afternoon", "Evening", "Night"]), "\n")


# 15. MENUITEM CLASS + RECOMMENDATIONS
class MenuItem:
    def __init__(self, name, category, price):
        self.name = name
        self.category = category
        self.price = price

    def __repr__(self):
        return f"MenuItem(name='{self.name}', category='{self.category}', price={self.price:,.2f})"


menu = [MenuItem(d, df.loc[df["Dish_Name"] == d, "Category"].mode()[0],
                 df.loc[df["Dish_Name"] == d, "Unit_Price"].mean())
        for d in df["Dish_Name"].unique()]
print("15. Menu items (category = most common one in the data):")
for m in menu:
    print("  ", m)

top_dish, low_dish = popular.index[0], popular.index[-1]
best_cat, worst_cat = category_revenue.index[0], category_revenue.index[-1]
best_mode = avg_bill.index[0]
peak_hours = hour_counts.sort_values(ascending=False).head(3).index.tolist()
best_time = df["Time_Category"].value_counts().idxmax()

recommendations = [
    f"Feature '{top_dish}' (most ordered) prominently on the menu and keep its ingredients well stocked.",
    f"Review or rework '{low_dish}' (least ordered of the top list) with a promotion or a new recipe before removing it.",
    f"Expand options in '{best_cat}' (highest revenue) and use combo offers to lift '{worst_cat}' (lowest revenue).",
    f"Staff and prepare for peak hours {sorted(peak_hours)} and the busiest day ({weekday_counts.idxmax()}); "
    f"run happy-hour deals in quieter hours. Most orders fall in the {best_time}.",
    f"'{best_mode}' orders have the highest average bill; promote add-ons and bundles there, and "
    f"offer upsell combos to push other order types toward that level.",
]
print("\nMENU-PLANNING RECOMMENDATIONS")
for i, r in enumerate(recommendations, 1):
    print(f"{i}. {r}")

# SAVE CLEANED DATA
df.to_csv("cleaned_restaurant_orders.csv", index=False)
print("\nSaved: cleaned_restaurant_orders.csv")
