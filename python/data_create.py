import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
import os

fake = Faker()
Faker.seed(42)
random.seed(42)

# Constants
brands = {
    "Smartphones": ["Apple", "Samsung", "Xiaomi", "Realme", "Vivo", "Nothing"],
    "Feature Phones": ["Jio"],
    "Gaming": ["ASUS", "MSI", "HP"],
    "Business": ["Dell", "HP", "Lenovo"],
    "Student": ["Acer", "Lenovo", "HP"],
    "Mobile Accessories": ["Boat", "Realme", "Samsung", "Mi"],
    "Laptop Accessories": ["Logitech", "HP", "Dell", "Zebronics"]
}

product_types = {
    "Mobiles": ["Smartphones", "Feature Phones"],
    "Laptops": ["Gaming", "Business", "Student"],
    "Mobile Accessories": ["Chargers", "Earphones", "Screen Protectors", "Cases"],
    "Laptop Accessories": ["Mouse", "Keyboard", "Bags", "Cooling Pads"]
}

gst_rates = {
    "Smartphones": 0.18,
    "Feature Phones": 0.12,
    "Gaming": 0.18,
    "Business": 0.18,
    "Student": 0.18,
    "Mobile Accessories": 0.18,
    "Laptop Accessories": 0.18
}

# Generate price ranges per brand
brand_price_range = {
    "Apple": (65000, 130000),
    "Samsung": (20000, 80000),
    "Xiaomi": (8000, 20000),
    "Realme": (8000, 20000),
    "Vivo": (8000, 20000),
    "Nothing": (25000, 50000),
    "Jio": (1000, 3000),
    "ASUS": (70000, 150000),
    "MSI": (80000, 160000),
    "HP": (30000, 100000),
    "Dell": (35000, 90000),
    "Lenovo": (30000, 85000),
    "Acer": (25000, 70000),
    "Boat": (500, 2000),
    "Mi": (500, 2000),
    "Logitech": (700, 3000),
    "Zebronics": (600, 2500)
}

payment_methods = ["Cash", "UPI", "Card"]
customer_types = ["New", "Returning"]

def generate_transaction_id(index):
    return f"TXN{100000 + index}"

def generate_data_for_day(date, transactions_per_day=30):
    data = []
    for i in range(transactions_per_day):
        main_category = random.choice(list(product_types.keys()))
        subcategory = random.choice(product_types[main_category])
        brand = random.choice(brands.get(subcategory, brands.get(main_category, ["Generic"])))
        product_name = f"{brand} {subcategory} {random.randint(100, 999)}"
        units_sold = random.randint(1, 5) if main_category in ["Mobiles", "Laptops"] else random.randint(1, 15)
        unit_price = random.randint(*brand_price_range.get(brand, (500, 2000)))
        discount = random.choices([0, 5, 10, 15, 20], weights=[50, 20, 15, 10, 5])[0]

        # Festival offer boost
        if date.month == 1 and date.day in range(24, 27):  # Republic Day
            discount += 5
        elif date.month == 3 and date.day in range(15, 18):  # Holi
            discount += 5
        discount = min(discount, 30)

        discounted_price = unit_price * (1 - discount / 100)
        total_revenue = discounted_price * units_sold
        gst_rate = gst_rates.get(subcategory, 0.18)
        gst_amount = total_revenue * gst_rate
        total_revenue_gst = total_revenue + gst_amount

        current_stock = random.randint(0, 100)
        stock_status = "In Stock" if current_stock > 30 else "Low Stock" if current_stock > 0 else "Out of Stock"

        data.append({
            "Transaction ID": generate_transaction_id(i),
            "Date": date.strftime("%Y-%m-%d"),
            "Product Category": main_category,
            "Subcategory": subcategory,
            "Product Name": product_name,
            "Brand Name": brand,
            "Units Sold": units_sold,
            "Unit Price": unit_price,
            "Discount (%)": discount,
            "Discounted Price": round(discounted_price, 2),
            "Total Revenue": round(total_revenue, 2),
            "GST_Percentage": int(gst_rate * 100),
            "GST_Amount": round(gst_amount, 2),
            "Total_Revenue_Incl_GST": round(total_revenue_gst, 2),
            "Current Stock": current_stock,
            "Stock_Status": stock_status,
            "Rating": round(random.uniform(2.5, 5.0), 1),
            "Returns": random.choices([0, 1], weights=[90, 10])[0],
            "Customer Type": random.choice(customer_types),
            "Region": "Main Market - Rohit Electronics",
            "Payment Method": random.choice(payment_methods)
        })
    return data

# Generate for past 3 months
end_date = datetime.now().date()
start_date = end_date - timedelta(days=90)
all_data = []
current_date = start_date

while current_date <= end_date:
    transactions_today = random.randint(20, 50)
    all_data.extend(generate_data_for_day(current_date, transactions_today))
    current_date += timedelta(days=1)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Save the CSV
dataset_dir = "D:\ML Folders\ml_env\GitHub\Hero-Product-Recommendation\Datasets"
os.makedirs(dataset_dir, exist_ok=True)
csv_path = os.path.join(dataset_dir, "rohit_electronics_sales_data.csv")
df.to_csv(csv_path, index=False)

csv_path

