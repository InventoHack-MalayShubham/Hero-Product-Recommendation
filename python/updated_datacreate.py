import os
import random
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

# Ensure the Datasets directory exists
output_dir = "Datasets"
os.makedirs(output_dir, exist_ok=True)

# Product structure
categories = {
    'Mobiles': ['Smartphone', 'Feature Phone'],
    'Laptop Accessories': ['Mouse', 'Keyboard', 'Charger', 'Bag'],
    'Mobile Accessories': ['Earbuds', 'Cable', 'Charger', 'Case'],
    'Laptops': ['Gaming Laptop', 'Ultrabook', 'Notebook']
}

brands = {
    'Mobiles': ['Apple', 'Samsung', 'Xiaomi', 'Realme', 'Jio', 'Vivo', 'Nothing', 'Mi'],
    'Laptops': ['HP', 'Dell', 'MSI', 'ASUS', 'Acer', 'Lenovo'],
    'Mobile Accessories': ['Boat', 'Zebronics', 'Realme', 'Logitech'],
    'Laptop Accessories': ['Logitech', 'Zebronics', 'HP', 'Dell']
}

# Generate 50 unique products only
unique_products = []
product_id = 1
while len(unique_products) < 50:
    category = random.choice(list(categories.keys()))
    subcategory = random.choice(categories[category])
    brand = random.choice(brands[category])
    name = f"{brand} {subcategory} {product_id}"
    if name not in unique_products:
        unique_products.append({
            'Category': category,
            'Subcategory': subcategory,
            'Brand': brand,
            'ProductName': name,
            'Price': random.randint(5000, 120000) if category == 'Mobiles' else
                     random.randint(25000, 180000) if category == 'Laptops' else
                     random.randint(200, 5000),
            'Stock': random.randint(15, 60)
        })
        product_id += 1

# Helper for GST
def get_gst(category):
    return 12 if category == 'Mobiles' else 18

# Ratings and returns logic
def generate_rating():
    return round(random.uniform(1.0, 5.0), 1)

def generate_return(rating):
    if rating <= 2.0:
        return random.choices([0, 1], weights=[60, 40])[0]
    return random.choices([0, 1], weights=[95, 5])[0]

# Customer and payment method
customer_types = ['New', 'Returning', 'Regular', 'Wholesale']
payment_methods = ['Cash', 'UPI', 'Credit Card', 'Debit Card', 'Net Banking']
customer_weights = [0.3, 0.25, 0.35, 0.1]
payment_weights = [0.15, 0.4, 0.2, 0.2, 0.05]

# Generate transactions
records = []
transaction_id = 1

while len(records) < 1000:
    product = random.choice(unique_products)
    if product['Stock'] <= 0:
        continue  # Skip out-of-stock

    transaction_code = f"T{str(transaction_id).zfill(4)}"
    date = fake.date_time_between(start_date='-30d', end_date='now')
    unit_price = product['Price']
    discount = random.randint(0, 30)
    discounted_price = round(unit_price * (1 - discount / 100), 2)
    gst_percent = get_gst(product['Category'])
    gst_amount = round(discounted_price * gst_percent / 100, 2)
    total_incl_gst = round(discounted_price + gst_amount, 2)
    stock = product['Stock'] - 1
    stock_status = 'OK' if stock > 10 else 'Low Stock' if stock > 0 else 'Out of Stock'
    rating = generate_rating()
    returned = generate_return(rating)
    cust_type = random.choices(customer_types, weights=customer_weights)[0]
    payment = random.choices(payment_methods, weights=payment_weights)[0]

    # Append record
    records.append({
        'Transaction ID': transaction_code,
        'Date': date.strftime('%Y-%m-%d %H:%M:%S'),
        'Product Category': product['Category'],
        'Subcategory': product['Subcategory'],
        'Product Name': product['ProductName'],
        'Brand Name': product['Brand'],
        'Unit Price': unit_price,
        'Discount (%)': discount,
        'Discounted Price': discounted_price,
        'Total Revenue': discounted_price,
        'GST_Percentage': gst_percent,
        'GST_Amount': gst_amount,
        'Total_Revenue_Incl_GST': total_incl_gst,
        'Current Stock': stock,
        'Stock_Status': stock_status,
        'Rating': rating,
        'Returns': returned,
        'Customer Type': cust_type,
        'Payment Method': payment
    })

    product['Stock'] -= 1
    transaction_id += 1

# Save to CSV at specified path
df = pd.DataFrame(records)
file_path = r"D:\ML Folders\ml_env\GitHub\Hero-Product-Recommendation\Datasets\rohit_electronics_sales.csv"
df.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}")

# # Save to CSV in current directory
# df = pd.DataFrame(records)
# file_path = "rohit_electronics_sales.csv"
# df.to_csv(file_path, index=False)
# print(f"Dataset saved to {file_path}")