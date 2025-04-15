from faker import Faker
import pandas as pd
import random

# Initialize Faker
fake = Faker('en_IN')  # Use Indian locale for more realistic data

# List of products with subcategories and brands
products = [
    ('Apple', 'Smartphones', 'Mobiles'),
    ('Samsung', 'Smartphones', 'Mobiles'),
    ('Xiaomi', 'Smartphones', 'Mobiles'),
    ('Realme', 'Smartphones', 'Mobiles'),
    ('Vivo', 'Smartphones', 'Mobiles'),
    ('Nothing', 'Smartphones', 'Mobiles'),
    ('Jio', 'Feature Phones', 'Mobiles'),
    ('ASUS', 'Gaming', 'Laptops'),
    ('MSI', 'Gaming', 'Laptops'),
    ('HP', 'Business', 'Laptops'),
    ('Dell', 'Business', 'Laptops'),
    ('Lenovo', 'Business', 'Laptops'),
    ('Acer', 'Student', 'Laptops'),
    ('Boat', 'Chargers', 'Mobile Accessories'),
    ('Realme', 'Earphones', 'Mobile Accessories'),
    ('Samsung', 'Screen Protectors', 'Mobile Accessories'),
    ('Mi', 'Cases', 'Mobile Accessories'),
    ('Logitech', 'Mouse', 'Laptop Accessories'),
    ('HP', 'Keyboard', 'Laptop Accessories'),
    ('Dell', 'Bags', 'Laptop Accessories'),
    ('Zebronics', 'Cooling Pads', 'Laptop Accessories')
]

# Generate random suppliers
suppliers = [fake.company() for _ in range(10)]

# Function to generate inventory data
def generate_inventory_data(products):
    inventory_data = []

    for brand, subcategory, category in products:
        product_id = fake.unique.random_int(min=1000, max=9999)
        product_name = f"{brand} {subcategory} {fake.random_int(min=1, max=100)}"
        
        # Set realistic price ranges based on brand and category
        if category == 'Mobiles':
            if subcategory == 'Feature Phones':
                unit_price = round(random.uniform(1000, 3000), 2)
            else:
                unit_price = round(random.uniform(65000, 130000), 2)
        elif category == 'Laptops':
            unit_price = round(random.uniform(20000, 100000), 2)
        else:  # Mobile Accessories and Laptop Accessories
            unit_price = round(random.uniform(500, 5000), 2)
        
        cost_price = round(unit_price * random.uniform(0.7, 0.9), 2)
        selling_price = round(unit_price * random.uniform(1.1, 1.3), 2)
        
        # Apply GST based on subcategory
        gst = 12 if subcategory == 'Feature Phones' else 18
        
        # Discount logic with festival boosts
        discount = random.choices([0, 5, 10, 15, 20], weights=[50, 20, 15, 10, 5])[0]
        
        current_stock = random.randint(0, 100)
        reorder_level = random.randint(5, 20)
        reorder_quantity = random.randint(10, 50)
        stock_status = 'Low' if current_stock < 10 else 'In Stock' if current_stock > 0 else 'Out of Stock'
        
        supplier = random.choice(suppliers)
        lead_time = random.randint(1, 7)

        inventory_data.append({
            'Product ID': product_id,
            'Product Name': product_name,
            'Brand Name': brand,
            'Product Category': category,
            'Subcategory': subcategory,
            'Current Stock': current_stock,
            'Reorder Level': reorder_level,
            'Reorder Quantity': reorder_quantity,
            'Stock Status': stock_status,
            'Unit Price': unit_price,
            'Cost Price': cost_price,
            'Selling Price': selling_price,
            'Discount (%)': discount,
            'GST (%)': gst,
            'Supplier Name': supplier,
            'Lead Time': lead_time
        })

    return pd.DataFrame(inventory_data)

# Generate the inventory dataset
inventory_df = generate_inventory_data(products)

# Display the generated inventory data
print(inventory_df)
import os
# Save the CSV
dataset_dir = "D:\ML Folders\ml_env\GitHub\Hero-Product-Recommendation\Datasets"
os.makedirs(dataset_dir, exist_ok=True)
csv_path = os.path.join(dataset_dir, "inventory_data_new1.csv")
inventory_df.to_csv(csv_path, index=False)