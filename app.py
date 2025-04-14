from flask import Flask, render_template, jsonify, request
import os
import json
import csv
from io import StringIO
import platform

app = Flask(__name__)

# Sample product data (replace with database later)
SAMPLE_PRODUCTS = [
    {
        "id": 1,
        "name": "Cola",
        "category": "Beverages",
        "stock": 15,
        "image_url": "https://placehold.co/200x200?text=Cola",
        "expiry_date": "2024-12-31"
    },
    {
        "id": 2,
        "name": "Chips",
        "category": "Snacks",
        "stock": 5,
        "image_url": "https://placehold.co/200x200?text=Chips",
        "expiry_date": "2024-06-30"
    },
    {
        "id": 3,
        "name": "Chocolate",
        "category": "Candies",
        "stock": 20,
        "image_url": "https://placehold.co/200x200?text=Chocolate",
        "expiry_date": "2024-08-15"
    },
    {
        "id": 4,
        "name": "Water",
        "category": "Beverages",
        "stock": 30,
        "image_url": "https://placehold.co/200x200?text=Water",
        "expiry_date": "2025-01-01"
    },
    {
        "id": 5,
        "name": "Cookies",
        "category": "Snacks",
        "stock": 8,
        "image_url": "https://placehold.co/200x200?text=Cookies",
        "expiry_date": "2024-07-20"
    },
    {
        "id": 6,
        "name": "Gum",
        "category": "Candies",
        "stock": 25,
        "image_url": "https://placehold.co/200x200?text=Gum",
        "expiry_date": "2024-10-10"
    },
    {
        "id": 7,
        "name": "Juice",
        "category": "Beverages",
        "stock": 12,
        "image_url": "https://placehold.co/200x200?text=Juice",
        "expiry_date": "2024-05-15"
    },
    {
        "id": 8,
        "name": "Pretzels",
        "category": "Snacks",
        "stock": 10,
        "image_url": "https://placehold.co/200x200?text=Pretzels",
        "expiry_date": "2024-09-30"
    },
    {
        "id": 9,
        "name": "Lollipop",
        "category": "Candies",
        "stock": 40,
        "image_url": "https://placehold.co/200x200?text=Lollipop",
        "expiry_date": "2024-11-30"
    },
    {
        "id": 10,
        "name": "Soda",
        "category": "Beverages",
        "stock": 18,
        "image_url": "https://placehold.co/200x200?text=Soda",
        "expiry_date": "2024-12-15"
    },
    {
        "id": 11,
        "name": "Nuts",
        "category": "Snacks",
        "stock": 7,
        "image_url": "https://placehold.co/200x200?text=Nuts",
        "expiry_date": "2024-08-01"
    },
    {
        "id": 12,
        "name": "Mints",
        "category": "Candies",
        "stock": 35,
        "image_url": "https://placehold.co/200x200?text=Mints",
        "expiry_date": "2024-10-20"
    },
    {
        "id": 13,
        "name": "Tea",
        "category": "Beverages",
        "stock": 22,
        "image_url": "https://placehold.co/200x200?text=Tea",
        "expiry_date": "2025-02-28"
    },
    {
        "id": 14,
        "name": "Popcorn",
        "category": "Snacks",
        "stock": 15,
        "image_url": "https://placehold.co/200x200?text=Popcorn",
        "expiry_date": "2024-07-10"
    },
    {
        "id": 15,
        "name": "Candy Bar",
        "category": "Candies",
        "stock": 28,
        "image_url": "https://placehold.co/200x200?text=CandyBar",
        "expiry_date": "2024-09-15"
    }
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/products')
def get_products():
    # Simulated pagination
    page = int(request.args.get('page', 1))
    per_page = 12
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Filter products by search term if provided
    search_term = request.args.get('search', '').lower()
    filtered_products = SAMPLE_PRODUCTS
    
    if search_term:
        filtered_products = [
            product for product in SAMPLE_PRODUCTS
            if search_term in product['name'].lower() or 
               search_term in product['category'].lower()
        ]
    
    products_page = filtered_products[start_idx:end_idx]
    return jsonify({
        'products': products_page,
        'has_more': end_idx < len(filtered_products)
    })

@app.route('/api/update_stock', methods=['POST'])
def update_stock():
    data = request.json
    product_id = data.get('product_id')
    delta = data.get('delta')
    
    # Find the product and update its stock
    for product in SAMPLE_PRODUCTS:
        if product['id'] == product_id:
            product['stock'] += delta
            return jsonify({'success': True})
    
    return jsonify({'success': False, 'message': 'Product not found'})

@app.route('/api/import_csv', methods=['POST'])
def import_csv():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'message': 'File must be a CSV'})
    
    # In a real application, you would process the CSV file
    # For this demo, we'll just return success
    return jsonify({'success': True})

def run_app():
    """Run the Flask application with appropriate settings based on Python version"""
    if platform.python_version_tuple()[0:2] >= ('3', '12'):
        # For Python 3.12+, disable the reloader to avoid signal errors
        app.run(debug=True, use_reloader=False)
    else:
        # For older Python versions, use the standard approach
        app.run(debug=True)

if __name__ == '__main__':
    run_app() 