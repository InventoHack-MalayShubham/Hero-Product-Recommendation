from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
model = None
df = None
sales_df = None
inventory_df = None
last_update = None

# Load dataset and enrich it
def load_data():
    global sales_df, inventory_df, last_update
    # Load sales data
    sales_df = pd.read_csv('D:\\ML Folders\\ml_env\\GitHub\\Hero-Product-Recommendation\\Datasets\\rohit_electronics_sales_1000.csv')
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    
    # Load inventory data
    inventory_df = pd.read_csv('D:\\ML Folders\\ml_env\\GitHub\\Hero-Product-Recommendation\\Datasets\\inventory_data_new1.csv')
    
    # Rename columns to avoid conflicts
    inventory_df = inventory_df.rename(columns={
        'Current Stock': 'Inventory_Current_Stock',
        'Stock Status': 'Inventory_Stock_Status'
    })
    
    # Merge data
    merged_df = pd.merge(sales_df, inventory_df, 
                        left_on=['Product Name', 'Brand Name', 'Product Category'],
                        right_on=['Product Name', 'Brand Name', 'Product Category'],
                        how='left')
    
    last_update = datetime.now()

    df = merged_df.copy()
    df['Revenue_Per_Unit'] = df['Total_Revenue_Incl_GST'] / df['Current Stock'].replace(0, 1)
    df['Popularity'] = df['Rating'] * df['Total_Revenue_Incl_GST']
    df['Stock_Status'] = df['Current Stock'].apply(lambda x: 'Low' if x <= 5 else 'OK')
    return df

# Build the ML model
def build_model(df):
    # Use only available columns
    features = ['Product Category', 'Brand Name', 'Rating', 'Current Stock']
    target = 'Revenue_Per_Unit'

    # Prepare features
    X = df[features].copy()
    y = df[target]

    # Convert categorical variables to numerical
    X = pd.get_dummies(X, columns=['Product Category', 'Brand Name'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2

# Generate analytics plot
def generate_plot():
    plt.figure(figsize=(8, 6))
    category_price = df.groupby('Product Category')['Unit Price'].mean().sort_values(ascending=False)
    sns.barplot(x=category_price.index, y=category_price.values, palette='viridis')
    plt.title('Average Unit Price by Product Category')
    plt.xticks(rotation=90)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return img_b64

def get_category_price_ranges():
    """Calculate price ranges for each category"""
    price_ranges = {}
    for category in ['Mobiles', 'Laptops', 'Mobile Accessories', 'Laptop Accessories']:
        category_data = sales_df[sales_df['Product Category'] == category]
        min_price = category_data['Unit Price'].min()
        max_price = category_data['Unit Price'].max()
        price_ranges[category] = {
            'min': min_price,
            'max': max_price
        }
    return price_ranges

def calculate_product_score(product):
    """Calculate a score for each product based on multiple factors"""
    score = 0
    
    # Rating weight (30%)
    score += product['Rating'] * 0.3
    
    # Revenue weight (40%)
    revenue = product['Total_Revenue_Incl_GST']
    max_revenue = sales_df['Total_Revenue_Incl_GST'].max()
    score += (revenue / max_revenue) * 0.4
    
    # Stock status weight (30%)
    if product['Stock_Status'] == 'OK':
        score += 0.3
    elif product['Stock_Status'] == 'Low':
        score += 0.15
    
    return score

def get_recommendations():
    """Generate automated recommendations for all categories"""
    if sales_df is None or inventory_df is None:
        load_data()
    
    recommendations = {}
    price_ranges = get_category_price_ranges()
    
    for category in ['Mobiles', 'Laptops', 'Mobile Accessories', 'Laptop Accessories']:
        # Filter products by category
        category_products = sales_df[sales_df['Product Category'] == category].copy()
        
        # Filter by price range
        min_price = price_ranges[category]['min']
        max_price = price_ranges[category]['max']
        category_products = category_products[
            (category_products['Unit Price'] >= min_price) &
            (category_products['Unit Price'] <= max_price)
        ]
        
        # Filter by minimum rating
        category_products = category_products[category_products['Rating'] >= 3.7]
        
        # Calculate scores
        category_products['Score'] = category_products.apply(calculate_product_score, axis=1)
        
        # Get top 5 products
        top_products = category_products.nlargest(5, 'Score')
        
        # Format recommendations
        recommendations[category] = []
        for _, product in top_products.iterrows():
            # Calculate units sold from revenue and unit price
            units_sold = product['Total_Revenue_Incl_GST'] / product['Unit Price']
            
            recommendations[category].append({
                'product_name': product['Product Name'],
                'brand': product['Brand Name'],
                'price': product['Unit Price'],
                'rating': product['Rating'],
                'stock_status': product['Stock_Status'],
                'current_stock': product['Current Stock'],
                'total_revenue': product['Total_Revenue_Incl_GST'],
                'units_sold': round(units_sold, 2),
                'score': product['Score']
            })
    
    return recommendations

def bcg_matrix(df, category=None):
    """Generate BCG Matrix recommendations"""
    if category:
        df = df[df['Product Category'] == category]
    
    # Calculate metrics for BCG matrix
    df['Revenue_Per_Unit'] = df['Total_Revenue_Incl_GST'] / df['Current Stock'].replace(0, 1)
    df['Sales'] = df['Total_Revenue_Incl_GST']
    df['Profit'] = df['Revenue_Per_Unit'] * df['Current Stock']
    
    # Calculate medians for classification
    sales_median = df['Sales'].median()
    profit_median = df['Profit'].median()
    
    # Classify products
    stars = df[(df['Sales'] > sales_median) & (df['Profit'] > profit_median)]
    cows = df[(df['Profit'] > profit_median) & (df['Sales'] <= sales_median)]
    question_marks = df[(df['Sales'] <= sales_median) & (df['Profit'] > profit_median)]
    dogs = df[(df['Sales'] <= sales_median) & (df['Profit'] <= profit_median)]
    
    # Format recommendations
    bcg_recommendations = {
        'stars': format_bcg_products(stars),
        'cows': format_bcg_products(cows),
        'question_marks': format_bcg_products(question_marks),
        'dogs': format_bcg_products(dogs)
    }
    
    return bcg_recommendations

def format_bcg_products(df):
    """Format products for BCG matrix display"""
    if df.empty:
        return []
    
    return df.nlargest(5, 'Profit').apply(lambda row: {
        'product_name': row['Product Name'],
        'brand': row['Brand Name'],
        'price': row['Unit Price'],
        'sales': row['Sales'],
        'profit': row['Profit']
    }, axis=1).tolist()

# Initialize
with app.app_context():
    df = load_data()
    model, mae, r2 = build_model(df)
    print(f"Model Initialized → MAE: {mae:.2f}, R2: {r2:.2f}")

# Homepage route
@app.route('/')
def home():
    return render_template('home.html')

# ML model route
@app.route('/model')
def model_page():
    # Generate recommendations
    recommendations = get_recommendations()
    
    # Generate BCG matrix recommendations
    bcg_recommendations = bcg_matrix(sales_df)
    
    # Check if data needs to be reloaded (every 24 hours)
    if last_update is None or (datetime.now() - last_update) > timedelta(hours=24):
        load_data()
        recommendations = get_recommendations()
        bcg_recommendations = bcg_matrix(sales_df)
    
    return render_template('model.html', 
                         recommendations=recommendations,
                         bcg_recommendations=bcg_recommendations)

# Analytics page
@app.route('/analytics')
def analytics():
    if sales_df is None or inventory_df is None:
        load_data()
    
    min_date = sales_df["Date"].min().date()
    max_date = sales_df["Date"].max().date()
    categories = ["All"] + sorted(sales_df["Product Category"].dropna().unique().tolist())
    
    return render_template("analytics.html", min_date=min_date, max_date=max_date, categories=categories)

# Fetch filtered data
@app.route("/get_data", methods=["POST"])
def get_data():
    if sales_df is None or inventory_df is None:
        load_data()
    
    start_date = pd.to_datetime(request.json["start"])
    end_date = pd.to_datetime(request.json["end"])
    category = request.json["category"]

    filtered_df = sales_df[(sales_df["Date"] >= start_date) & (sales_df["Date"] <= end_date)]

    if category != "All":
        filtered_df = filtered_df[filtered_df["Product Category"] == category]

    if filtered_df.empty:
        return jsonify({"message": "No data found for selected filters."})

    daily_revenue = (
        filtered_df.groupby(filtered_df["Date"].dt.strftime('%Y-%m-%d'))["Total_Revenue_Incl_GST"]
        .sum().to_dict()
    )

    revenue_by_category = (
        filtered_df.groupby("Product Category")["Total_Revenue_Incl_GST"]
        .sum().to_dict()
    )

    top_brands = (
        filtered_df.groupby("Brand Name")["Total_Revenue_Incl_GST"]
        .sum().sort_values(ascending=False).head(5).to_dict()
    )

    customer_type = filtered_df["Customer Type"].value_counts().to_dict()
    payment_methods = filtered_df["Payment Method"].value_counts().to_dict()
    
    # Get stock status from sales data
    stock_status = filtered_df["Stock_Status"].value_counts().to_dict()

    return jsonify({
        "dailyRevenue": daily_revenue,
        "revenueByCategory": revenue_by_category,
        "topBrands": top_brands,
        "customerType": customer_type,
        "paymentMethods": payment_methods,
        "stockStatus": stock_status
    })

if __name__ == '__main__':
    load_data()
    app.run(debug=True)
