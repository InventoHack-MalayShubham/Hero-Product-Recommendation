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
    try:
        # Load sales data
        sales_df = pd.read_csv('D:\\ML Folders\\ml_env\\GitHub\\Hero-Product-Recommendation\\Datasets\\rohit_electronics_sales_1000.csv')
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        
        # Print column names for debugging
        print("Sales DataFrame Columns:", sales_df.columns.tolist())
        
        # Load inventory data
        inventory_df = pd.read_csv('D:\\ML Folders\\ml_env\\GitHub\\Hero-Product-Recommendation\\Datasets\\inventory_data_new1.csv')
        
        # Print column names for debugging
        print("Inventory DataFrame Columns:", inventory_df.columns.tolist())
        
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
        return merged_df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Build the ML model
def build_model(df):
    # Use only available columns
    features = ['Product Category', 'Brand Name', 'Rating', 'Current Stock']
    target = 'Total_Revenue_Incl_GST'

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

def calculate_kpis(filtered_sales, comparison_sales=None):
    """Calculate key performance indicators from sales data"""
    try:
        # Print column names for debugging
        print("Filtered Sales Columns:", filtered_sales.columns.tolist())
        
        # Calculate basic KPIs
        total_revenue = filtered_sales['Total_Revenue_Incl_GST'].sum()
        total_orders = filtered_sales['Transaction ID'].nunique()  # Using Transaction ID instead of Order_ID
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        # Calculate stock turnover rate
        total_quantity = filtered_sales['Current Stock'].sum()  # Using Current Stock directly
        avg_stock = filtered_sales['Current Stock'].mean()
        stock_turnover = total_quantity / avg_stock if avg_stock > 0 else 0
        
        # Calculate customer retention rate
        total_customers = filtered_sales['Customer Type'].nunique()  # Using Customer Type instead of Customer_ID
        repeat_customers = filtered_sales.groupby('Customer Type').size()
        repeat_customers = (repeat_customers > 1).sum()
        retention_rate = (repeat_customers / total_customers * 100) if total_customers > 0 else 0
        
        # Find most popular product
        popular_product = filtered_sales.groupby('Product Name')['Total_Revenue_Incl_GST'].sum().idxmax()
        
        # Calculate growth rate if comparison data is available
        growth_rate = 0
        if comparison_sales is not None:
            prev_revenue = comparison_sales['Total_Revenue_Incl_GST'].sum()
            growth_rate = ((total_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
        
        # Calculate inventory value
        inventory_value = filtered_sales['Current Stock'].sum() * filtered_sales['Unit Price'].mean()
        
        return {
            'total_revenue': total_revenue,
            'avg_order_value': avg_order_value,
            'total_orders': total_orders,
            'stock_turnover': stock_turnover,
            'retention_rate': retention_rate,
            'popular_product': popular_product,
            'growth_rate': growth_rate,
            'inventory_value': inventory_value
        }
        
    except Exception as e:
        print(f"Error calculating KPIs: {str(e)}")
        return {
            'total_revenue': 0,
            'avg_order_value': 0,
            'total_orders': 0,
            'stock_turnover': 0,
            'retention_rate': 0,
            'popular_product': '-',
            'growth_rate': 0,
            'inventory_value': 0
        }

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
@app.route('/get_data', methods=['POST'])
def get_data():
    try:
        data = request.json
        date_range = int(data.get('dateRange', 30))  # Default to 30 days if not specified
        category = data.get('category', 'all')
        compare_period = data.get('comparePeriod', 'none')

        # Calculate date range
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=date_range)

        # Filter data based on date range and category
        filtered_sales = sales_df[
            (sales_df['Date'] >= start_date) & 
            (sales_df['Date'] <= end_date)
        ]
        
        if category != 'all':
            filtered_sales = filtered_sales[filtered_sales['Product Category'] == category]

        # Calculate comparison data if requested
        comparison_sales = None
        if compare_period != 'none':
            if compare_period == 'previous':
                comp_start = start_date - pd.Timedelta(days=date_range)
                comp_end = start_date
            else:  # year
                comp_start = start_date - pd.Timedelta(days=365)
                comp_end = end_date - pd.Timedelta(days=365)
            
            comparison_sales = sales_df[
                (sales_df['Date'] >= comp_start) & 
                (sales_df['Date'] <= comp_end)
            ]
            
            if category != 'all':
                comparison_sales = comparison_sales[comparison_sales['Product Category'] == category]

        # Calculate KPIs
        kpis = calculate_kpis(filtered_sales, comparison_sales if compare_period != 'none' else None)

        # Prepare chart data
        chart_data = {
            'revenueByCategory': filtered_sales.groupby('Product Category')['Total_Revenue_Incl_GST'].sum().to_dict(),
            'dailyRevenue': filtered_sales.groupby(filtered_sales['Date'].dt.strftime('%Y-%m-%d'))['Total_Revenue_Incl_GST'].sum().to_dict(),
            'topBrands': filtered_sales.groupby('Brand Name')['Total_Revenue_Incl_GST'].sum().nlargest(5).to_dict(),
            'customerType': filtered_sales.groupby('Customer Type')['Transaction ID'].count().to_dict(),
            'stockStatus': filtered_sales.groupby('Stock_Status')['Transaction ID'].count().to_dict(),
            'priceRange': filtered_sales.groupby('Unit Price')['Transaction ID'].count().to_dict()
        }

        return jsonify({
            'kpis': kpis,
            'charts': chart_data
        })

    except Exception as e:
        print(f"Error in get_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_data()
    app.run(debug=True)
