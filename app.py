from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, Response
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
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Create database tables
with app.app_context():
    db.create_all()

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
        sales_df = pd.read_csv('Datasets/rohit_electronics_sales_1000.csv')
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        
        # Print column names for debugging
        print("Sales DataFrame Columns:", sales_df.columns.tolist())
        
        # Load inventory data
        inventory_df = pd.read_csv('Datasets/inventory_data_new1.csv')
        
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

def get_recommendations(category=None, n=5):
    """Get top n product recommendations for a given category or all categories."""
    if sales_df is None or inventory_df is None:
        load_data()
    
    if category:
        # Get recommendations for a specific category
        category_products = sales_df[sales_df['Product Category'] == category].copy()
        category_products['score'] = category_products.apply(calculate_product_score, axis=1)
        category_products = category_products.sort_values('score', ascending=False)
        category_products = category_products.drop_duplicates(subset=['Product Name'], keep='first')
        top_products = category_products.head(n)
        
        # Format the output for single category
        recommendations = []
        for _, product in top_products.iterrows():
            recommendations.append({
                'product_name': product['Product Name'],
                'brand': product['Brand Name'],
                'price': product['Unit Price'],
                'rating': product['Rating'],
                'stock_status': product['Stock_Status'],
                'current_stock': product['Current Stock'],
                'total_revenue': product['Total_Revenue_Incl_GST'],
                'score': product['score']
            })
    else:
        # Get recommendations for all categories
        recommendations = {}
        for cat in ['Mobiles', 'Laptops', 'Mobile Accessories', 'Laptop Accessories']:
            cat_products = sales_df[sales_df['Product Category'] == cat].copy()
            cat_products['score'] = cat_products.apply(calculate_product_score, axis=1)
            cat_products = cat_products.sort_values('score', ascending=False)
            cat_products = cat_products.drop_duplicates(subset=['Product Name'], keep='first')
            top_products = cat_products.head(n)
            
            # Format products for this category
            cat_recommendations = []
            for _, product in top_products.iterrows():
                cat_recommendations.append({
                    'product_name': product['Product Name'],
                    'brand': product['Brand Name'],
                    'price': product['Unit Price'],
                    'rating': product['Rating'],
                    'stock_status': product['Stock_Status'],
                    'current_stock': product['Current Stock'],
                    'total_revenue': product['Total_Revenue_Incl_GST'],
                    'score': product['score']
                })
            
            recommendations[cat] = cat_recommendations
    
    return recommendations

def bcg_matrix(df, category=None):
    """Generate BCG Matrix recommendations"""
    if category:
        df = df[df['Product Category'] == category]
    
    # Calculate metrics for BCG matrix
    df['Revenue_Per_Unit'] = df['Total_Revenue_Incl_GST'] / df['Current Stock'].replace(0, 1)
    df['Sales'] = df['Total_Revenue_Incl_GST']
    df['Profit'] = df['Revenue_Per_Unit'] * df['Current Stock']
    
    # Group by product and get max values
    df_grouped = df.groupby(['Product Name', 'Brand Name']).agg({
        'Unit Price': 'first',
        'Sales': 'max',
        'Profit': 'max'
    }).reset_index()
    
    # Calculate thresholds for classification
    sales_median = df_grouped['Sales'].median()
    profit_median = df_grouped['Profit'].median()
    
    # Adjust thresholds to be more lenient
    sales_threshold = sales_median * 0.8  # 80% of median
    profit_threshold = profit_median * 0.8  # 80% of median
    
    # Classify products
    stars = df_grouped[
        (df_grouped['Sales'] > sales_threshold) & 
        (df_grouped['Profit'] > profit_threshold)
    ]
    
    cows = df_grouped[
        (df_grouped['Sales'] <= sales_threshold) & 
        (df_grouped['Profit'] > profit_threshold)
    ]
    
    question_marks = df_grouped[
        (df_grouped['Sales'] > sales_threshold) & 
        (df_grouped['Profit'] <= profit_threshold)
    ]
    
    dogs = df_grouped[
        (df_grouped['Sales'] <= sales_threshold) & 
        (df_grouped['Profit'] <= profit_threshold)
    ]
    
    # If any category is empty, redistribute products
    if len(cows) == 0 or len(question_marks) == 0:
        # Sort all products by profit
        all_products = df_grouped.sort_values('Profit', ascending=False)
        
        # Take top products for each category
        n_products = len(all_products)
        if n_products >= 12:  # Ensure we have enough products
            stars = all_products.iloc[:3]
            cows = all_products.iloc[3:6]
            question_marks = all_products.iloc[6:9]
            dogs = all_products.iloc[9:12]
        else:
            # If not enough products, distribute them evenly
            chunk_size = max(1, n_products // 4)
            stars = all_products.iloc[:chunk_size]
            cows = all_products.iloc[chunk_size:2*chunk_size]
            question_marks = all_products.iloc[2*chunk_size:3*chunk_size]
            dogs = all_products.iloc[3*chunk_size:]
    
    # Format recommendations
    bcg_recommendations = {
        'stars': format_bcg_products(stars, n=3),
        'cows': format_bcg_products(cows, n=3),
        'question_marks': format_bcg_products(question_marks, n=3),
        'dogs': format_bcg_products(dogs, n=3)
    }
    
    return bcg_recommendations

def format_bcg_products(df, n=3):
    """Format products for BCG matrix display"""
    if df.empty:
        return []
    
    return df.head(n).apply(lambda row: {
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

def get_top_rated_products(df, n=3):
    """Get top n highest rated products from the dataset."""
    # Filter out products with no ratings
    rated_products = df[df['rating'].notna()]
    # Sort by rating in descending order and get top n
    top_products = rated_products.sort_values('rating', ascending=False).head(n)
    # Format the data for display
    return top_products[['product_name', 'brand', 'price', 'rating', 'reviews']].to_dict('records')

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
@login_required
def model_page():
    try:
        # Load the Amazon dataset
        amazon_df = pd.read_csv('D:/ML Folders/ml_env/GitHub/Hero-Product-Recommendation/Datasets/amazon_data_processed.csv')
        
        # Get top rated products
        top_rated = get_top_rated_products(amazon_df)
        
        # Get recommendations for all categories
        recommendations = get_recommendations()
        
        # Get BCG matrix recommendations
        bcg_recommendations = bcg_matrix(sales_df)
        
        return render_template('model.html',
                             top_rated=top_rated,
                             recommendations=recommendations,
                             bcg_recommendations=bcg_recommendations)
    except Exception as e:
        flash('Error loading model data. Please try again later.', 'danger')
        return redirect(url_for('home'))

# Analytics page
@app.route('/analytics')
@login_required
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

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return redirect(url_for('signup'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))

if __name__ == '__main__':
    load_data()
    app.run(debug=True)
