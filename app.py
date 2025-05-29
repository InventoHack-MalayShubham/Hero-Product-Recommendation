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
# import matplotlib.pyplot as plt
# import seaborn as sns
import io
import base64
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hero_inventory.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_name = db.Column(db.String(200), nullable=False)
    brand_name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    subcategory = db.Column(db.String(100), nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    cost_price = db.Column(db.Float, nullable=False)
    gst_percentage = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    inventory = db.relationship('Inventory', backref='product', lazy=True)
    transactions = db.relationship('Transaction', backref='product', lazy=True)

class Inventory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    current_stock = db.Column(db.Integer, nullable=False)
    reorder_level = db.Column(db.Integer, nullable=False)
    reorder_quantity = db.Column(db.Integer, nullable=False)
    stock_status = db.Column(db.String(50), nullable=False)
    supplier_name = db.Column(db.String(200))
    lead_time = db.Column(db.Integer)  # in days
    last_restocked = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(50), unique=True, nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    units_sold = db.Column(db.Integer, nullable=False)
    discount_percentage = db.Column(db.Float, default=0)
    discounted_price = db.Column(db.Float, nullable=False)
    total_revenue = db.Column(db.Float, nullable=False)
    gst_amount = db.Column(db.Float, nullable=False)
    total_revenue_incl_gst = db.Column(db.Float, nullable=False)
    customer_type = db.Column(db.String(50))
    payment_method = db.Column(db.String(50))
    rating = db.Column(db.Float)
    returns = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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
    
    # Create default admin user if not exists
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@heroinventory.com',
            is_admin=True
        )
        admin.set_password('admin123')  # Change this password in production
        db.session.add(admin)
        db.session.commit()

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
        print("Loading sales data...")
        sales_df = pd.read_csv('Datasets/rohit_electronics_sales_1000.csv')
        print(f"Sales data loaded with shape: {sales_df.shape}")
        
        # Convert date column
        print("Converting date column...")
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        
        # Print column names for debugging
        print("Sales DataFrame Columns:", sales_df.columns.tolist())
        
        # Load inventory data
        print("Loading inventory data...")
        inventory_df = pd.read_csv('Datasets/inventory_data_new1.csv')
        print(f"Inventory data loaded with shape: {inventory_df.shape}")
        
        # Print column names for debugging
        print("Inventory DataFrame Columns:", inventory_df.columns.tolist())
        
        # Rename columns to avoid conflicts
        print("Renaming columns...")
        inventory_df = inventory_df.rename(columns={
            'Current Stock': 'Inventory_Current_Stock',
            'Stock Status': 'Inventory_Stock_Status'
        })
        
        # Merge data
        print("Merging datasets...")
        merged_df = pd.merge(sales_df, inventory_df, 
                           left_on=['Product Name', 'Brand Name', 'Product Category'],
                           right_on=['Product Name', 'Brand Name', 'Product Category'],
                           how='left')
        print(f"Merged dataset shape: {merged_df.shape}")
        
        # Check for missing values
        print("\nMissing values in merged dataset:")
        print(merged_df.isnull().sum())
        
        last_update = datetime.now()
        return merged_df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return None

def load_inventory_dataset():
    try:
        print("Loading inventory dataset...")
        df = pd.read_csv('generated_1000_inventory_dataset.csv')
        print(f"Inventory dataset loaded with shape: {df.shape}")
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        
        with app.app_context():
            # Clear existing data
            Transaction.query.delete()
            Inventory.query.delete()
            Product.query.delete()
            db.session.commit()
            
            # Process each row
            for _, row in df.iterrows():
                # Create or get product
                product = Product(
                    product_name=row['Product Name'],
                    brand_name=row['Brand Name'],
                    category=row['Product Category'],
                    subcategory=row['Subcategory'],
                    unit_price=float(row['Unit Price']),
                    cost_price=float(row['Cost Price']),
                    gst_percentage=float(row['GST (%)']),
                    description=f"{row['Brand Name']} {row['Product Name']} - {row['Product Category']} {row['Subcategory']}"
                )
                db.session.add(product)
                db.session.flush()  # Get the product ID
                
                # Create inventory record
                inventory = Inventory(
                    product_id=product.id,
                    current_stock=int(row['Current Stock']),
                    reorder_level=int(row['Reorder Level']),
                    reorder_quantity=int(row['Reorder Quantity']),
                    stock_status=row['Stock Status'],
                    supplier_name=row['Supplier Name'],
                    lead_time=int(row['Lead Time']),
                    last_restocked=row['Date']
                )
                db.session.add(inventory)
                
                # Create transaction record
                transaction = Transaction(
                    transaction_id=row['Transaction ID'],
                    product_id=product.id,
                    date=row['Date'],
                    units_sold=1,  # Assuming 1 unit per transaction
                    discount_percentage=float(row['Discount (%)']),
                    discounted_price=float(row['Discounted Price']),
                    total_revenue=float(row['Total Revenue']),
                    gst_amount=float(row['GST_Amount']),
                    total_revenue_incl_gst=float(row['Total_Revenue_Incl_GST']),
                    customer_type=row['Customer Type'],
                    payment_method=row['Payment Method'],
                    rating=float(row['Rating']),
                    returns=bool(row['Returns'])
                )
                db.session.add(transaction)
            
            # Commit all changes
            db.session.commit()
            print("Successfully loaded inventory dataset into database")
            return True
            
    except Exception as e:
        print(f"Error loading inventory dataset: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return False

# Build the ML model
def build_model(df):
    try:
        print("\nBuilding model...")
        # Use only available columns
        features = ['Product Category', 'Brand Name', 'Rating', 'Current Stock']
        target = 'Total_Revenue_Incl_GST'

        print("Checking required columns...")
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Prepare features
        print("Preparing features...")
        X = df[features].copy()
        y = df[target]

        # Check for missing values
        print("\nMissing values in features:")
        print(X.isnull().sum())
        print("\nMissing values in target:")
        print(y.isnull().sum())

        # Convert categorical variables to numerical
        print("Converting categorical variables...")
        X = pd.get_dummies(X, columns=['Product Category', 'Brand Name'])

        # Split the data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")

        # Scale the features
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        print("Training model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel evaluation:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.2f}")

        return model, mae, r2
        
    except Exception as e:
        print(f"Error building model: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return None, None, None

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
    try:
        print("\nGetting recommendations...")
        if sales_df is None or inventory_df is None:
            print("Reloading data...")
            load_data()
        
        if category:
            print(f"Getting recommendations for category: {category}")
            category_products = sales_df[sales_df['Product Category'] == category].copy()
            category_products['score'] = category_products.apply(calculate_product_score, axis=1)
            category_products = category_products.sort_values('score', ascending=False)
            category_products = category_products.drop_duplicates(subset=['Product Name'], keep='first')
            top_products = category_products.head(n)
            
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
            print("Getting recommendations for all categories")
            recommendations = {}
            for cat in ['Mobiles', 'Laptops', 'Mobile Accessories', 'Laptop Accessories']:
                cat_products = sales_df[sales_df['Product Category'] == cat].copy()
                cat_products['score'] = cat_products.apply(calculate_product_score, axis=1)
                cat_products = cat_products.sort_values('score', ascending=False)
                cat_products = cat_products.drop_duplicates(subset=['Product Name'], keep='first')
                top_products = cat_products.head(n)
                
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
        
        print("Recommendations generated successfully")
        return recommendations
        
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return {} if category is None else []

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
    try:
        print("\nGetting top rated products...")
        # Filter out products with no ratings
        rated_products = df[df['rating'].notna()]
        print(f"Found {len(rated_products)} products with ratings")
        
        # Sort by rating in descending order and get top n
        top_products = rated_products.sort_values('rating', ascending=False).head(n)
        print(f"Selected top {len(top_products)} products")
        
        # Format the data for display
        return top_products[['product_name', 'brand', 'price', 'rating', 'reviews']].to_dict('records')
    except Exception as e:
        print(f"Error in get_top_rated_products: {str(e)}")
        return []

# Initialize
with app.app_context():
    try:
        print("Starting model initialization...")
        df = load_data()
        if df is None:
            raise ValueError("Failed to load data")
            
        print("\nData loaded successfully. Building model...")
        model, mae, r2 = build_model(df)
        
        if model is None:
            raise ValueError("Failed to build model")
            
        print(f"\nModel Initialized Successfully:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        # Store model and metrics in app context
        app.config['model'] = model
        app.config['model_mae'] = mae
        app.config['model_r2'] = r2
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        app.config['model'] = None
        app.config['model_mae'] = None
        app.config['model_r2'] = None

# Homepage route
@app.route('/')
def home():
    return render_template('home.html')

# ML model route
@app.route('/model')
@login_required
def model_page():
    try:
        # Check if model is available
        if app.config.get('model') is None:
            flash('Model is not initialized. Please try again later.', 'danger')
            return redirect(url_for('home'))
            
        print("\nLoading Amazon dataset...")
        amazon_df = pd.read_csv('Datasets/amazon_data_processed.csv')
        print(f"Amazon data loaded with shape: {amazon_df.shape}")
        
        print("\nGetting top rated products...")
        top_rated = get_top_rated_products(amazon_df)
        print(f"Found {len(top_rated)} top rated products")
        
        print("\nGetting recommendations...")
        recommendations = get_recommendations()
        print("Recommendations generated")
        
        print("\nGenerating BCG matrix...")
        bcg_recommendations = bcg_matrix(sales_df)
        print("BCG matrix generated")
        
        return render_template('model.html',
                             top_rated=top_rated,
                             recommendations=recommendations,
                             bcg_recommendations=bcg_recommendations)
                             
    except Exception as e:
        print(f"Error in model_page: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        flash('Error loading model data. Please try again later.', 'danger')
        return redirect(url_for('home'))

# Analytics page
@app.route('/analytics')
@login_required
def analytics():
    try:
        sales_data = []
        transactions = Transaction.query.all()
        for t in transactions:
            sales_data.append({
                'Date': t.date,
                'Product Category': t.product.category,
                'Brand Name': t.product.brand_name,
                'Total_Revenue_Incl_GST': t.total_revenue_incl_gst,
                'Transaction ID': t.transaction_id,
                'Customer Type': t.customer_type,
                'Unit Price': t.product.unit_price,
                'Stock_Status': t.product.inventory[0].stock_status if t.product.inventory else 'Unknown'
            })
        sales_df = pd.DataFrame(sales_data)
        print(f"Created DataFrame with shape: {sales_df.shape}")
        if len(sales_df) == 0:
            print("No sales data available")
            flash('No sales data available. Please add some transactions first.', 'warning')
            return redirect(url_for('inventory'))
        min_date = sales_df["Date"].min().date()
        max_date = sales_df["Date"].max().date()
        categories = ["All"] + sorted(sales_df["Product Category"].dropna().unique().tolist())
        print(f"Date range: {min_date} to {max_date}")
        print(f"Categories: {categories}")
        return render_template("analytics.html", min_date=min_date, max_date=max_date, categories=categories)
    except Exception as e:
        print(f"Error in analytics route: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        flash('Error loading analytics data. Please try again.', 'error')
        return redirect(url_for('inventory'))

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

        # Get transactions from database
        query = Transaction.query.filter(
            Transaction.date >= start_date,
            Transaction.date <= end_date
        )
        
        if category != 'all':
            query = query.join(Product).filter(Product.category == category)
            
        transactions = query.all()
        
        # Convert to DataFrame
        sales_data = []
        for t in transactions:
            sales_data.append({
                'Date': t.date,
                'Product Category': t.product.category,
                'Brand Name': t.product.brand_name,
                'Total_Revenue_Incl_GST': t.total_revenue_incl_gst,
                'Transaction ID': t.transaction_id,
                'Customer Type': t.customer_type,
                'Unit Price': t.product.unit_price,
                'Stock_Status': t.product.inventory[0].stock_status if t.product.inventory else 'Unknown'
            })
        
        filtered_sales = pd.DataFrame(sales_data)

        # Calculate comparison data if requested
        comparison_sales = None
        if compare_period != 'none':
            if compare_period == 'previous':
                comp_start = start_date - pd.Timedelta(days=date_range)
                comp_end = start_date
            else:  # year
                comp_start = start_date - pd.Timedelta(days=365)
                comp_end = end_date - pd.Timedelta(days=365)
            
            comp_query = Transaction.query.filter(
                Transaction.date >= comp_start,
                Transaction.date <= comp_end
            )
            
            if category != 'all':
                comp_query = comp_query.join(Product).filter(Product.category == category)
                
            comp_transactions = comp_query.all()
            
            comp_data = []
            for t in comp_transactions:
                comp_data.append({
                    'Date': t.date,
                    'Product Category': t.product.category,
                    'Brand Name': t.product.brand_name,
                    'Total_Revenue_Incl_GST': t.total_revenue_incl_gst,
                    'Transaction ID': t.transaction_id,
                    'Customer Type': t.customer_type,
                    'Unit Price': t.product.unit_price,
                    'Stock_Status': t.product.inventory[0].stock_status if t.product.inventory else 'Unknown'
                })
            
            comparison_sales = pd.DataFrame(comp_data)

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
            user.last_login = datetime.utcnow()
            db.session.commit()
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
    
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

@app.route('/model_status')
@login_required
def model_status():
    """Check the status of the model and data loading"""
    try:
        status = {
            'model_initialized': app.config.get('model') is not None,
            'data_loaded': sales_df is not None and inventory_df is not None,
            'last_update': last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else None,
            'model_metrics': {
                'mae': app.config.get('model_mae'),
                'r2': app.config.get('model_r2')
            }
        }
        
        if sales_df is not None:
            status['sales_data_shape'] = sales_df.shape
            status['sales_data_columns'] = sales_df.columns.tolist()
            
        if inventory_df is not None:
            status['inventory_data_shape'] = inventory_df.shape
            status['inventory_data_columns'] = inventory_df.columns.tolist()
            
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload_model')
@login_required
def reload_model():
    """Reload the model and data"""
    try:
        global sales_df, inventory_df, last_update
        
        # Reload data
        df = load_data()
        if df is None:
            raise ValueError("Failed to load data")
            
        # Rebuild model
        model, mae, r2 = build_model(df)
        if model is None:
            raise ValueError("Failed to build model")
            
        # Update app config
        app.config['model'] = model
        app.config['model_mae'] = mae
        app.config['model_r2'] = r2
        
        flash('Model reloaded successfully!', 'success')
        return redirect(url_for('model_page'))
        
    except Exception as e:
        flash(f'Error reloading model: {str(e)}', 'danger')
        return redirect(url_for('home'))

@app.route('/load_inventory_dataset')
@login_required
def load_dataset():
    if not session.get('user_id'):
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = User.query.get(session['user_id'])
    if not user or not user.is_admin:
        return jsonify({'error': 'Not authorized'}), 403
    
    success = load_inventory_dataset()
    if success:
        flash('Inventory dataset loaded successfully', 'success')
    else:
        flash('Error loading inventory dataset', 'error')
    
    return redirect(url_for('inventory'))

@app.route('/inventory')
@login_required
def inventory():
    return render_template('inventory.html')

@app.route('/api/products', methods=['GET'])
@login_required
def get_products():
    products = Product.query.all()
    return jsonify([{
        'id': p.id,
        'name': p.product_name,
        'brand': p.brand_name,
        'category': p.category,
        'subcategory': p.subcategory,
        'unit_price': p.unit_price,
        'gst_percentage': p.gst_percentage,
        'current_stock': p.inventory[0].current_stock if p.inventory else 0,
        'stock_status': p.inventory[0].stock_status if p.inventory else 'out'
    } for p in products])

@app.route('/api/products', methods=['POST'])
@login_required
def add_product():
    data = request.json
    
    # Check if product already exists
    existing_product = Product.query.filter_by(
        product_name=data['product_name'],
        brand_name=data['brand_name']
    ).first()
    
    if existing_product:
        # Update existing product
        existing_product.unit_price = float(data['unit_price'])
        existing_product.cost_price = float(data['cost_price'])
        existing_product.gst_percentage = float(data['gst_percentage'])
        
        # Update inventory
        inventory = existing_product.inventory[0]
        inventory.current_stock += int(data['current_stock'])
        inventory.reorder_level = int(data['reorder_level'])
        inventory.reorder_quantity = int(data['reorder_quantity'])
        inventory.supplier_name = data['supplier_name']
        inventory.lead_time = int(data['lead_time'])
        inventory.last_restocked = datetime.utcnow()
        
        # Update stock status
        if inventory.current_stock <= 0:
            inventory.stock_status = 'out'
        elif inventory.current_stock <= inventory.reorder_level:
            inventory.stock_status = 'low'
        else:
            inventory.stock_status = 'ok'
    else:
        # Create new product
        product = Product(
            product_name=data['product_name'],
            brand_name=data['brand_name'],
            category=data['category'],
            subcategory=data['subcategory'],
            unit_price=float(data['unit_price']),
            cost_price=float(data['cost_price']),
            gst_percentage=float(data['gst_percentage'])
        )
        db.session.add(product)
        db.session.flush()  # Get the product ID
        
        # Create inventory record
        inventory = Inventory(
            product_id=product.id,
            current_stock=int(data['current_stock']),
            reorder_level=int(data['reorder_level']),
            reorder_quantity=int(data['reorder_quantity']),
            supplier_name=data['supplier_name'],
            lead_time=int(data['lead_time']),
            last_restocked=datetime.utcnow()
        )
        
        # Set initial stock status
        if inventory.current_stock <= 0:
            inventory.stock_status = 'out'
        elif inventory.current_stock <= inventory.reorder_level:
            inventory.stock_status = 'low'
        else:
            inventory.stock_status = 'ok'
        
        db.session.add(inventory)
    
    db.session.commit()
    return jsonify({'message': 'Product saved successfully'})

@app.route('/api/sales', methods=['POST'])
@login_required
def create_sale():
    data = request.json
    transaction_id = f"TRX{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    try:
        # Create transaction records for each product
        for item in data['items']:
            product = Product.query.get(item['product_id'])
            if not product:
                return jsonify({'error': f'Product {item["product_id"]} not found'}), 404
            
            inventory = product.inventory[0]
            if inventory.current_stock < item['quantity']:
                return jsonify({'error': f'Insufficient stock for {product.product_name}'}), 400
            
            # Calculate prices
            unit_price = product.unit_price
            quantity = item['quantity']
            subtotal = unit_price * quantity
            gst_amount = subtotal * (product.gst_percentage / 100)
            discount = subtotal * (item.get('discount', 0) / 100)
            total = subtotal + gst_amount - discount
            
            # Create transaction record
            transaction = Transaction(
                transaction_id=transaction_id,
                product_id=product.id,
                date=datetime.utcnow(),
                units_sold=quantity,
                discount_percentage=item.get('discount', 0),
                discounted_price=unit_price * (1 - item.get('discount', 0) / 100),
                total_revenue=subtotal,
                gst_amount=gst_amount,
                total_revenue_incl_gst=total,
                customer_type=data.get('customer_type', 'regular'),
                payment_method=data['payment_method']
            )
            db.session.add(transaction)
            
            # Update inventory
            inventory.current_stock -= quantity
            if inventory.current_stock <= 0:
                inventory.stock_status = 'out'
            elif inventory.current_stock <= inventory.reorder_level:
                inventory.stock_status = 'low'
            else:
                inventory.stock_status = 'ok'
        
        db.session.commit()
        return jsonify({
            'message': 'Sale completed successfully',
            'transaction_id': transaction_id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/products/search', methods=['GET'])
@login_required
def search_products():
    query = request.args.get('q', '')
    category = request.args.get('category', '')
    stock_status = request.args.get('stock_status', '')
    
    products_query = Product.query
    
    if query:
        products_query = products_query.filter(
            (Product.product_name.ilike(f'%{query}%')) |
            (Product.brand_name.ilike(f'%{query}%'))
        )
    
    if category:
        products_query = products_query.filter(Product.category == category)
    
    products = products_query.all()
    
    # Filter by stock status if needed
    if stock_status:
        products = [p for p in products if p.inventory and p.inventory[0].stock_status == stock_status]
    
    return jsonify([{
        'id': p.id,
        'name': p.product_name,
        'brand': p.brand_name,
        'category': p.category,
        'subcategory': p.subcategory,
        'unit_price': p.unit_price,
        'gst_percentage': p.gst_percentage,
        'current_stock': p.inventory[0].current_stock if p.inventory else 0,
        'stock_status': p.inventory[0].stock_status if p.inventory else 'out',
        'reorder_level': p.inventory[0].reorder_level if p.inventory else 0
    } for p in products])

@app.route('/api/sold_products', methods=['GET'])
@login_required
def api_sold_products():
    transactions = Transaction.query.order_by(Transaction.date.desc()).all()
    sold_products = []
    for t in transactions:
        sold_products.append({
            'date': t.date.isoformat() if t.date else '',
            'product_name': t.product.product_name if t.product else '',
            'brand': t.product.brand_name if t.product else '',
            'category': t.product.category if t.product else '',
            'units_sold': t.units_sold,
            'customer_type': t.customer_type,
            'payment_method': t.payment_method,
            'total_revenue_incl_gst': t.total_revenue_incl_gst
        })
    return jsonify(sold_products)

if __name__ == '__main__':
    load_data()
    app.run(debug=True)
