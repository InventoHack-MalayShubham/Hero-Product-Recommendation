from flask import Flask, render_template, request, jsonify
import pandas as pd
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

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
model = None
df = None

# Load dataset and enrich it
def load_data():
    df = pd.read_csv('rohit_electronics_sales_1000.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Revenue_Per_Unit'] = df['Total_Revenue_Incl_GST'] / df['Current Stock'].replace(0, 1)
    df['Popularity'] = df['Rating'] * df['Total_Revenue_Incl_GST']
    df['Stock_Status'] = df['Current Stock'].apply(lambda x: 'Low' if x <= 5 else 'OK')
    return df

# Build the ML model
def build_model(df):
    features = ['Product Category', 'Subcategory', 'Brand Name',
                'Unit Price', 'Discount (%)', 'Current Stock',
                'Month', 'DayOfWeek', 'Rating']
    target = 'Revenue_Per_Unit'

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Unit Price', 'Discount (%)', 'Current Stock', 'Month', 'DayOfWeek', 'Rating']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Product Category', 'Subcategory', 'Brand Name'])
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
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
    return render_template('model.html', mae=round(mae, 2), r2=round(r2, 2))

# Analytics page
@app.route('/analytics')
def analytics():
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    categories = ["All"] + sorted(df["Product Category"].dropna().unique().tolist())
    return render_template("analytics.html", min_date=min_date, max_date=max_date, categories=categories)

# Fetch filtered data
@app.route("/get_data", methods=["POST"])
def get_data():
    start_date = pd.to_datetime(request.json["start"])
    end_date = pd.to_datetime(request.json["end"])
    category = request.json["category"]

    filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

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
    low_stock_items = filtered_df[filtered_df["Current Stock"] < 10]
    stock_status = low_stock_items["Stock_Status"].value_counts().to_dict()

    return jsonify({
        "dailyRevenue": daily_revenue,
        "revenueByCategory": revenue_by_category,
        "topBrands": top_brands,
        "customerType": customer_type,
        "paymentMethods": payment_methods,
        "stockStatus": stock_status
    })

if __name__ == '__main__':
    app.run(debug=True)
