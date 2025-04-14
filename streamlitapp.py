import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import base64
from PIL import Image
import requests
from urllib.parse import urlparse

# Set page configuration
st.set_page_config(
    page_title="Rohit Electronics Inventory",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .expiry-warning {
        color: #ff4b4b;
        font-weight: bold;
    }
    .low-stock {
        color: #ff4b4b;
        font-weight: bold;
    }
    .product-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .product-image {
        max-width: 100%;
        height: auto;
        border-radius: 5px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-left: 5px;
    }
    .badge-warning {
        background-color: #ffcc00;
        color: #000;
    }
    .badge-danger {
        background-color: #ff4b4b;
        color: #fff;
    }
    .badge-info {
        background-color: #00ccff;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with specific dataset structure
if "product_data" not in st.session_state:
    try:
        # Load the specific dataset
        df = pd.read_csv("Datasets\\rohit_electronics_sales_data.csv")
        
        # Ensure all required columns are present
        required_columns = [
            "Transaction ID", "Date", "Product Category", "Subcategory", "Product Name",
            "Brand Name", "Units Sold", "Unit Price", "Discount", "Discounted Price",
            "Total Revenue", "GST_Percentage", "GST_Amount", "Total_Revenue_Incl_GST",
            "Current Stock", "Stock_Status", "Rating", "Returns", "Customer Type",
            "Region", "Payment Method"
        ]
        
        # Validate data types
        df["Date"] = pd.to_datetime(df["Date"])
        df["Units Sold"] = pd.to_numeric(df["Units Sold"], errors='coerce')
        df["Unit Price"] = pd.to_numeric(df["Unit Price"], errors='coerce')
        df["Discount"] = pd.to_numeric(df["Discount"], errors='coerce')
        df["Current Stock"] = pd.to_numeric(df["Current Stock"], errors='coerce')
        df["Rating"] = pd.to_numeric(df["Rating"], errors='coerce')
        
        st.session_state.product_data = df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.session_state.product_data = pd.DataFrame(columns=required_columns)

# Function to save data
def save_data():
    st.session_state.product_data.to_csv("inventory_data.csv", index=False)

# Function to validate image URL
def is_valid_image_url(url):
    if not url:
        return False
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# Function to get image from URL with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_image_from_url(url):
    if not url or not is_valid_image_url(url):
        return None
    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes
        return Image.open(io.BytesIO(response.content))
    except (requests.RequestException, IOError, Image.UnidentifiedImageError) as e:
        st.warning(f"Failed to load image from URL: {url}")
        return None

# Function to check if stock is low
def is_low_stock(stock, threshold=30):
    return stock < threshold

# Function to validate product data
def validate_product_data(data):
    errors = []
    
    # Required fields based on dataset
    required_fields = [
        "Product Name", "Product Category", "Subcategory", "Brand Name",
        "Units Sold", "Unit Price", "Current Stock"
    ]
    
    for field in required_fields:
        if not data.get(field):
            errors.append(f"{field} is required")
    
    # Numeric validation
    try:
        units = int(data.get("Units Sold", 0))
        if units < 0:
            errors.append("Units Sold cannot be negative")
    except ValueError:
        errors.append("Units Sold must be a valid number")
    
    try:
        price = float(data.get("Unit Price", 0))
        if price < 0:
            errors.append("Unit Price cannot be negative")
    except ValueError:
        errors.append("Unit Price must be a valid number")
    
    try:
        stock = int(data.get("Current Stock", 0))
        if stock < 0:
            errors.append("Current Stock cannot be negative")
    except ValueError:
        errors.append("Current Stock must be a valid number")
    
    # Category validation
    valid_categories = ['Mobiles', 'Laptop Accessories', 'Mobile Accessories', 'Laptops']
    if data.get("Product Category") not in valid_categories:
        errors.append(f"Product Category must be one of: {', '.join(valid_categories)}")
    
    return errors

# Function to validate CSV data
def validate_csv_data(df):
    required_columns = ["Product Name", "Product Category", "Current Stock", "Unit Price"]
    errors = []
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check data types
    try:
        df["Current Stock"] = pd.to_numeric(df["Current Stock"], errors='raise')
        if (df["Current Stock"] < 0).any():
            errors.append("Stock values cannot be negative")
    except ValueError:
        errors.append("Stock values must be numeric")
    
    try:
        df["Unit Price"] = pd.to_numeric(df["Unit Price"], errors='raise')
        if (df["Unit Price"] < 0).any():
            errors.append("Price values cannot be negative")
    except ValueError:
        errors.append("Price values must be numeric")
    
    return errors

# Function to perform search with specific categories
def perform_search(df, search_term, category_filter=None, subcategory_filter=None, low_stock_filter=False):
    if not search_term and not category_filter and not subcategory_filter and not low_stock_filter:
        return df
    
    mask = pd.Series(True, index=df.index)
    
    if search_term:
        search_mask = (
            df['Product Name'].str.contains(search_term, case=False, na=False) |
            df['Brand Name'].str.contains(search_term, case=False, na=False) |
            df['Product Category'].str.contains(search_term, case=False, na=False) |
            df['Subcategory'].str.contains(search_term, case=False, na=False)
        )
        mask &= search_mask
    
    if category_filter and category_filter != "All":
        mask &= df['Product Category'] == category_filter
    
    if subcategory_filter and subcategory_filter != "All":
        mask &= df['Subcategory'] == subcategory_filter
    
    if low_stock_filter:
        mask &= df['Current Stock'] < 30
    
    return df[mask]

# Main app
df = st.session_state.product_data
st.title("📱 Rohit Electronics Inventory Dashboard")

# Sidebar - Upload CSV
with st.sidebar:
    st.header("Import Inventory CSV")
    uploaded_file = st.file_uploader("📁 Import CSV", type=["csv"])
    if uploaded_file:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            errors = validate_csv_data(df_uploaded)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                st.session_state.product_data = pd.concat([df, df_uploaded], ignore_index=True)
                save_data()  # Save the updated data
                st.success("CSV data imported successfully!")
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

# Filters
st.subheader("🔍 Filter Options")
col1, col2, col3, col4 = st.columns(4)
with col1:
    search_term = st.text_input("Search Products")
with col2:
    category_filter = st.selectbox(
        "Filter by Category",
        options=["All"] + sorted(df['Product Category'].unique().tolist())
    )
with col3:
    # Only show subcategories for selected category
    if category_filter != "All":
        subcategories = ["All"] + sorted(df[df['Product Category'] == category_filter]['Subcategory'].unique().tolist())
    else:
        subcategories = ["All"] + sorted(df['Subcategory'].unique().tolist())
    subcategory_filter = st.selectbox("Filter by Subcategory", options=subcategories)
with col4:
    low_stock_filter = st.checkbox("Show only low stock (<30)")

# Apply filters
filtered_data = perform_search(
    df,
    search_term,
    None if category_filter == "All" else category_filter,
    None if subcategory_filter == "All" else subcategory_filter,
    low_stock_filter
)

# Product Display
st.subheader("📦 Product Inventory")
if len(filtered_data) == 0:
    st.info("No matching products found.")
else:
    for idx, row in filtered_data.iterrows():
        with st.container():
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 3])

            with col1:
                # Display product image if available
                if pd.notna(row.get('Image URL')) and is_valid_image_url(row['Image URL']):
                    img = get_image_from_url(row['Image URL'])
                    if img:
                        st.image(img, use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/150", use_column_width=True)
                else:
                    st.image("https://via.placeholder.com/150", use_column_width=True)

            with col2:
                # Product details
                st.markdown(f"**Product Name:** {row['Product Name']}")
                st.markdown(f"**Brand:** {row['Brand Name']} | **Category:** {row['Product Category']}")
                st.markdown(f"**Subcategory:** {row['Subcategory']}")
                
                # Stock information with warning badges
                stock_status = f"**Stock:** {row['Current Stock']}"
                if row['Current Stock'] < 30:
                    stock_status += ' <span class="badge badge-danger">Low Stock</span>'
                elif row['Current Stock'] == 0:
                    stock_status += ' <span class="badge badge-danger">Out of Stock</span>'
                st.markdown(stock_status, unsafe_allow_html=True)
                
                # Sales information
                st.markdown(f"**Units Sold:** {row['Units Sold']}")
                st.markdown(f"**Price:** ₹{row['Unit Price']:,.2f}")
                if row['Discount'] > 0:
                    st.markdown(f"**Discount:** {row['Discount']}% | **Discounted Price:** ₹{row['Discounted Price']:,.2f}")
                
                # Financial information
                st.markdown(f"**GST:** {row['GST_Percentage']}% | **GST Amount:** ₹{row['GST_Amount']:,.2f}")
                st.markdown(f"**Total Revenue (incl. GST):** ₹{row['Total_Revenue_Incl_GST']:,.2f}")
                
                # Additional information
                st.markdown(f"**Rating:** {row['Rating']} ⭐")
                st.markdown(f"**Customer Type:** {row['Customer Type']}")
                st.markdown(f"**Payment Method:** {row['Payment Method']}")
                st.markdown(f"**Region:** {row['Region']}")

            st.markdown('</div>', unsafe_allow_html=True)

# Export Buttons
st.subheader("📤 Export Inventory")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Download Full CSV",
        df.to_csv(index=False),
        file_name="rohit_inventory_full.csv",
        mime="text/csv"
    )
with col2:
    st.download_button(
        "Download Filtered CSV",
        filtered_data.to_csv(index=False),
        file_name="rohit_inventory_filtered.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("🛍️ Rohit Electronics | Vending UI Inventory Dashboard")

# Update the manual entry form section
with st.form("product_form"):
    # ... existing form fields ...
    
    if st.form_submit_button("Add Product"):
        new_product = {
            "Product Name": product_name,
            "Product Category": category,
            "Current Stock": stock,
            "Unit Price": price,
            "Date": date,
            "Image URL": image_url
            # ... other fields ...
        }
        
        errors = validate_product_data(new_product)
        if errors:
            for error in errors:
                st.error(error)
        else:
            # Add the product to the DataFrame
            st.session_state.product_data = pd.concat([
                st.session_state.product_data,
                pd.DataFrame([new_product])
            ], ignore_index=True)
            save_data()  # Save the updated data
            st.success("Product added successfully!")

# Add summary statistics section
st.subheader("📊 Summary Statistics")

# Create columns for different metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Products", len(df))
    st.metric("Total Revenue", f"₹{df['Total_Revenue_Incl_GST'].sum():,.2f}")

with col2:
    st.metric("Average Rating", f"{df['Rating'].mean():.1f} ⭐")
    st.metric("Total Units Sold", f"{df['Units Sold'].sum():,}")

with col3:
    low_stock = len(df[df['Current Stock'] < 30])
    st.metric("Low Stock Items", low_stock)
    st.metric("Out of Stock Items", len(df[df['Current Stock'] == 0]))

with col4:
    st.metric("Average Discount", f"{df['Discount'].mean():.1f}%")
    st.metric("Total GST Collected", f"₹{df['GST_Amount'].sum():,.2f}")

# Add visualizations
st.subheader("📈 Product Analysis")

# Category distribution
st.write("### Product Categories")
category_counts = df['Product Category'].value_counts()
st.bar_chart(category_counts)

# Stock status distribution
st.write("### Stock Status")
stock_status = df['Stock_Status'].value_counts()
st.bar_chart(stock_status)

# Price distribution
st.write("### Price Distribution")
st.line_chart(df['Unit Price'].value_counts().sort_index())

# Rating distribution
st.write("### Rating Distribution")
rating_counts = df['Rating'].value_counts().sort_index()
st.bar_chart(rating_counts)

# Payment method distribution
st.write("### Payment Methods")
payment_counts = df['Payment Method'].value_counts()
st.pie_chart(payment_counts)