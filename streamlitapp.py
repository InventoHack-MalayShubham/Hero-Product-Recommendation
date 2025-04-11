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
    page_title="Inventory Dashboard",
    page_icon="📦",
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

# Initialize session state
if "product_data" not in st.session_state:
    st.session_state.product_data = pd.DataFrame(columns=[
        "Product ID", "Name", "Category", "Stock", "Expiry Date", "GST %", "Image URL"
    ])

# GST category mapping
GST_CATEGORIES = {
    "Food": 5,
    "Clothing": 12,
    "Electronics": 18,
    "Books": 0,
    "Luxury": 28,
    "Other": 18
}

# Function to validate image URL
def is_valid_image_url(url):
    if not url:
        return False
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# Function to get image from URL
def get_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        return None
    except:
        return None

# Function to check if product is expiring soon
def is_expiring_soon(expiry_date, days=7):
    if pd.isna(expiry_date):
        return False
    today = datetime.now().date()
    expiry = pd.to_datetime(expiry_date).date()
    days_until_expiry = (expiry - today).days
    return 0 <= days_until_expiry <= days

# Function to check if stock is low
def is_low_stock(stock, threshold=30):
    return stock < threshold

# Main app
st.title("📦 Inventory Dashboard")

# Sidebar for mode selection and CSV import
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Choose Mode", ["Manual Entry", "Bill Verification"], horizontal=True)
    
    st.header("Data Import")
    uploaded_file = st.file_uploader("📁 Import CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Check if required columns exist, if not add them
        for col in ["GST %", "Image URL"]:
            if col not in df.columns:
                df[col] = None
        st.session_state.product_data = pd.concat([st.session_state.product_data, df], ignore_index=True)
        st.success("CSV data imported!")

# Manual Entry
if mode == "Manual Entry":
    with st.expander("➕ Add Product", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Product Name")
            category = st.selectbox("Category", options=list(GST_CATEGORIES.keys()))
            gst_percentage = GST_CATEGORIES[category]
            st.info(f"GST: {gst_percentage}%")
        with col2:
            stock = st.number_input("Stock", min_value=0, step=1)
            expiry = st.date_input("Expiry Date", value=datetime.now().date() + timedelta(days=30))
        with col3:
            image_url = st.text_input("Image URL (optional)")
            if image_url and not is_valid_image_url(image_url):
                st.error("Invalid image URL")
            
            if st.button("Add Product"):
                new_row = {
                    "Product ID": len(st.session_state.product_data) + 1,
                    "Name": name,
                    "Category": category,
                    "Stock": stock,
                    "Expiry Date": expiry,
                    "GST %": gst_percentage,
                    "Image URL": image_url if is_valid_image_url(image_url) else None
                }
                st.session_state.product_data = pd.concat([st.session_state.product_data, pd.DataFrame([new_row])], ignore_index=True)
                st.success(f"Product '{name}' added.")

# Search and filters
st.subheader("🔍 Search and Filter")
col1, col2 = st.columns(2)
with col1:
    search_term = st.text_input("Search by name or category")
with col2:
    filter_expiry = st.checkbox("Show only expiring soon (within 7 days)")
    filter_low_stock = st.checkbox("Show only low stock (< 30 units)")

# Apply filters
filtered_data = st.session_state.product_data.copy()

if search_term:
    filtered_data = filtered_data[
        filtered_data['Name'].str.contains(search_term, case=False, na=False) |
        filtered_data['Category'].str.contains(search_term, case=False, na=False)
    ]

if filter_expiry:
    filtered_data = filtered_data[filtered_data['Expiry Date'].apply(is_expiring_soon)]

if filter_low_stock:
    filtered_data = filtered_data[filtered_data['Stock'].apply(is_low_stock)]

# Product Grid Display
st.subheader("🛒 Products")
if len(filtered_data) == 0:
    st.info("No products found matching your criteria.")
else:
    for idx, row in filtered_data.iterrows():
        with st.container():
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            
            # Check for expiry and low stock
            expiry_warning = is_expiring_soon(row['Expiry Date'])
            low_stock_warning = is_low_stock(row['Stock'])
            
            col1, col2, col3 = st.columns([2, 4, 2])
            
            # Product image if available
            with col1:
                if pd.notna(row['Image URL']):
                    image = get_image_from_url(row['Image URL'])
                    if image:
                        st.image(image, caption=row['Name'], use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/150?text=No+Image", caption=row['Name'], use_column_width=True)
                else:
                    st.image("https://via.placeholder.com/150?text=No+Image", caption=row['Name'], use_column_width=True)
            
            # Product details
            with col2:
                st.markdown(f"**🆔 ID**: {row['Product ID']}")
                st.markdown(f"**Name**: {row['Name']}")
                st.markdown(f"**Category**: {row['Category']}")
                
                # Stock with warning if low
                stock_text = f"**Stock**: {row['Stock']}"
                if low_stock_warning:
                    stock_text += '<span class="badge badge-danger">Low Stock</span>'
                st.markdown(stock_text, unsafe_allow_html=True)
                
                # Expiry with warning if soon
                expiry_text = f"**Expiry**: {row['Expiry Date']}"
                if expiry_warning:
                    expiry_text += '<span class="badge badge-warning">⚠️ Expiring Soon</span>'
                st.markdown(expiry_text, unsafe_allow_html=True)
                
                # GST percentage
                st.markdown(f"**GST**: {row['GST %']}%")
            
            # Stock update
            with col3:
                with st.expander("🛠 Update Stock"):
                    change = st.number_input("Change Stock", min_value=-row["Stock"], step=1, key=f"chg_{idx}")
                    if st.button("Update", key=f"btn_{idx}"):
                        st.session_state.product_data.at[idx, "Stock"] += change
                        st.success(f"Stock updated for {row['Name']}")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Export options
st.subheader("📤 Export Data")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Export Full Inventory CSV", 
        st.session_state.product_data.to_csv(index=False), 
        file_name="full_inventory.csv",
        mime="text/csv"
    )
with col2:
    st.download_button(
        "Export Filtered View CSV", 
        filtered_data.to_csv(index=False), 
        file_name="filtered_inventory.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("📊 Inventory Dashboard v2.0 | Built with Streamlit")