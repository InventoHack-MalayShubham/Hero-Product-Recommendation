# DukanBuddy - AI Product Recommendation System

## Project Overview
DukanBuddy is an intelligent product recommendation system designed to help businesses optimize their inventory management and boost sales through data-driven insights. The system combines machine learning algorithms with real-time analytics to provide personalized product recommendations and inventory management solutions.

## Key Features

### 1. Product Recommendations
- **Smart Recommendations**: Uses machine learning to suggest products based on multiple factors:
  - Product ratings and reviews
  - Historical sales data
  - Current inventory levels
  - Brand reputation
  - Category performance
- **BCG Matrix Analysis**: Categorizes products into:
  - Stars (High growth, High market share)
  - Cash Cows (Low growth, High market share)
  - Question Marks (High growth, Low market share)
  - Dogs (Low growth, Low market share)

### 2. Analytics Dashboard
- **Real-time Metrics**:
  - Total Revenue
  - Average Order Value
  - Most Popular Products
- **Interactive Charts**:
  - Revenue by Category
  - Daily Revenue Trends
  - Top Performing Brands
  - Stock Status Distribution
- **Filtering Capabilities**:
  - Date range selection
  - Category filters
  - Comparison periods

### 3. Inventory Management
- **Real-time Stock Monitoring**:
  - Current stock levels
  - Reorder levels
  - Stock status (In Stock/Low/Out of Stock)
- **Product Details**:
  - Unit prices
  - Cost prices
  - Selling prices
  - Discount percentages
  - GST rates
- **Supplier Information**:
  - Supplier details
  - Lead times
  - Reorder quantities

## Technical Architecture

### Frontend
- **Template Engine**: Jinja2
- **Styling**: 
  - Custom CSS with CSS Variables
  - Bootstrap 5
  - Font Awesome Icons
- **JavaScript**:
  - Interactive charts
  - Real-time search functionality
  - Responsive design

### Backend
- **Framework**: Flask
- **Database**: SQLite
- **Machine Learning**:
  - scikit-learn
  - Random Forest Regressor
  - Feature Engineering Pipeline

### Security Features
- User authentication
- Password hashing
- Session management
- Login required decorators

## Data Structure
The system uses multiple datasets:

### 1. Sales Data (`rohit_electronics_sales_1000.csv`)
- Transaction details
- Product information
- Customer data
- Revenue metrics

### 2. Inventory Data (`inventory_data_new1.csv`)
- Product details
- Stock levels
- Pricing information
- Supplier details

### 3. Amazon Dataset (`amazon_data_processed.csv`)
- Product ratings
- Reviews
- Additional product metadata

## Installation and Setup

### Prerequisites
```bash
Python 3.8+
pip
virtualenv
```

### Installation Steps
1. Clone the repository:
```bash
git clone [repository-url]
cd DukanBuddy
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Unix
venv\Scripts\activate     # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

5. Run the application:
```bash
python app.py
```

## Project Structure
```
DukanBuddy/
├── app.py                 # Main application file
├── templates/            # HTML templates
│   ├── base.html        # Base template
│   ├── home.html        # Homepage
│   ├── model.html       # Recommendation page
│   ├── analytics.html   # Analytics dashboard
│   └── inventory.html   # Inventory management
├── static/              # Static files
│   ├── images/         # Images and icons
│   └── css/            # CSS files
├── Datasets/           # Data files
│   ├── rohit_electronics_sales_1000.csv
│   ├── inventory_data_new1.csv
│   └── amazon_data_processed.csv
└── requirements.txt    # Project dependencies
```

## Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Features**:
  - Product Category
  - Brand Name
  - Rating
  - Current Stock
- **Target Variable**: Total Revenue
- **Performance Metrics**:
  - Mean Absolute Error (MAE)
  - R² Score

## API Routes
- `/` - Homepage
- `/recommendation` - Product recommendations
- `/analytics` - Analytics dashboard
- `/inventory` - Inventory management
- `/get_data` - API endpoint for analytics data
- `/model_status` - Model status and metrics
- `/reload_model` - Reload ML model

## Security and Authentication
- User registration and login system
- Password encryption using Werkzeug
- Session-based authentication
- Admin privileges for specific functions

## Future Enhancements
1. Advanced recommendation algorithms
2. Real-time price optimization
3. Automated inventory forecasting
4. Mobile application
5. Integration with e-commerce platforms
6. Enhanced data visualization
7. Export functionality for reports

## Contributors
- [List of contributors]

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Flask framework
- scikit-learn library
- Bootstrap framework
- Font Awesome icons 