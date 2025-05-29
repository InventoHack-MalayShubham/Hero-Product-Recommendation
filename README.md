# Product Recommendation System

A comprehensive product recommendation and analytics system that helps shopkeepers make data-driven decisions about their inventory and sales strategies.

## Features

### 1. Product Recommendations
- **Top Rated Products**: Displays the highest-rated products from Amazon dataset
- **Category-wise Recommendations**: Smart recommendations based on product categories
- **BCG Matrix Analysis**: Classifies products into Stars, Cash Cows, Question Marks, and Dogs
- **Score-based Ranking**: Products are scored based on multiple factors including:
  - Rating
  - Revenue
  - Stock status
  - Customer reviews

### 2. Analytics Dashboard
- **Interactive Charts**: Visual representation of sales data
- **Key Performance Indicators (KPIs)**:
  - Total Revenue
  - Average Order Value
  - Most Popular Product
- **Data Filtering**:
  - Date range selection
  - Category filtering
  - Period comparison
- **Data Export**: Export data in multiple formats (CSV, Excel)

### 3. User Authentication
- Secure login and signup system
- Protected routes for sensitive data
- User session management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Hero-Product-Recommendation.git
cd Hero-Product-Recommendation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Project Structure

```
Hero-Product-Recommendation/
├── app.py                  # Main application file
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
├── Datasets/              # Data files
│   ├── amazon_data_processed.csv
│   ├── rohit_electronics_sales_1000.csv
│   └── inventory_data_new1.csv
├── static/                # Static files
│   └── css/
│       └── styles.css
└── templates/             # HTML templates
    ├── base.html
    ├── home.html
    ├── model.html
    ├── analytics.html
    ├── login.html
    └── signup.html
```

## Usage

1. **Home Page**
   - Overview of the system
   - Quick access to all features
   - System statistics

2. **Model Page**
   - View top-rated products
   - Get category-wise recommendations
   - Analyze products using BCG Matrix
   - Access detailed product information

3. **Analytics Dashboard**
   - Monitor sales performance
   - Track key metrics
   - Generate reports
   - Export data

## Data Sources

1. **Amazon Dataset**
   - Product information
   - Ratings and reviews
   - Price data

2. **Sales Data**
   - Transaction history
   - Customer information
   - Revenue data

3. **Inventory Data**
   - Stock levels
   - Product categories
   - Brand information

## Security Features

- Password hashing
- Session management
- Protected routes
- Secure data handling

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask framework
- Pandas for data analysis
- Chart.js for visualizations
- SQLAlchemy for database management 