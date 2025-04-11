# Hero Product Recommendation

This repository contains two inventory management applications:

1. A Flask-based web application (`app.py`)
2. A Streamlit-based dashboard (`streamlitapp.py`)

## Streamlit Dashboard

The Streamlit dashboard provides a modern, responsive interface for inventory management with the following features:

### Features

- **Manual Entry Mode**: Add products with details including name, category, stock, expiry date, and optional image URL
- **Bill Verification Mode**: Verify bills against inventory
- **CSV Import**: Import product data from CSV files
- **Search and Filter**: Search by product name or category, filter by expiry date and stock levels
- **Expiry Alerts**: Products expiring within 7 days are highlighted with a warning badge
- **Low Stock Alerts**: Products with stock below 30 units are highlighted with a warning badge
- **GST Auto-Suggestion**: GST percentage is automatically suggested based on product category
- **Product Images**: Optional image URLs for products
- **Export Options**: Export full inventory or filtered view as CSV

### Setup and Running

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run streamlitapp.py
   ```

3. Access the dashboard in your web browser at http://localhost:8501

## Flask Web Application

The Flask application provides a RESTful API for inventory management.

### Features

- Product listing with pagination
- Search functionality
- Stock updates
- CSV import

### Setup and Running

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask app:
   ```
   python app.py
   ```
   
   If you encounter a signal error with Python 3.12+, you can use the alternative runner:
   ```
   python run_flask.py
   ```

3. Access the web interface at http://localhost:5000

## Requirements

- Python 3.10+
- Streamlit 1.33+
- Flask 3.0+
- Pandas 2.0+
- Other dependencies listed in requirements.txt

## Directory Structure

```
.
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── static/
│   ├── css/
│   │   └── styles.css # Stylesheet
│   └── js/
│       └── main.js    # Frontend logic
└── templates/
    └── index.html     # Main template
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 