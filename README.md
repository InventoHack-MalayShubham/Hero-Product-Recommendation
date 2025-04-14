# Inventory Dashboard

A modern, responsive inventory management system built with Streamlit.

## Features

- **Manual Entry Mode**: Add products with details including name, category, stock, expiry date, and optional image URL
- **Bill Verification Mode**: Verify bills against inventory
- **CSV Import**: Import product data from CSV files
- **Search and Filter**: Search by product name or category, filter by expiry date and stock levels
- **Expiry Alerts**: Products expiring within 7 days are highlighted with a warning badge
- **Low Stock Alerts**: Products with stock below 30 units are highlighted with a warning badge
- **GST Auto-Suggestion**: GST percentage is automatically suggested based on product category
- **Product Images**: Optional image URLs for products
- **Export Options**: Export full inventory or filtered view as CSV

## Setup and Running

1. Install the required dependencies:
   ```
   pip install -r requirements_streamlit.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run streamlitapp.py
   ```

3. Access the dashboard in your web browser at http://localhost:8501

## Requirements

- Python 3.10+
- Streamlit 1.33+
- Pandas 2.0+
- NumPy 1.24+
- Pillow 10.0+
- Requests 2.31+
- Python-dateutil 2.8.2+

## Directory Structure

```
.
├── streamlitapp.py          # Main Streamlit application
├── requirements_streamlit.txt # Dependencies
└── README.md               # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 