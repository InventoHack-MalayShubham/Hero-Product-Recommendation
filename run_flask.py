from app import app

if __name__ == '__main__':
    # Run the Flask app without the reloader to avoid signal errors
    app.run(debug=True, use_reloader=False) 