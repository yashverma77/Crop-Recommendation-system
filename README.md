# Crop Recommendation System

This is a machine learning-based web application that recommends the best crop to grow based on soil and environmental conditions. The application uses a Random Forest model trained on a dataset of crop requirements.

## Features

- **Modern & Responsive UI:** A clean and intuitive user interface built with Bootstrap.
- **Live Input Validation:** Real-time validation for user inputs to prevent errors and guide the user.
- **Enhanced Input Aids:** Sliders for temperature, humidity, and pH to ensure realistic values.
- **Asynchronous Predictions:** A farm-themed loading animation provides a smooth user experience while the model generates a prediction.
- **Detailed & Transparent Results:** The recommendation card provides comprehensive information, including:
  - **Confidence Score:** The model's confidence percentage for the prediction.
  - **Ideal Growing Conditions:** A summary of the optimal N, P, K, temperature, humidity, pH, and rainfall for the suggested crop.
  - **Detailed Crop Care:** A "Why this crop?" button reveals a modal with the ideal sowing period, growth cycle, and specific care tips.

## How to Run the Project

1.  **Clone the repository (or download the source code).**

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Flask application:**
    ```bash
    python app.py
    ```

4.  **Open your web browser** and navigate to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Project Structure
```
.
├── app.py                      # Main Flask application
├── model.pkl                   # Trained RandomForest model
├── minmaxscaler.pkl            # MinMaxScaler object
├── Crop_recommendation.csv     # Dataset used for training
├── requirements.txt            # Python dependencies
├── templates
│   └── index.html              # Frontend HTML and CSS
└── static
    └── crop.png                # Image asset
``` 