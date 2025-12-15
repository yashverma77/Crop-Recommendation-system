import os
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import sklearn
import pickle

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
mx_path = os.path.join(os.path.dirname(__file__), 'minmaxscaler.pkl')

model = pickle.load(open(model_path, 'rb'))
mx = pickle.load(open(mx_path, 'rb'))

# Load the crop data and calculate summary statistics
crop_data = pd.read_csv('Crop_recommendation.csv')
crop_summary = crop_data.groupby('label').agg(['mean'])
crop_summary.columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Create a mapping from label number to crop name
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

crop_details = {
    'lentil': {
        'sowing_period': 'October to November',
        'growth_cycle': '100-130 days',
        'care_tips': 'Requires well-drained soil. Avoid waterlogging. Minimal fertilizer needed.'
    },
    'rice': {
        'sowing_period': 'June to July (Kharif season)',
        'growth_cycle': '90-150 days',
        'care_tips': 'Requires abundant water and high humidity. Plant in puddled fields.'
    },
    'maize': {
        'sowing_period': 'June to July (Kharif) or October to November (Rabi)',
        'growth_cycle': '90-120 days',
        'care_tips': 'Needs well-drained soil and good nitrogen supply. Sensitive to water stress during tasseling.'
    },
    'jute': {
        'sowing_period': 'March to May',
        'growth_cycle': '120-150 days',
        'care_tips': 'Prefers warm, humid climates and well-drained loamy soil. Requires significant rainfall.'
    },
    'cotton': {
        'sowing_period': 'April to June',
        'growth_cycle': '140-160 days',
        'care_tips': 'Needs a long frost-free period, plenty of sun, and moderate rainfall. Sandy loam soil is ideal.'
    },
    'coconut': {
        'sowing_period': 'Year-round in tropical climates',
        'growth_cycle': '6-10 years to first fruit',
        'care_tips': 'Thrives in sandy soils with high humidity and regular rainfall. Very salt-tolerant.'
    },
    'papaya': {
        'sowing_period': 'Spring and Summer',
        'growth_cycle': '7-11 months to first fruit',
        'care_tips': 'Loves sun and heat. Requires well-drained soil to avoid root rot. Short-lived but fast-growing.'
    },
    'orange': {
        'sowing_period': 'Spring or Fall',
        'growth_cycle': '3-5 years to first fruit',
        'care_tips': 'Prefers well-drained soil and full sun. Protect from frost. Regular watering and fertilization are key.'
    },
    'apple': {
        'sowing_period': 'Spring or Fall',
        'growth_cycle': '2-5 years to first fruit',
        'care_tips': 'Requires a cold period for dormancy. Well-drained soil and full sun are essential. Pruning is important for fruit production.'
    },
    'grapes': {
        'sowing_period': 'Early Spring',
        'growth_cycle': '2-3 years to first fruit',
        'care_tips': 'Needs full sun and well-drained soil. Trellising is required. Prune heavily in winter.'
    },
    'mango': {
        'sowing_period': 'Late Spring or Summer',
        'growth_cycle': '3-6 years to first fruit',
        'care_tips': 'Tropical tree that needs lots of sun and protection from frost. Well-drained soil is crucial.'
    }
}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        mx_features = mx.transform(single_pred)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(mx_features)
        confidence = np.max(probabilities)
        prediction_label = model.classes_[np.argmax(probabilities)]
        
        crop_name_lower = crop_dict.get(prediction_label, "Unknown").lower()
        
        if crop_name_lower != "unknown":
            ideal_conditions = crop_summary.loc[crop_name_lower].to_dict()
            
            explanation = f"This crop is recommended because your soil and weather conditions are close to its ideal needs."
            
            details = crop_details.get(crop_name_lower, {
                'sowing_period': 'N/A',
                'growth_cycle': 'N/A',
                'care_tips': 'No specific care tips available.'
            })

            result = {
                'crop_name': crop_name_lower.capitalize(),
                'confidence': f"{confidence*100:.0f}%",
                'ideal_conditions': {k: f"{v:.2f}" for k, v in ideal_conditions.items()},
                'explanation': explanation,
                'details': details
            }
            return jsonify(result)
        else:
            return jsonify({'error': "Could not determine the best crop."}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
