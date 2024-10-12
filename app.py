from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import numpy as np
import pickle
import os 

app = Flask(__name__)

# Load models
models = {
    'Naive Bayes': 'models/NaiveBayesModel.pkl',
    'Logistic Regression': 'models/LogisticRegressionModel.pkl',
    'Decision Tree': 'models/DecisionTreeModel.pkl',
    'Random Forest': 'models/RandomForestModel.pkl',
    'SVM': 'models/SVMModel.pkl'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    random_news_input = request.form['news']

    predictions = []

    for model_name, model_file in models.items():
        with open(model_file, 'rb') as f:
            loaded_model = pickle.load(f)
            random_prediction = loaded_model.predict([random_news_input])
            predictions.append((model_name, random_prediction[0].capitalize()))  # Ensure consistent prediction labels

    # Calculate final prediction
    fake_count = sum(1 for _, prediction in predictions if prediction == 'Fake')
    not_fake_count = len(predictions) - fake_count
    final_prediction = 'Fake' if fake_count > not_fake_count else 'Not Fake'

    accuracy = 98.32  # Average of all model accuracy

    return render_template('result.html', news=random_news_input, predictions=predictions, final_prediction=final_prediction, accuracy=accuracy)

@app.route('/analyze_via_link', methods=['POST'])
def analyze_via_link():
    news_link = request.form['news_link']
    if not news_link:
        return render_template('index.html', error='Please provide a news link.')

    # Extract content from the provided link
    try:
        response = requests.get(news_link)
        soup = BeautifulSoup(response.text, 'html.parser')
        scrapped_content = soup.get_text()
    except Exception as e:
        return render_template('index.html', error='Failed to extract content from the provided link.')

    predictions = []

    for model_name, model_file in models.items():
        with open(model_file, 'rb') as f:
            loaded_model = pickle.load(f)
            random_prediction = loaded_model.predict([scrapped_content])
            predictions.append((model_name, random_prediction[0].capitalize()))  # Ensure consistent prediction labels

    # Calculate final prediction
    fake_count = sum(1 for _, prediction in predictions if prediction == 'Fake')
    not_fake_count = len(predictions) - fake_count
    final_prediction = 'Fake' if fake_count > not_fake_count else 'Not Fake'

    accuracy = 98.32  # Average of all model accuracy

    return render_template('result.html', news=scrapped_content, predictions=predictions, final_prediction=final_prediction, accuracy=accuracy)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
