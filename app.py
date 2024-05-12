from flask import Flask, render_template, request
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

app = Flask(__name__)

# Load models
models = {
    'Naive Bayes': 'NaiveBayesModel.pkl',
    'Logistic Regression': 'LogisticRegressionModel.pkl',
    'Decision Tree': 'DecisionTreeModel.pkl',
    'Random Forest': 'RandomForestModel.pkl',
    'SVM': 'SVMModel.pkl'
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
            predictions.append((model_name, random_prediction[0]))

    # Calculate final prediction
    fake_count = sum(1 for _, prediction in predictions if prediction == 'fake')
    not_fake_count = len(predictions) - fake_count
    final_prediction = 'Fake' if fake_count > not_fake_count else 'Not Fake'

    accuracy = 98.32  # Adjust accuracy accordingly

    return render_template('result.html', news=random_news_input, predictions=predictions, final_prediction=final_prediction, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
