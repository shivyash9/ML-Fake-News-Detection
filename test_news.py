import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Models to test data 
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=20, splitter='best', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, criterion="entropy"),
    'SVM': SVC(kernel='linear')
}

# Load models and test them with random news input
loaded_models = {}

random_news_input = "The government announced new policies to tackle unemployment."

print("News:")
print(random_news_input)

predictions = []

for model_name, model in models.items():
    with open(f'{model_name.replace(" ", "")}Model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
        loaded_models[model_name] = loaded_model

        # Test the loaded model with random news input
        random_prediction = loaded_model.predict([random_news_input])
        predictions.append((model_name, random_prediction[0]))

        print(f"Model Name: {model_name}\tPrediction: {'Fake' if random_prediction[0] == 'fake' else 'Not Fake'}")

# Calculate the weighted contribution of each model based on accuracy
model_weights = {
    "Decision Tree": 0.9971,
    "SVM": 0.9945,
    "Random Forest": 0.9872,
    "Logistic Regression": 0.9872,
    "Naive Bayes": 0.9506
}

# Count the number of predictions for each class
fake_count = 0
not_fake_count = 0
for model_name, prediction in predictions:
    if prediction == 'fake':
        fake_count += model_weights.get(model_name, 0)
    else:
        not_fake_count += model_weights.get(model_name, 0)

# Decide the final prediction based on the majority
if fake_count == not_fake_count:
    final_prediction = 'Undetermined'
else:
    final_prediction = 'Fake' if fake_count > not_fake_count else 'Not Fake'

print("\nFinal Prediction:")
print(f"The majority of models predict that the news is: {final_prediction}")
print("Accuracy: 98.32%")
