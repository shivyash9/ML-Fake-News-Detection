import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import string
from nltk.corpus import stopwords
from nltk import tokenize
from wordcloud import WordCloud
import seaborn as sns
import nltk
import pickle
import os

nltk.download('stopwords')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Read data
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add target labels
fake['target'] = 'fake'
true['target'] = 'true'

# Concatenate dataframes
data = pd.concat([fake, true]).reset_index(drop=True)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Remove unnecessary columns
data.drop(["date", "title"], axis=1, inplace=True)

# Text preprocessing
stop = stopwords.words('english')
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop])
    return text

data['text'] = data['text'].apply(preprocess_text)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

# Data exploration
# 1. How many articles per subject?
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.title("Number of Articles per Subject")
plt.xlabel("Subject")
plt.ylabel("Number of Articles")
plt.show()

# 2. How many fake and real articles?
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.title("Number of Fake and Real Articles")
plt.xlabel("Target")
plt.ylabel("Number of Articles")
plt.show()

# 3. Word cloud for fake news
fake_data = data[data["target"] == "fake"]
all_words_fake = ' '.join([text for text in fake_data.text])
wordcloud_fake = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(all_words_fake)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.title("Word Cloud for Fake News")
plt.axis("off")
plt.show()

# 4. Word cloud for real news
real_data = data[data["target"] == "true"]
all_words_real = ' '.join([text for text in real_data.text])
wordcloud_real = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(all_words_real)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.title("Word Cloud for Real News")
plt.axis("off")
plt.show()

# 5. Most frequent words counter
token_space = tokenize.WhitespaceTokenizer()

def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()), "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns="Frequency", n=quantity)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_frequency, x="Word", y="Frequency", color='blue')
    ax.set(ylabel="Count")
    plt.xticks(rotation='vertical')
    plt.title("Most Frequent Words")
    plt.show()

# 6. Most frequent words in fake news
counter(data[data["target"] == "fake"], "text", 20)

# 7. Most frequent words in real news
counter(data[data["target"] == "true"], "text", 20)

# Models to train
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=20, splitter='best', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, criterion="entropy"),
    'SVM': SVC(kernel='linear')
}

# Train and save models
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

for model_name, model_obj in models.items():
    print(f"Training {model_name}...")
    model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', model_obj)
    ])
    model.fit(X_train, y_train)
    
    # Save the trained model
    with open(os.path.join(MODELS_DIR, f'{model_name.replace(" ", "")}Model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    # Test the model with test data
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, prediction)
    plt.figure()
    plot_confusion_matrix(cm, classes=['Fake', 'Real'], title=f'Confusion matrix for {model_name}')
    plt.show()
