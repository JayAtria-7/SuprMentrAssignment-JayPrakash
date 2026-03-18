'''
Task 1: Movie Review Classifier

 Build a model:

Dataset: 20+ sentences

Classes: Positive / Negative / Neutral

 Bonus:

Use TF-IDF instead of CountVectorizer
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer   
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
data = {
    "text": [
        "I love this movie",
        "This film was amazing",
        "Absolutely fantastic experience",
        "I really enjoyed the film",
        "Best movie ever",
        "It was a great performance",
        "I am very happy with this movie",

        "Worst movie ever",
        "I hate this film",
        "Terrible experience",
        "Not worth watching",
        "Very bad acting",
        "I am disappointed",
        "Awful storyline",

        "It was okay",
        "Average movie",
        "Nothing special",
        "Just fine",
        "It was neither good nor bad",
        "Mediocre experience"
    ],
    "label": [
        "positive","positive","positive","positive","positive","positive","positive",
        "negative","negative","negative","negative","negative","negative","negative",
        "neutral","neutral","neutral","neutral","neutral","neutral"
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

sample_text = ["I really enjoyed this film"]
sample_vec = vectorizer.transform(sample_text)
prediction = model.predict(sample_vec)

print("Prediction for sample text:", prediction[0])