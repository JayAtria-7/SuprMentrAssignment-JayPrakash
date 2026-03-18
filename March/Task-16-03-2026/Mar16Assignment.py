from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
documents = [
    "I love this movie",
    "This movie is terrible",
    "Amazing acting",
    "Worst film ever"
]
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(documents)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print("Bag of Words Matrix:")
print(bow_matrix.toarray())
print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("\nFeature Names (Bag of Words):")
print(bow_vectorizer.get_feature_names_out())
print("\nFeature Names (TF-IDF):")
print(tfidf_vectorizer.get_feature_names_out())
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("\nBag of Words DataFrame:")
print(bow_df)
print("\nTF-IDF DataFrame:")
print(tfidf_df)
