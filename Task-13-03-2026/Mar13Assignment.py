'''
Task 1: Build Text Preprocessing Tool

Create a program that:

Input:

"I am learning NLP and it is very exciting!!!"

Your program should:

convert to lowercase

remove punctuation

tokenize text

remove stopwords

apply stemming

Output example:

['learn','nlp','excit']
'''

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
text="I am learning NLP and it is very exciting!!!"
text=text.lower()
print(text)
text=text.translate(str.maketrans('','',string.punctuation))
print(text)
tokens=word_tokenize(text)
print(tokens)
stop_words=set(stopwords.words('english'))
filtered_words=[word for word in tokens if word not in stop_words]
print(filtered_words)
stemmer=PorterStemmer()
stemmed_words=[stemmer.stem(word) for word in filtered_words]
print(stemmed_words)