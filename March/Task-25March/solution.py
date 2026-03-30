'''
Task 1: Build Smarter Text Generator

 Improve mini LLM:

Add:
Sentence input from user
Better prediction logic
Generate 10-word sentence
'''

import random
from collections import Counter
text = input("Enter training text: ")
words = text.split()
model = {}
for i in range(len(words) - 1):
    word = words[i]
    next_word = words[i + 1]
    if word not in model:
        model[word] = []
    model[word].append(next_word)
def predict_smart(word):
    if word in model:
        next_words = model[word]
        most_common = Counter(next_words).most_common(1)
        return most_common[0][0]
    else:
        return "unknown"
def generate_text(start_word, length=10):
    result = [start_word]
    for _ in range(length - 1):
        next_word = predict_smart(result[-1])
        result.append(next_word)
        if next_word == "unknown":
            break
    return ' '.join(result)
start = input("Enter starting word: ")
print(generate_text(start))