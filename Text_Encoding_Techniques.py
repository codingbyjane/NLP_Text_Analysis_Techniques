# Importing Dependencies

# Data Manipulation
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# NLP Libraries
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Manual Bag of Words (BoW) Implementation

# Defining Sample Corpus
corpus = ['Loved the sound, no battery issues','Sound quality is good; battery life not good']

# Load English Stopwords
stopwords_set = set(stopwords.words('english'))

# Discarding negation terms from the stopwords list as they can significantly alter the meaning of the sentence and are important for sentiment analysis
stopwords_set.discard('not')
stopwords_set.discard('no')

# Text Preprocessing: Tokenization, Lowercasing, and Stopword Removal
reviews=[]
vocabulary=set([]) # Set to store unique words for building the vocabulary

for review in corpus:
    cleaned_text = [word.lower() for word in word_tokenize(review) if word.isalpha() and word.lower() not in stopwords_set]
    reviews.append(cleaned_text)

    for word in cleaned_text:
            vocabulary.add(word)

print(f"Vocabulary Size: {len(vocabulary)}")
print(f"Vocabulary: {vocabulary}")

# Building the index dictionary to map each word in the vocabulary to a unique index for constructing the BoW matrix
index_dictionary = {word:index for index, word in enumerate(vocabulary)}
print(f"Index Dictionary: {index_dictionary}")


# Initialize BoW matrix: rows = reviews, columns = vocabulary
bow_matrix = np.zeros((len(reviews), len(vocabulary)), dtype=int)

for index, review in enumerate(reviews):
      word_counts = Counter(review) # Count the frequency of each word in the review

      for word, count in word_counts.items(): # Iterate over each tuple of (word, count) in the word_counts dictionary
            if word in index_dictionary:
                  bow_matrix[index, index_dictionary[word]] = count # Update the BoW matrix with the count of each word

# Display the BoW matrix for each review
for i, vector in enumerate(bow_matrix):
    print(f"\nBoW for review {i}: {vector}")


# Using Scikit-learn's CountVectorizer for BoW

review_1="Loved the sound, no battery issues"
review_2="Sound quality is good; battery life not good"

# Initialize CountVectorizer with English stopwords and n-gram range of (1, 2) to capture both unigrams and bigrams
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))

# Fit the vectorizer to the corpus and transform the reviews into a BoW matrix
vectorize_bow_matrix = vectorizer.fit_transform([review_1, review_2])

# Create a DataFrame to display the BoW matrix with feature names as columns
bow_df = pd.DataFrame(vectorize_bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Display the BoW DataFrame
print(f"\nBoW Matrix using CountVectorizer: {bow_df}")