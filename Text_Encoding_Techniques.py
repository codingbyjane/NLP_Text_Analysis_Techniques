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

# Machine Learning Pipeline
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

# Evaluation Metrics
from sklearn.metrics import classification_report


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

# Define the reviews to be vectorized
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


# Implementing a Naive Bayes classifier pipeline

# Create a sample dataset of reviews and their corresponding labels (1 for positive, 0 for negative)
reviews_df = pd.DataFrame({
      'review text': ['Loved the sound, no battery issues',
                    'Sound quality is good; battery life not good', 
                    'The sound is amazing and battery lasts long', 
                    'Not satisfied with the sound quality and battery performance',
                    'Battery life is excellent but sound quality is poor',
                    'I am very happy with the sound and battery',
                    'The sound is terrible and battery dies quickly',
                    'Great sound quality but battery life is disappointing',
                    "The sound is fantastic but battery life is mediocre",
                    "I am impressed with the sound quality but disappointed with the battery performance"],

      'label': [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    })

naive_bayes_pipeline = Pipeline([
      ("vectorizer", CountVectorizer(stop_words=list(stopwords_set), ngram_range=(1, 2))), # Step 1: Convert text to BoW features
      ("classifier", MultinomialNB()) # Step 2: Train a Naive Bayes classifier on the BoW features
])

naive_bayes_pipeline.fit(reviews_df['review text'], reviews_df['label']) # Fit the pipeline to the reviews and their labels

# Predict the labels for the training data and generate a classification report
predictions = naive_bayes_pipeline.predict(reviews_df['review text'])
print(f"\nPredictions: {predictions}")

# Generate a classification report
classification_report_dict = classification_report(reviews_df['label'], predictions, output_dict=True)

# Create a DataFrame to display the classification report
classification_report_df = pd.DataFrame(classification_report_dict).transpose()
print(f"\nClassification Report:\n{classification_report_df}")


# Using TfidfVectorizer for TF-IDF representation
tfidf_vectorizer = TfidfVectorizer(stop_words=list(stopwords_set), min_df=1, ngram_range=(1, 2)) # Initialize TfidfVectorizer with the previously defined set of English stopwords, set minimum document frequency, and n-gram range (unigrams and bigrams)

# Fitting the TfidfVectorizer to the reviews and transforming them into a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_df['review text'])

# Display the learned vocabulary and feature names from the TfidfVectorizer after fitting the data
vocabulary = tfidf_vectorizer.vocabulary_
feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"\nLearned vocabulary: {vocabulary}")
print(f"\nFeature names: {feature_names}")

# Create a DataFrame to display the TF-IDF matrix with feature names as columns
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
print(f"\nTF-IDF Matrix using TfidfVectorizer:\n{tfidf_df}")

tfidf_word_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten() # Calculate the sum of TF-IDF scores for each word across all reviews
tfidf_word_scores_dict = dict(zip(feature_names, tfidf_word_scores)) # Create a dictionary to map each word to its corresponding TF-IDF score
print(f"\nTF-IDF Word Scores: {list(tfidf_word_scores_dict.items())[:10]}\n")


# Display the score of each word in the TF-IDF matrix for each word in the feature names list
for word in feature_names:
      index = tfidf_vectorizer.vocabulary_.get(word) # Get the index of the word in the TF-IDF matrix
      print(f"Word: '{word}' - TF-IDF Score: {tfidf_word_scores_dict.get(word, 0)}")