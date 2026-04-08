# Importing Dependencies
import pandas as pd

# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample Corpus
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


# Initialize the TF-IDF Vectorizer (create an instance of the TfidfVectorizer class)
tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=20, smooth_idf=True) # Smooth IDF adds 1 to document frequencies to prevent the zero division possibility

# Fit the TF-IDF Vectorizer to the review texts and transform them into a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_df['review text'])