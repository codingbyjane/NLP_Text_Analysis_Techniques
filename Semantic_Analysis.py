# Importing Dependencies
import pandas as pd
import numpy as np

# Plotting & Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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

    'label': [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'review_id': ["Review 1", "Review 2", "Review 3", "Review 4", "Review 5", "Review 6", "Review 7", "Review 8", "Review 9", "Review 10"]
    })


# Initialize the TF-IDF Vectorizer (create an instance of the TfidfVectorizer class)
tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=20, smooth_idf=True) # Smooth IDF adds 1 to document frequencies to prevent the zero division possibility

# Fit the TF-IDF Vectorizer to the review texts and transform them into a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_df['review text'])
dense_tfidf_matrix = np.array(tfidf_matrix.toarray())

# Display the shape of the TF-IDF matrix
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}\n") # This is the matrix B shaped n*m where n=10 (number of reviews) and m=20 (number of features)
print(f"TF-IDF Matrix (dense representation):\n{dense_tfidf_matrix}\n")

# Plot the TF-IDF matrix as a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(dense_tfidf_matrix, annot=True, cmap="Blues", xticklabels=tfidf_vectorizer.get_feature_names_out(), yticklabels=reviews_df['review_id'])
plt.title("TF-IDF Matrix Heatmap")
plt.xlabel("Features (Top 20 Words)")
plt.ylabel("Reviews")
#plt.show()