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
plt.figure(figsize=(10, 8))
sns.heatmap(dense_tfidf_matrix, annot=True, cmap="coolwarm", xticklabels=tfidf_vectorizer.get_feature_names_out(), yticklabels=reviews_df['review_id'])
plt.title("TF-IDF Matrix Heatmap")
plt.xlabel("Features (Top 20 Words)")
plt.ylabel("Reviews")
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.yticks(rotation=0) # Keep y-axis labels horizontal
#plt.show()

# Initialize the LSA Model using TruncatedSVD (Singular Value Decomposition)
LSA_model = TruncatedSVD(n_components=2, n_iter=10, algorithm='randomized', random_state=42) # Reduce the dimensionality to 2 components for visualization (those components are the topics and ultimately the dimensions of the new space)

# Fit the LSA model to the TF-IDF matrix and transform it into a lower-dimensional space
lsa_matrix = LSA_model.fit_transform(tfidf_matrix)


print(f"LSA Matrix Shape: {lsa_matrix.shape}\n") # This is the matrix A shaped n*k where n=10 (number of reviews) and k=2 (number of topics/components)
print(f"LSA Matrix (lower-dimensional representation):\n{lsa_matrix}\n")

singular_values = LSA_model.singular_values_ # Get the singular values from the fitted LSA model
print(f"Singular Values: {singular_values}\n")


# Display the right singular matrix P which contains the document-topic associations. Each row corresponds to one review and each column represents a topic. The values indicate the strength of correlation between each review and each topic. Higher values indicate a stronger association with that topic.
P = LSA_model.transform(tfidf_matrix)
print(f"Document-Topic Matrix (P):\n{P}\n")

# Display the left singular matrix K^T (transposed) which contains the topic-word associations. Each row represents a topic and each column corresponds to a word/feature. The values indicate the strength of association between each topic and each word.
K_transposed = LSA_model.components_
print(f"LSA Components (Topic-Word Associations):\n{K_transposed}\n")

# Display the diagonal matrix L which represents the importance of each topic (singular values). The values on the diagonal indicate the amount of variance explained by each topic. Higher values indicate more important topics that capture more of the underlying structure in the data.
L = np.diag(singular_values)
print(f"Diagonal Matrix L (Topic Importance):\n{L}\n")


l=lsa_matrix[0]
print("Review 0:")
for i,topic in enumerate(l): 
    print("Topic ", i, ": ", topic*100)