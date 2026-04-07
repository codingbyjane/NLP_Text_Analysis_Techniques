# Importing Dependencies

# Data Manipulation
import numpy as np
import pandas as pd
from collections import defaultdict

# NLP Libraries
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer