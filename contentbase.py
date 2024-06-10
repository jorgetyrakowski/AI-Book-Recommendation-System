import pandas as pd
import numpy as np
import math as math
from ast import literal_eval
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import time

def load_data():
    """
    Load the dataset from a CSV file and return the loaded data.

    Returns:
    pandas.DataFrame: The loaded dataset.
    """
    data = pd.read_csv(r'books_enriched.csv', index_col=[0], converters={col: literal_eval for col in ['authors', 'genres', 'author2']})
    return data



data = load_data()
data['description'] = data['description'].apply(lambda x: BeautifulSoup(x).get_text() if pd.isnull(x)==False else x)
# Selecting the features for the content-based filtering model

features = data[["book_id","authors","original_publication_year","genres","title","description"]]
features = features.astype(str)

# Combining the 'authors', 'genres', and 'description' columns into a single column
features["content"] = features['authors'] + ' '  + features['genres'] + ' '  + features['description']
features = features.reset_index()


features["book_id"] = features["book_id"].astype(str).astype(int)
features["title"] = features["title"].astype(str)


start=time.time()
    
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

features['processed_content'] = features['content'].apply(preprocess_text)
end=time.time()


indices = pd.Series(features.index, index=features['title'])
tfv = TfidfVectorizer(stop_words='english')
tf_matrix = tfv.fit_transform(features["content"])
cos_distance_extended = cosine_similarity(tf_matrix,tf_matrix)


def get_recommendations(
    title, distance_metric=cos_distance_extended, num_books=10, book_id=False
):
    if title not in indices:
        raise ValueError(f"Title '{title}' not found in indices.")
    idx = indices[title]

    # Get the pairwise similarity scores using the specified distance
    sim_scores = list(enumerate(distance_metric[idx]))

    # Sort the list of indices and scores by descending similarity scores
    sim_scores = sorted(sim_scores, key=lambda tup: tup[1].sum(), reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:num_books]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    if book_id == True:
        # Return the top 10 most similar books
        return list(features["book_id"].iloc[book_indices])

    else:
        return list(features["title"].iloc[book_indices])

books=get_recommendations("Harry Potter Boxset (Harry Potter, #1-7)", distance_metric=cos_distance_extended)

with open('tfid.pkl','wb') as file:
    pickle.dump(cos_distance_extended,file)
