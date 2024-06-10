import streamlit as st
import pandas as pd
import pickle
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from ast import literal_eval

import re


# Function to load the pre-trained SVD model
def load_model(filepath):
    with open(filepath, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


# Function to get book details (title and image URL) given a book ID
def get_book_info(book_id, books_df):
    """ Get the book title and image URL given a book ID. """
    book_info = books_df.loc[books_df['book_id'] == book_id, ['title', 'image_url', 'authors']].iloc[0]
    return book_info


# Function to recommend books similar to a given book using SVD
def recommend_similar_books_svd(book_id, svd_model, books_sample, k=5):
    """ Recommend books similar to the given book ID using SVD model. """
    try:
        # Get the inner ID of the book
        book_inner_id = svd_model.trainset.to_inner_iid(book_id)
    except ValueError:
        return pd.DataFrame()

    # Get item factors from the SVD model
    item_factors = svd_model.qi
    # Calculate similarities with all other books
    similarities = item_factors.dot(item_factors[book_inner_id])
    # Get the top k similar books
    similar_books_inner_ids = similarities.argsort()[::-1][1:k + 1]
    similar_books_ids = [svd_model.trainset.to_raw_iid(inner_id) for inner_id in similar_books_inner_ids]
    # Get details of similar books
    similar_books_details = books_sample[books_sample['book_id'].isin(similar_books_ids)][
        ['book_id', 'title', 'authors', 'image_url']]

    return similar_books_details


def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text


def recommend_similar_books_tfid(book_name, tfid_model, books_tfid, k=5):
    """ Recommend books similar to the given book name using TF-IDF model. """
    features = books_tfid[["book_id", "authors", "title", "description"]]
    features = features.astype(str)
    features["content"] = features['authors'] + ' ' + features['description']
    features['processed_content'] = features['content'].apply(preprocess_text)
    indices = pd.Series(features.index, index=features['title'])

    try:
        idx = indices[book_name]
    except KeyError:
        st.error(f"Book title '{book_name}' not found.")
        return []

    sim_scores = list(enumerate(tfid_model[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k + 1]
    book_indices = [i[0] for i in sim_scores]

    recommended_books = books_tfid.iloc[book_indices][['book_id', 'title', 'authors', 'image_url']]
    return recommended_books


# Main function for the Streamlit app
def streamlit_app():
    st.title('Hybrid Book Recommendation System')

    # Load the pre-trained models
    svd_model = load_model('models/svd_model.pkl')
    tfid_model = load_model('models/tfid.pkl')

    # Load the datasets
    books_df = pd.read_csv('data/books_enriched.csv')
    ratings_df = pd.read_csv('data/ratings.csv')

    # Create a dictionary to map book titles to their IDs
    title_to_book_id = books_df.set_index('title')['book_id'].to_dict()
    book_id_to_title = books_df.set_index('book_id')['title'].to_dict()

    # Allow the user to select a book by its title
    selected_title = st.selectbox('Choose a book to get recommendations:', books_df['title'].unique())
    selected_book_id = title_to_book_id[selected_title]

    # Display the selected book's title and image
    book_info = get_book_info(selected_book_id, books_df)
    st.image(book_info['image_url'], width=200)

    # Button to get recommendations
    if st.button('Recommend Books'):
        recommended_books_svd = recommend_similar_books_svd(selected_book_id, svd_model, books_df)
        recommended_books_tfid = recommend_similar_books_tfid(selected_title, tfid_model, books_df)

        # Combine the recommendations and remove duplicates
        all_recommended_books = pd.concat([recommended_books_svd, recommended_books_tfid]).drop_duplicates(
            subset='book_id').head(10)

        if not all_recommended_books.empty:
            st.write('Recommended Books:')

            # Create columns for each recommendation and display images
            cols = st.columns(5)  # Adjust the number of columns based on the desired layout
            for i, (_, book) in enumerate(all_recommended_books.iterrows()):
                with cols[i % 5]:
                    st.image(book['image_url'], width=150)
                    st.write(book['title'])
                    st.write(f"by {book['authors']}")


# Run the Streamlit app
if __name__ == '__main__':
    streamlit_app()
