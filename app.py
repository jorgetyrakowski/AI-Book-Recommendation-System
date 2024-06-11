import streamlit as st
import pandas as pd
import pickle
from surprise import Dataset, Reader, NormalPredictor
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
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

# Same as previous function, but with cosimilarity matrix (for baseline test).
def recommend_similar_books_tfid_baseline(book_name, tfid_model, books_tfid, k=5):
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

    # Calculate cosine similarity
    sim_matrix = cosine_similarity(tfid_model, tfid_model)
    sim_scores = list(enumerate(sim_matrix[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k + 1]
    book_indices = [i[0] for i in sim_scores]

    recommended_books = books_tfid.iloc[book_indices][['book_id', 'title', 'authors', 'image_url']]
    return recommended_books

# Prediction Model for Baseline Test Comparison
def hybrid_predict(user_id, book_id, svd_model, tfid_model, books_df):
    """ Predict the rating for a given user and book using hybrid model. """
    # SVD prediction
    try:
        svd_prediction = svd_model.predict(user_id, book_id).est
    except ValueError:
        svd_prediction = np.nan
    
    # TF-IDF prediction (using average rating of similar books)
    #book_name = books_df.loc[books_df['book_id'] == book_id, 'title'].values[0]
    #similar_books = recommend_similar_books_tfid_baseline(book_name, tfid_model, books_df)
    #if not similar_books.empty:
    #    tfid_prediction = books_df.loc[books_df['book_id'].isin(similar_books['book_id']), 'average_rating'].mean()
    #else:
    #    tfid_prediction = np.nan
    tfid_prediction = np.nan
    
    # Combine the predictions (simple average, you can use a weighted average or other methods)
    if np.isnan(svd_prediction) and np.isnan(tfid_prediction):
        return np.nan
    elif np.isnan(svd_prediction):
        return tfid_prediction
    elif np.isnan(tfid_prediction):
        return svd_prediction
    else:
        return (svd_prediction + tfid_prediction) / 2

# For Baseline Test Comparison
def compute_rmse_mae(testset, svd_model, tfid_model, books_df):
    """ Compute RMSE and MAE for the hybrid model. """
    hybrid_predictions = []
    actual_ratings = []

    for uid, iid, true_r in testset:
        pred = hybrid_predict(uid, int(iid), svd_model, tfid_model, books_df)
        if not np.isnan(pred):
            hybrid_predictions.append((uid, iid, true_r, pred, None))  # Format for accuracy metrics
            actual_ratings.append((uid, iid, true_r, pred, None))  # Format for accuracy metrics

    rmse = accuracy.rmse(hybrid_predictions, verbose=False)
    mae = accuracy.mae(hybrid_predictions, verbose=False)
    return rmse, mae

'''
ver 1.1 (With Baseline Test)
'''

# Main function for the Streamlit app
def streamlit_app():
    st.title('Hybrid Book Recommendation System')

    script_dir = os.path.dirname(os.path.abspath(__file__))     # M
    fp_svd = os.path.join(script_dir, 'models','svd_model.pkl')
    fp_tfid = os.path.join(script_dir, 'models','tfid.pkl')
    fp_bookdf = os.path.join(script_dir, 'data','books_enriched.csv')
    fp_ratings = os.path.join(script_dir, 'data','ratings.csv')

    # Load the pre-trained models
    svd_model = load_model(fp_svd)
    tfid_model = load_model(fp_tfid)

    # Load the datasets
    books_df = pd.read_csv(fp_bookdf)
    ratings_df = pd.read_csv(fp_ratings)

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

    # Button for Baseline test using surprise
    elif st.button('Baseline Test'):        # M

        recommended_books_svd = recommend_similar_books_svd(selected_book_id, svd_model, books_df)
        recommended_books_tfid = recommend_similar_books_tfid(selected_title, tfid_model, books_df)

        # Combine the recommendations and remove duplicates
        all_recommended_books = pd.concat([recommended_books_svd, recommended_books_tfid]).drop_duplicates(
            subset='book_id').head(10)
        
        status_text = st.status("Evaluating baseline model... Please wait.")
        progress_bar = st.progress(0)
        
        ###########################################################
        ### Evaluate the baseline model using cross-validation
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['user_id', 'book_id', 'rating']], reader)
        _, testset = train_test_split(data, test_size=0.2)

        baseline_model = NormalPredictor()
        progress_bar.progress(20)

        cv_results = cross_validate(baseline_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        progress_bar.progress(36)
        baseline_rmse = cv_results['test_rmse'].mean()
        baseline_mae = cv_results['test_mae'].mean()
        progress_bar.progress(50)

        st.write(f"Baseline Model - NormalPredictor")
        st.write(f"Average RMSE: {baseline_rmse}")
        st.write(f"Average MAE: {baseline_mae}")

        ##############################################
        ### Evaluate our hybrid model 

        status_text.update(label = "Evaluating hybrid model... Please wait.")

        hybrid_rmse, hybrid_mae = compute_rmse_mae(testset, svd_model, tfid_model, books_df)

        progress_bar.progress(100)
        status_text.update(label = 'Baseline Test Complete!', state="complete")

        st.write(f"Hybrid Model (SVD + TF-IDF)")
        st.write(f"RMSE: {hybrid_rmse}")
        st.write(f"MAE: {hybrid_mae}")


# Run the Streamlit app
if __name__ == '__main__':
    streamlit_app()
