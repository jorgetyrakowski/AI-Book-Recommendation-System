#!/usr/bin/env python
# coding: utf-8

# ## Book Recommendation System 
# 

# ## Dataset Structure and Contents
# ##### 1.	ratings.csv: contains book ratings given by users, with a scale from 1 to 5. It is crucial for a collaborative recommendation model as it provides direct interaction between user and book that can be used to calculate similarities between users or between items (books).
#     o	Columns: user_id, book_id, rating
#     o	Size: 5,976,479 rows
#     o	Details: Contains ratings given by users to books, with ratings ranging from 1 to 5.
#     
# ##### 2.	to_read.csv: provides metadata about each book, such as Goodreads IDs, authors, titles and average ratings. This data enriches the recommendations and allows content-based filtering if you want to combine both methods.
#     o	Columns: user_id, book_id
#     o	Size: 912,705 rows
#     o	Details: Lists books marked as "to-read" by users.
#     
# ##### 3.	books.csv: includes tags or genres assigned by users to books. These tags help to identify the characteristics of the books and can be used to improve the accuracy of the recommendation system, especially for content-based filtering.
#     o	Columns: 23 (including metadata such as title, author, average rating, etc.)
#     o	Size: 10,000 rows
#     o	Details: Provides metadata for each book, including Goodreads IDs and other relevant information.
#     
# ##### 4.	book_tags.csv: translates tag IDs to names. This is useful for interpreting tags in book_tags.csv and for improving the presentation and comprehensibility of the final recommendations.
# 
#     o	Columns: goodreads_book_id, tag_id, count
#     o	Size: 999,912 rows
#     o	Details: Contains tags assigned by users to books, along with the count of how many times each tag was used.
#     
# ##### 5.	tags.csv: lists books that users have tagged as "to read", indicating preliminary interest. This dataset can be used to better understand user preferences and to refine recommendations.
#     o	Columns: tag_id, tag_name
#     o	Size: 34,252 rows
#     o	Details: Maps tag IDs to their respective names.
#     
# ## Data Loading
# 
# In this section, we will load the datasets required for our book recommendation system and perform an initial exploration of the data.
# 
# 
# 
# ### Data Loading
# 
# In this section, we will load the datasets required for our book recommendation system and perform an initial exploration of the data

# In[11]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get filepaths         # M
script_dir = os.path.dirname(os.path.abspath(__file__))
fp_ratings = os.path.join(script_dir,'data','ratings.csv')
fp_book_tags = os.path.join(script_dir,'data','book_tags.csv')
fp_to_read = os.path.join(script_dir,'data','to_read.csv')
fp_books = os.path.join(script_dir,'data','books_enriched.csv')
fp_tags = os.path.join(script_dir,'data','tags.csv')

# Load the datasets
ratings_sample = pd.read_csv(fp_ratings, nrows=1000000)
book_tags_sample = pd.read_csv(fp_book_tags)
to_read_sample = pd.read_csv(fp_to_read)
books_sample = pd.read_csv(fp_books)
tags_sample = pd.read_csv(fp_tags)

# Display the first few rows of each dataset
print("Ratings Sample:")
print(ratings_sample.head())

print("\nBook Tags Sample:")
print(book_tags_sample.head())

print("\nTo Read Sample:")
print(to_read_sample.head())

print("\nBooks Sample:")
print(books_sample.head())

print("\nTags Sample:")
print(tags_sample.head())


# ### Data Cleaning
# 
# In this section, we will clean the datasets to ensure data quality and consistency. This includes removing duplicate ratings, filtering users with insufficient ratings, and handling missing values in the books dataset.

# In[12]:


# Data Cleaning
# Remove duplicate ratings
ratings_sample['count'] = ratings_sample.groupby(['user_id', 'book_id'])['rating'].transform('count')
ratings_sample = ratings_sample[ratings_sample['count'] == 1].drop(columns=['count'])

# Remove users who rated fewer than 3 books
ratings_sample['count'] = ratings_sample.groupby('user_id')['rating'].transform('count')
ratings_sample = ratings_sample[ratings_sample['count'] > 2].drop(columns=['count'])

# Cleaning the books_sample for missing values
books_sample['isbn'].fillna('Unknown', inplace=True)
books_sample['isbn13'].fillna('Unknown', inplace=True)
books_sample['original_publication_year'].fillna(books_sample['original_publication_year'].median(), inplace=True)
books_sample['original_title'].fillna('Unknown', inplace=True)
books_sample['language_code'].fillna('Unknown', inplace=True)


# ## Data Preparation
# 
# In this section, we will prepare the data for model training by creating a user-item interaction matrix and splitting the data into training and testing sets.
# 
# #### Create User-Item Interaction Matrix
#     1.	Pivot Table: We create a user-item interaction matrix where rows represent users, columns represent books, and values represent the ratings given by users to books. This matrix helps us understand the interactions between users and books, which is essential for collaborative filtering methods.
# 
# #### Split the Data into Training and Testing Sets
#     1.	Fill Missing Values: We fill any missing values in the interaction matrix with zeros. This step ensures that the matrix is complete and can be used for training.
#     2.	Train-Test Split: We split the data into training and testing sets using an 80-20 split. The random_state=42 parameter ensures reproducibility of the split, meaning the same split will be produced each time the code is run.
# 
# #### Ensure Test Data Contains Only Known Users
#     1.	Filter Test Data: We filter the test data to ensure that it only contains users who are also present in the training data. This step is important to ensure that our model is tested on users it has seen during training, providing a fair evaluation of the model's performance.
# 

# In[13]:


# Data Preparation
# Create user-item interaction matrix
user_item_matrix = ratings_sample.pivot(index='user_id', columns='book_id', values='rating')

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
user_item_matrix_df = user_item_matrix.fillna(0)
train_data, test_data = train_test_split(user_item_matrix_df, test_size=0.2, random_state=42)

# Ensure that the test data only contains users that are also in the training data
test_data = test_data[test_data.index.isin(train_data.index)]


# ## Exploratory Data Analysis (EDA)
# 
# #### Ratings Distribution
#     1.	Plot Ratings Distribution: We create a count plot to visualize the distribution of ratings. This helps us understand how the ratings are spread across different values (e.g., 1 to 5 stars).
#         •	Figure Size: Sets the size of the plot.
#         •	Count Plot: Plots the count of each rating value.
#         •	Title, X-Label, Y-Label: Adds a title and labels to the plot for better readability.
#     2.	Identify Most Rated Books: We identify the top 10 most rated books by counting the number of ratings each book has received.
#         •	Value Counts: Counts the number of ratings for each book.
#         •	Head: Selects the top 10 books.
#     3.	Plot Most Rated Books: We create a bar plot to visualize the most rated books.
#         •	Bar Plot: Plots the number of ratings for the top 10 books.
#         •	Title, X-Label, Y-Label: Adds a title and labels to the plot.
#         •	X-Ticks Rotation: Rotates the x-axis labels for better readability.
#     4.	Identify Most Active Users: We identify the top 10 most active users by counting the number of ratings each user has provided.
#         •	Value Counts: Counts the number of ratings each user has provided.
#         •	Head: Selects the top 10 users.
#     5.	Plot Most Active Users: We create a bar plot to visualize the most active users.
#         •	Bar Plot: Plots the number of ratings for the top 10 users.
#         •	Title, X-Label, Y-Label: Adds a title and labels to the plot.
#     6.	Identify Most Common Tags: We identify the top 10 most common tags by counting the number of times each tag appears.
#         •	Value Counts: Counts the number of times each tag appears.
#         •	Head: Selects the top 10 tags.
#     7.	Plot Distribution of Tags: We create a bar plot to visualize the distribution of the most common tags.
#         •	Bar Plot: Plots the count of the top 10 tags.
#         •	Title, X-Label, Y-Label: Adds a title and labels to the plot.
#         •	X-Ticks Rotation: Rotates the x-axis labels for better readability.

# In[14]:


# Exploratory Data Analysis (EDA)
# Ratings distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=ratings_sample)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Most rated books
most_rated_books = ratings_sample['book_id'].value_counts().head(10).index
most_rated_books_details = books_sample[books_sample['book_id'].isin(most_rated_books)]

plt.figure(figsize=(12, 8))
sns.barplot(x=most_rated_books_details['title'], y=most_rated_books_details['book_id'])
plt.title('Top 10 Most Rated Books')
plt.xlabel('Book Title')
plt.ylabel('Number of Ratings')
plt.xticks(rotation=45)
plt.show()

# Most active users
most_active_users = ratings_sample['user_id'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=most_active_users.index, y=most_active_users.values)
plt.title('Top 10 Most Active Users')
plt.xlabel('User ID')
plt.ylabel('Number of Ratings')
plt.show()

# Distribution of tags
tag_counts = book_tags_sample['tag_id'].value_counts().head(10).index
tag_counts_details = tags_sample[tags_sample['tag_id'].isin(tag_counts)]

plt.figure(figsize=(12, 8))
sns.barplot(x=tag_counts_details['tag_name'], y=tag_counts_details['tag_id'])
plt.title('Top 10 Most Common Tags')
plt.xlabel('Tag')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# ## User-Based Collaborative Filtering
# 
# 
# #### Calculate User Similarity Matrix
#     1.	User Similarity Matrix: We calculate the cosine similarity between users based on their ratings. This results in a similarity matrix where each entry (i, j) represents the similarity between user i and user j.
# Recommend Books with Details
# 
# #### 2.	Recommendation Function: We define a function recommend_books_with_details to recommend books for a given user based on the ratings of similar users.
#     •	Find Similar Users: Identify users who are similar to the given user, excluding the user themselves.
#     •	Get Ratings of Similar Users: Retrieve the ratings provided by similar users.
#     •	Identify Unrated Books: Find books that the given user has not rated yet.
#     •	Calculate Recommendations: Compute the average ratings for the unrated books from the similar users' ratings and sort them to get the top k recommendations.
#     •	Get Book Details: Retrieve the titles and authors of the recommended books.
#     
# Recommend Books Similar to a Given Book
# #### 3.	Book-Based Recommendation Function: We define a function recommend_books_user_based to recommend books similar to a given book based on user ratings.
#     •	Item Similarity Matrix: Calculate the cosine similarity between books based on user ratings.
#     •	Find Similar Books: Identify the top k books that are similar to the given book.
#     •	Get Book Details: Retrieve the titles and authors of the similar books and print them along with the original book’s details.
#     
# Example Usage of the Functions
# #### 4.	User-Based Recommendations: We provide an example of how to use the recommend_books_with_details function to get book recommendations for a specific user.
# #### 5.	Book-Based Recommendations: We provide an example of how to use the recommend_books_user_based function to get book recommendations based on a specific book.
# 

# In[15]:


# User-Based Collaborative Filtering
# Calculate user similarity matrix
user_similarity = cosine_similarity(train_data)
user_similarity_df = pd.DataFrame(user_similarity, index=train_data.index, columns=train_data.index)

def recommend_books_with_details(user_id, user_similarity_df, user_item_matrix_df, books_sample, k=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    similar_users_ratings = user_item_matrix_df.loc[similar_users]
    user_unrated_books = user_item_matrix_df.loc[user_id][user_item_matrix_df.loc[user_id] == 0]
    recommendations = similar_users_ratings[user_unrated_books.index].mean().sort_values(ascending=False).head(k)
    recommended_books = books_sample[books_sample['book_id'].isin(recommendations.index)][['title', 'authors']]
    return recommended_books

# Function to recommend books similar to a given book
def recommend_books_user_based(book_id, user_item_matrix_df, books_sample, k=5):
    # Calculate item similarity matrix
    item_similarity = cosine_similarity(user_item_matrix_df.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix_df.columns, columns=user_item_matrix_df.columns)
    
    # Find the top k similar books based on item similarity
    similar_books = item_similarity_df[book_id].sort_values(ascending=False).index[1:k+1]
    
    # Get book details for the similar books
    similar_books_details = books_sample[books_sample['book_id'].isin(similar_books)][['title', 'authors']]
    
    # Print the book details of the original book
    original_book_details = books_sample[books_sample['book_id'] == book_id][['title', 'authors']].iloc[0]
    print(f"Original Book: {original_book_details['title']} by {original_book_details['authors']}")
    
    # Print the recommended books
    print("\nRecommended Books:")
    for index, row in similar_books_details.iterrows():
        print(f"{row['title']} by {row['authors']}")
    
    return similar_books_details

# Example usage of the functions

# User-based recommendations
user_id_example = train_data.index[0]
recommended_books = recommend_books_with_details(user_id_example, user_similarity_df, user_item_matrix_df, books_sample)
user_rated_books = ratings_sample[ratings_sample['user_id'] == user_id_example].merge(books_sample, on='book_id')
#print(f"Books rated by User {user_id_example}:\n", user_rated_books[['title', 'authors', 'rating']].head())
#print("\nRecommended books:\n", recommended_books)

# Book-based recommendations
book_id_example = 13  # Use a book_id from the ratings sample
recommended_books_user_based = recommend_books_user_based(book_id_example, user_item_matrix_df, books_sample)


# ## Item-Based Collaborative Filtering
# 
# #### Calculate Item Similarity Matrix
#     1.	Item Similarity Matrix: We calculate the cosine similarity between items (books) based on user ratings. This results in a similarity matrix where each entry (i, j) represents the similarity between book i and book j.
# Recommend Books with Details for a User
# #### 2.	Recommendation Function: We define a function recommend_books_with_details2 to recommend books for a given user based on the similarity of books they have already rated.
#     •	User Ratings: Retrieve the ratings given by the user.
#     •	Rated Books: Identify the books that the user has already rated.
#     •	Calculate Similarities: For each rated book, find similar books that the user has not rated.
#     •	Aggregate Recommendations: Combine the similarity scores of these unrated books to generate a list of recommended books.
#     •	Get Book Details: Retrieve the titles and authors of the recommended books.
# Recommend Books Similar to a Given Book
# #### 3.	Book-Based Recommendation Function: We define a function recommend_books_item_based to recommend books similar to a given book based on item similarity.
#     •	Find Similar Books: Identify the top k books that are similar to the given book based on the item similarity matrix.
#     •	Get Book Details: Retrieve the titles and authors of the similar books and print them along with the original book’s details.
# Example Usage of the Functions
# #### 4.	User-Based Recommendations: We provide an example of how to use the recommend_books_with_details2 function to get book recommendations for a specific user.
# #### 5.	Book-Based Recommendations: We provide an example of how to use the recommend_books_item_based function to get book recommendations based on a specific book.
# 

# In[18]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Item-Based Collaborative Filtering
# Calculate item similarity matrix
item_similarity = cosine_similarity(train_data.T)
item_similarity_df = pd.DataFrame(item_similarity, index=train_data.columns, columns=train_data.columns)

def recommend_books_with_details2(user_id, user_item_matrix_df, item_similarity_df, books_sample, k=5):
    user_ratings = user_item_matrix_df.loc[user_id]
    rated_books = user_ratings[user_ratings > 0].index
    recommendations = {}
    for book_id in rated_books:
        similar_books = item_similarity_df[book_id].sort_values(ascending=False)
        for similar_book_id in similar_books.index:
            if user_ratings[similar_book_id] == 0:
                if similar_book_id not in recommendations:
                    recommendations[similar_book_id] = similar_books[similar_book_id]
                else:
                    recommendations[similar_book_id] += similar_books[similar_book_id]
    recommended_books_ids = sorted(recommendations, key=recommendations.get, reverse=True)[:k]
    recommended_books = books_sample[books_sample['book_id'].isin(recommended_books_ids)][['title', 'authors']]
    return recommended_books

# Function to recommend books similar to a given book
def recommend_books_item_based(book_id, item_similarity_df, books_sample, k=5):
    # Find the top k similar books based on item similarity
    similar_books = item_similarity_df[book_id].sort_values(ascending=False).index[1:k+1]
    
    # Get book details for the similar books
    similar_books_details = books_sample[books_sample['book_id'].isin(similar_books)][['title', 'authors']]
    
    # Print the book details of the original book
    original_book_details = books_sample[books_sample['book_id'] == book_id][['title', 'authors']].iloc[0]
    print(f"Original Book: {original_book_details['title']} by {original_book_details['authors']}")
    
    # Print the recommended books
    print("\nRecommended Books:")
    for index, row in similar_books_details.iterrows():
        print(f"{row['title']} by {row['authors']}")
    
    return similar_books_details

# Example usage of the functions

# User-based recommendations
user_id_example = train_data.index[0]
recommended_books_with_details = recommend_books_with_details2(user_id_example, user_item_matrix_df, item_similarity_df, books_sample)
user_rated_books = ratings_sample[ratings_sample['user_id'] == user_id_example].merge(books_sample, on='book_id')
#print(f"Books rated by User {user_id_example}:\n", user_rated_books[['title', 'authors', 'rating']].head())
#print("\nItem-Based Recommended books:\n", recommended_books_item_based)

# Book-based recommendations
book_id_example = 13  # Use a book_id from the ratings sample
recommended_books_based_on_book = recommend_books_item_based(book_id_example, item_similarity_df, books_sample)


# ## Singular Value Decomposition (SVD) for Collaborative Filtering
# 
# ## Main Approach: Singular Value Decomposition (SVD) for Collaborative Filtering
# 
# #### Overview
#     The main approach for our book recommendation system is based on Singular Value Decomposition (SVD) for collaborative filtering. SVD is a matrix factorization technique that decomposes the user-item interaction matrix into lower-dimensional matrices, capturing the latent factors that influence user preferences and item characteristics. This method is effective in providing personalized recommendations by leveraging patterns in the user rating data.
# 
# ### Algorithm Details
# #### 1.	Inputs and Data Preparation:
#     •	Input Data: The primary input for our SVD-based recommendation system is the user-item interaction matrix derived from the ratings data.
#         o    ratings.csv: Contains user ratings for books.
#         o    books.csv: Provides metadata for each book, which can be useful for additional content-based recommendations.
#     •	Data Cleaning: Before applying the SVD algorithm, we clean the data by removing duplicates, handling missing values, and filtering users with insufficient ratings.
#     
# #### 2.	User-Item Interaction Matrix:
#     o	We construct a user-item interaction matrix where rows represent users, columns represent books, and values represent the ratings given by users to books. This matrix serves as the basis for collaborative filtering.
# 
# | User/Book | Book1 | Book2 | Book3 | ... |
# |-----------|-------|-------|-------|-----|
# | User1     |  5    |  0    |  3    | ... |
# | User2     |  4    |  3    |  0    | ... |
# | ...       | ...   | ...   | ...   | ... |
# 
# #### 3.	 Matrix Factorization Using SVD:
#     •	SVD Decomposition: The user-item matrix is decomposed into three matrices: U, Σ, and V^T.
#         o	U(User Matrix): Captures the latent features of users.
#         o	Σ(Diagonal Matrix): Contains the singular values, which represent the importance of each latent feature.
#         o	V^T (Item Matrix): Captures the latent features of items (books).
#     •	Formula: A≈UΣV^T, where A is the original user-item matrix.
#     
# #### 4.Training the SVD Model:
#     •	We use the Surprise library to train the SVD model on the user-item interaction data.
#     •	Training Process:
#         o	Dataset Preparation: Load the ratings data and prepare it for the Surprise library.
#         o	Train-Test Split: Split the data into training and testing sets to evaluate the model's performance.
#         o	Model Training: Fit the SVD model on the training data.
#     •	Evaluation Metrics: We use RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) to evaluate the model's accuracy.
# 
# #### 5. Generating Recommendations:
#     •	For a User:
#         o	Predict ratings for unrated books and recommend the top k books with the highest predicted ratings.
#     •	For a Book:
#         o	Calculate the similarity between the given book and all other books using the item factors (Q matrix) and recommend the top k similar books.
#     •	Functions:
#         o	recommend_books_svd(user_id, svd_model, user_item_matrix_df, books_sample, k): Recommends books for a given user.
#         o	recommend_similar_books_svd(book_id, svd_model, books_sample, k): Recommends books similar to a given book.
# 
# ### Inputs, Outputs, and Variables
# ### •	Inputs:
#         o	ratings.csv: Contains user ratings for books.
#         o	books.csv: Contains metadata about books.
#         o	user_id and book_id: Identifiers for users and books.
#         o	user_item_matrix_df: User-item interaction matrix.
# ### •	Outputs:
#         o	Recommended books for a given user or similar books for a given book.
# ### •	Variables:
#         o	svd_model: Trained SVD model.
#         o	item_factors: Latent factors for items (books).
#         o	similarities: Similarity scores between items.
#         o	recommended_books: DataFrame containing recommended books with their details.

# #### Prepare the Data for the Surprise Library
# #### 1.	Load and Format Data: We load the ratings data into a DataFrame and prepare it for the Surprise library.
#         •	Subset Ratings Data: Select relevant columns (user_id, book_id, rating).
#         •	Reader: Define the rating scale.
#         •	Dataset: Load the DataFrame into the Surprise Dataset format.
#         
# #### 2.	Train-Test Split: We split the data into training and testing sets using an 80-20 split.
#         •	Training Set: Used to train the model.
#         •	Testing Set: Used to evaluate the model's performance.
#         
# #### 3.	Train SVD: We initialize the SVD model and train it on the training set.
#         •	Fit: Train the SVD model using the training set.
# #### 4.	Evaluate Model: We evaluate the performance of the SVD model on the test set using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
#         •	Predict: Generate predictions for the test set.
#         •	Calculate RMSE and MAE: Measure the accuracy of the predictions.
# #### 5.	Recommendation Function: We define a function recommend_similar_books_svd to recommend books similar to a given book based on the trained SVD model.
#         •	Inner ID: Convert the book_id to the internal id used by the Surprise library.
#         •	Item Factors: Retrieve the item factors (Q matrix).
#         •	Calculate Similarities: Compute the cosine similarity between the given book and all other books.
#         •	Top k Similar Books: Identify the top k similar books.
#         •	Get Book Details: Retrieve the titles and authors of the recommended books.
#         •	Print Details: Display the original book and the recommended books.
# #### 6.	User Recommendation Function: We define a function recommend_books_svd to recommend books for a given user based on the trained SVD model.
#         •	Identify Unrated Books: Find books that the user has not rated.
#         •	Generate Predictions: Predict ratings for these unrated books.
#         •	Top k Recommendations: Identify the top k recommended books.
#         •	Get Book Details: Retrieve the titles and authors of the recommended books.
# #### 7.	Example Usage:
#         •	Book-Based Recommendations: Use recommend_similar_books_svd to get book recommendations based on a specific book.
#         •	User-Based Recommendations: Use recommend_books_svd to get book recommendations for a specific user.
# 

# In[17]:


from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split

# Prepare the data for the Surprise library
ratings_data = ratings_sample[['user_id', 'book_id', 'rating']]
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_data, reader)

# Split the data into training and testing sets for Surprise
trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)
# Train the SVD model
svd_model = SVD()
svd_model.fit(trainset)

# Evaluate the SVD model
predictions = svd_model.test(testset)
rmse_svd = accuracy.rmse(predictions)
mae_svd = accuracy.mae(predictions)

# Function to recommend similar books based on SVD
def recommend_similar_books_svd(book_id, svd_model, books_sample, k=5):
    # Get the inner id of the book
    try:
        book_inner_id = svd_model.trainset.to_inner_iid(book_id)
    except ValueError:
        print(f"Book ID {book_id} is not part of the trainset.")
        return pd.DataFrame()
    
    # Get the item vectors (Q matrix in SVD)
    item_factors = svd_model.qi
    # Calculate the similarity with all other books
    similarities = item_factors.dot(item_factors[book_inner_id])
    # Get the top k similar books
    similar_books_inner_ids = similarities.argsort()[::-1][1:k+1]
    similar_books_ids = [svd_model.trainset.to_raw_iid(inner_id) for inner_id in similar_books_inner_ids]
    # Get book details for the similar books
    similar_books_details = books_sample[books_sample['book_id'].isin(similar_books_ids)][['title', 'authors']]
    
    # Print the book details of the original book
    original_book_details = books_sample[books_sample['book_id'] == book_id][['title', 'authors']].iloc[0]
    print(f"Original Book: {original_book_details['title']} by {original_book_details['authors']}")
    
    # Print the recommended books
    print("\nRecommended Books:")
    for index, row in similar_books_details.iterrows():
        print(f"{row['title']} by {row['authors']}")
    
    return similar_books_details

def recommend_books_svd(user_id, svd_model, user_item_matrix_df, books_sample, k=5):
    user_unrated_books = user_item_matrix_df.loc[user_id][user_item_matrix_df.loc[user_id] == 0].index.tolist()
    predictions = [svd_model.predict(user_id, book_id) for book_id in user_unrated_books]
    predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    top_k_predictions = predictions[:k]
    top_k_recommendations = [pred.iid for pred in top_k_predictions]
    recommended_books = books_sample[books_sample['book_id'].isin(top_k_recommendations)][['title', 'authors']]
    return recommended_books

# Example usage of the function
book_id_example = 13  # Replace with the actual book_id you are interested in
recommended_books = recommend_similar_books_svd(book_id_example, svd_model, books_sample)


# In[10]:


# Example usage of the function
book_id_example = 13  # Replace with the actual book_id you are interested in
print("\nMatrix Factorization SVD")
recommended_books = recommend_similar_books_svd(book_id_example, svd_model, books_sample)
print("\nUser-Based Collaborative Filtering")
recommended_books_user_based= recommend_books_user_based(book_id_example, user_item_matrix_df, books_sample)
print("\nItem-Based Collaborative Filtering")
recommended_books_item_based = recommend_books_item_based(book_id_example, item_similarity_df, books_sample)


# In[84]:


import pickle

models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

file_path = os.path.join(models_dir, "svd_model.pkl")

# Save the SVD model to a pickle file
with open(file_path, 'wb') as file:
    pickle.dump(svd_model, file)


# In[85]:

pickle.load(open(file_path, 'rb'))

