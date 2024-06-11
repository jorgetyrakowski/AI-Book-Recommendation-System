# AI-Book-Recommendation-System
AI Course Final Project 

##Installation
To get the Hybrid Book Recommendation System up and running on your local machine, follow these step-by-step instructions:

#### Step 1: Clone the Repository
First, clone the project repository to your local machine using Git. Open your terminal and run the 
following command:
git clone https://github.com/your-username/AI-Book-Recommendation-System.git

Navigate into the project directory:
cd AI-Book-Recommendation-System

#### Step 2: Install Required Packages
Install all the necessary packages listed in requirements.txt. Ensure you are in the project directory and run:
pip install -r requirements.txt

This command will install all the dependencies needed for the project, such as Flask, NumPy, Pandas, Scikit-Learn, and NLTK.

## Setting Up the Models
To ensure the Hybrid Book Recommendation System operates effectively, you will need to set up both the collaborative filtering and content-based models. Follow these instructions to generate and store the necessary models in the appropriate directory.

#### Step 1: Generate the Collaborative Filtering Model
Run the collabfilt.py script to create the collaborative filtering model. This script will process the user-book interaction data and utilize the SVD algorithm to generate a model that predicts user preferences based on similar user behaviors. 

Here’s how to execute the script:
python collabfilt.py

This will create a svd_model.pkl file inside the model directory, which contains the trained collaborative filtering model.

#### Step 2: Generate the Content-Based Filtering Model
Next, execute the contentbase.py script to build the content-based filtering model. This model uses TF-IDF (Term Frequency-Inverse Document Frequency) to analyze book descriptions and recommend books that are textually similar to those a user has liked in the past. 

## Run the following command:
python contentbase.py

This command generates a tfid.pkl file within the model directory, storing the trained content-based model.

## Running the Application
Once the models are set up and all dependencies are installed, you can launch the Hybrid Book Recommendation System to start receiving book recommendations. Follow these steps to run the application:

#### Step 1: Start the Application
To run the web application, use Streamlit, which provides an intuitive web interface. Open your terminal, navigate to the project directory, and execute the following command:

streamlit run app.py
This command starts the server and opens the application in your default web browser.

#### Step 2: Using the Web Interface
Upon launching the application, you will be directed to the main page where you can interact with the Hybrid Book Recommendation System. The interface will allow you to:

Select a Book: Browse through a list of books or search for a specific book you like.
Get Recommendations: After selecting a book, the system will use both the collaborative and content-based models to suggest other books you might enjoy.

![image](https://github.com/jorgetyrakowski/AI-Book-Recommendation-System/assets/88347278/620ac8b3-083d-45d5-a570-4bfdc7d47b68)

![image](https://github.com/jorgetyrakowski/AI-Book-Recommendation-System/assets/88347278/e8b0256c-a6d0-4a39-aefc-86274ef255f5)

![image](https://github.com/jorgetyrakowski/AI-Book-Recommendation-System/assets/88347278/e4d69562-02d6-4f5f-b160-a86967a08e20)


