Content-Based Movie Recommender 🎬
A full-stack web application that suggests movies based on content similarity. This project leverages Natural Language Processing (NLP) to analyze movie features and recommend titles that are most similar to a user's selection. It demonstrates an end-to-end machine learning workflow, from processing a raw dataset to deploying a functional recommendation engine with an interactive front-end.

The core of the recommender is a content-based filtering model built using TF-IDF and Cosine Similarity on the Netflix Movies and TV Shows dataset from Kaggle.

✨ Features
Dynamic Recommendations: Select any movie from the list and get 10 recommendations based on similarities in genre, director, cast, and plot.

Interactive Front-End: A sleek, Netflix-inspired user interface built with Tailwind CSS that fetches data and recommendations from a live backend.

RESTful API Backend: A Python backend powered by Flask serves the recommendation model, providing endpoints to fetch the movie list and generate recommendations.

Content-Based Filtering: The recommendation logic is based on the content of the movies, not user ratings, making it a robust example of item-to-item recommendation.

🛠️ Tech Stack
Backend: Python, Flask, scikit-learn, Pandas, Joblib

Frontend: HTML, Tailwind CSS, JavaScript

ML Technique: TF-IDF for text vectorization and Cosine Similarity for calculating the likeness between movies.

Dataset: Netflix Movies and TV Shows on Kaggle

🚀 How to Run This Project
Follow these steps to get the application running on your local machine.

1. Clone the Repository
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender

2. Set Up the Python Environment
It's recommended to use a virtual environment.

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required libraries
pip install pandas scikit-learn joblib flask flask-cors

3. Build the Recommendation Model
First, you need to process the dataset and create the model files.

Download the netflix_titles.csv file from the Kaggle link above and place it in the project's root directory.

Run the model builder script:

python model_builder.py

This will create two files: cosine_sim_matrix.joblib and movie_titles.pkl.

4. Run the Flask API
Start the backend server which will serve the model.

python app.py

The server will start running on http://127.0.0.1:5000.

5. Launch the Web Application
Open the index.html file in your web browser. The page will automatically connect to your running Flask API, populate the movie list, and allow you to start getting recommendations!
