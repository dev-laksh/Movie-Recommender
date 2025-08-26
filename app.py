# --- 1. Import Libraries ---
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import pandas as pd

print("Libraries imported.")

# --- 2. Create Flask App ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

print("Flask app created.")

# --- 3. Load Saved Model Files ---
try:
    # Load the pre-calculated cosine similarity matrix
    cosine_sim = joblib.load('cosine_sim_matrix.joblib')
    
    # Load the movie titles DataFrame
    titles_df = pd.read_pickle('movie_titles.pkl')
    
    # Create a mapping from movie title to index number
    indices = pd.Series(titles_df.index, index=titles_df['title'])
    
    print("Model files loaded successfully.")
except FileNotFoundError:
    print("Error: Model files ('cosine_sim_matrix.joblib' or 'movie_titles.pkl') not found.")
    print("Please run the model_builder.py script first to create these files.")
    exit()

# --- 4. Create API Endpoints ---

@app.route('/movies', methods=['GET'])
def get_movies():
    """
    Endpoint to get the list of all movie titles.
    This will be used to populate the dropdown on the website.
    """
    movie_list = titles_df['title'].tolist()
    return jsonify(movie_list)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    Endpoint to get movie recommendations based on a selected movie.
    """
    # Get the movie title from the JSON data sent in the request
    data = request.get_json()
    title = data.get('title')

    if not title or title not in indices:
        return jsonify({'error': 'Movie title not found in the database.'}), 404

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies (index 0 is the movie itself)
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 10 most similar movies
    recommended_titles = titles_df['title'].iloc[movie_indices].tolist()
    
    return jsonify(recommended_titles)


# --- 5. Run The App ---
if __name__ == '__main__':
    # Running on 0.0.0.0 makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
