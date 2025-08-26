# --- 1. Import Libraries ---
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

print("Libraries imported.")

# --- 2. Load and Prepare the Data ---
try:
    # Load the dataset from a CSV file
    df = pd.read_csv('netflix_titles.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'netflix_titles.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the same folder as this script.")
    exit()

# For this recommender, we'll only focus on movies
df = df[df['type'] == 'Movie'].copy()

# Fill missing values (NaNs) in key columns with empty strings
features_to_fill = ['director', 'cast', 'listed_in', 'description']
for feature in features_to_fill:
    df[feature] = df[feature].fillna('')

print("Data loaded and cleaned. Only using 'Movie' types.")

# --- 3. Feature Engineering: Create the "Content Soup" ---

def clean_text(text):
    """A simple function to clean text data by removing spaces."""
    if isinstance(text, list):
        return [str(i).replace(" ", "") for i in text]
    else:
        return str(text).replace(" ", "")

# Clean director and cast names to treat them as single entities
df['director'] = df['director'].apply(clean_text)
df['cast'] = df['cast'].apply(lambda x: ' '.join([clean_text(i) for i in x.split(', ')[:3]])) # Top 3 actors

# Combine all the features into a single string for each movie
def create_soup(x):
    return x['director'] + ' ' + x['cast'] + ' ' + x['listed_in'] + ' ' + x['description']

df['soup'] = df.apply(create_soup, axis=1)

print("Created 'content soup' for each movie.")

# --- 4. Build the Recommendation Model ---

# Initialize the TF-IDF Vectorizer
# stop_words='english' removes common English words that don't add much meaning
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the 'soup' column to create the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(df['soup'])

# Calculate the cosine similarity matrix from the TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("TF-IDF matrix and Cosine Similarity matrix created.")

# --- 5. Save the Model and Data ---
# We need to save both the similarity matrix and the movie data (for titles)

# Save the cosine similarity matrix
joblib.dump(cosine_sim, 'cosine_sim_matrix.joblib')

# Save the DataFrame (or at least the movie titles)
# We reset the index to easily find movies by their index number later
df = df.reset_index()
df[['title']].to_pickle('movie_titles.pkl')


print("\n--- Model Building Complete ---")
print("Saved 'cosine_sim_matrix.joblib' and 'movie_titles.pkl'.")
print("These files are now ready for the Flask API.")

