import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Load datasets

# These are the functions used for both content and case based RS
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    return movies, credits

# Merge the datasets on 'title'
# We have removed budget, average_vote count, poster, original language etc since they were not needed for making recommendations
def preprocess_data(movies, credits):
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    return movies

# Convert the list of dictionaries into a normal list so that looping and other operations can be done
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Apply conversion to 'genres' and 'keywords' columns
def apply_conversions(movies):
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)

    # We will be applying this function to 'cast', thus fetching only top 3 actors
    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L

    movies['cast'] = movies['cast'].apply(convert3)

    # Fetching the director name from the crew column
    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    movies['crew'] = movies['crew'].apply(fetch_director)

    # Fix for 'overview' column: Check if the value is a string before applying split()
    movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

    # Removed the blank spaces between the words
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

    return movies





# THese are the functions used in content based RS


# Combine relevant columns into 'tags'
def create_tags_column(movies):
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x)) 
    new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())  
    return new_df

# Stem the tags column
# Stemming is converting the words to root words like dance, dancing, danced will be converted to dance
def stem_tags(new_df):
    ps = PorterStemmer()

    def stem(text):
        y = [ps.stem(i) for i in text.split()]
        return " ".join(y)

    new_df.loc[:, 'tags'] = new_df['tags'].apply(stem) 
    return new_df

# Get cosine similarity between movie tags
def compute_similarity(new_df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity


# Recommend movies based on similarity
def recommend(similarity, new_df, movie, K=5):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    # x:x[1] indicates that sorting should be done based on the 2nd value, which is the similarity score
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:K+1]

    recommended_movies = []
    for i in movies_list:
        # Return the movies based on the title
        recommended_movies.append(new_df.iloc[i[0]].title)

    return recommended_movies

# Generate mock relevant movies list based on the top K similar movies
def generate_relevant_movies(new_df, similarity, K=5):
    relevant_dict = {}
    for movie in new_df['title']:
        recommended = recommend(similarity, new_df, movie, K)
        relevant_dict[movie] = recommended
    return relevant_dict

# Evaluation metrics

# Mean Average Precision at K
def mean_average_precision_at_k(recommended, relevant, K=5):
    avg_precisions = []
    for rec, rel in zip(recommended, relevant):
        hits = 0
        avg_precision = 0
        for i, movie in enumerate(rec[:K]):
            if movie in rel:
                hits += 1
                avg_precision += hits / (i + 1)
        avg_precisions.append(avg_precision / min(K, len(rel)))
    return np.mean(avg_precisions)

# Normalized Discounted Cumulative Gain at K
def normalized_discounted_cumulative_gain_at_k(recommended, relevant, K=5):
    ndcgs = []
    for rec, rel in zip(recommended, relevant):
        dcg = 0
        idcg = 0
        for i, movie in enumerate(rec[:K]):
            if movie in rel:
                dcg += 1 / np.log2(i + 2)
            if movie in rel[:K]:
                idcg += 1 / np.log2(i + 2)
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(ndcgs)

# Recall at K
def recall_at_k(recommended, relevant, K=5):
    recalls = []
    for rec, rel in zip(recommended, relevant):
        recalls.append(len(set(rec[:K]) & set(rel)) / len(rel))
    return np.mean(recalls)






# THese Functions are used for Case based RS



def compute_set_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def compute_similarity2(movie_1, movie_2, weights):
    similarity = 0
    total_weight = sum(weights.values())
    
    # Compare each textual feature and compute weighted similarity
    similarity += weights['overview'] * compute_set_similarity(set(movie_1['overview']), set(movie_2['overview']))
    similarity += weights['genres'] * compute_set_similarity(set(movie_1['genres']), set(movie_2['genres']))
    similarity += weights['keywords'] * compute_set_similarity(set(movie_1['keywords']), set(movie_2['keywords']))
    similarity += weights['cast'] * compute_set_similarity(set(movie_1['cast']), set(movie_2['cast']))
    similarity += weights['crew'] * compute_set_similarity(set(movie_1['crew']), set(movie_2['crew']))
    
    return similarity / total_weight

def recommend_case_based(movies, selected_movie, weights, top_n=5):
    similarities = []
    
    # Get the selected movie's features
    selected_movie_features = movies[movies['title'] == selected_movie].iloc[0]
    
    # Calculate similarity for each movie
    for _, movie in movies.iterrows():
        if movie['title'] != selected_movie:  # Skip the selected movie itself
            similarity = compute_similarity2(selected_movie_features, movie, weights)
            similarities.append((movie['title'], similarity))
    
    # Sort movies by similarity in descending order
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_n]







# Main FUnction

def main():
    # Load data
    movies, credits = load_data()

    # Preprocess data
    movies = preprocess_data(movies, credits)
    movies = apply_conversions(movies)

    choice = input('Enter your choice A for Content based Rs and B for Case Based Rs: ')

    if choice == 'A':
        # Create tags and stem them
        new_df = create_tags_column(movies)
        new_df = stem_tags(new_df)

        # Compute cosine similarity
        similarity = compute_similarity(new_df)

        # Generate relevant movies based on similarity
        relevant_movies = generate_relevant_movies(new_df, similarity, K=5)

        # Get movie recommendation
        movie = input('Enter Your Movie: ')  # Example movie to recommend similar ones for
        recommendations = recommend(similarity, new_df, movie, K=5)

        # Print recommended movies
        print(f"Movies similar to '{movie}':")
        for i, rec_movie in enumerate(recommendations, 1):
            print(f"{i}. {rec_movie}")

        # Evaluate the recommendations using MAP@K, NDCG@K, and Recall@K
        recommended = [recommendations]
        relevant = [relevant_movies.get(movie, [])]  # Use the relevant movies for this movie

        map_at_k = mean_average_precision_at_k(recommended, relevant, K=5)
        ndcg_at_k = normalized_discounted_cumulative_gain_at_k(recommended, relevant, K=5)
        recall_at_k_value = recall_at_k(recommended, relevant, K=5)

        # Print evaluation metrics
        print(f"MAP@5: {map_at_k}")
        print(f"NDCG@5: {ndcg_at_k}")
        print(f"Recall@5: {recall_at_k_value}")

    else:
        selected_movie = input("Enter a movie name: ")
        weights = {
            'overview': 0.8,    
            'genres': 0.9,      
            'keywords': 1.2,    
            'cast': 1.3,        
            'crew': 0.7         
        }

        # Get top 5 recommendations
        recommendations = recommend_case_based(movies, selected_movie, weights, top_n=5)

        # Print recommendations
        if recommendations:
            print("Recommended Movies:")
            for i, (title, similarity) in enumerate(recommendations, 1):
                print(f"{i}. {title} (Similarity: {similarity:.2f})")
        else:
            print(f"No recommendations found for the movie: {selected_movie}")

# Run the main function
if __name__ == "__main__":
    main()
