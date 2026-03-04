import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os

def hybrid_recommendations(user_id, ratings_df, items_df, top_n=5):
    # Ενσωματωμένα είδη (genre vector)
    genre_cols = items_df.columns[5:24]
    items_df['genre_vector'] = items_df[genre_cols].values.tolist()
    item_features = np.stack(items_df['genre_vector'].values)

    # Ταινίες που έχει ήδη βαθμολογήσει ο χρήστης
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    rated_item_ids = user_ratings['item_id'].values
    if len(rated_item_ids) == 0:
        return ["Ο χρήστης δεν έχει κάνει ακόμα αξιολογήσεις."]

    # Προφίλ χρήστη
    rated_vectors = item_features[[item_id - 1 for item_id in rated_item_ids]]
    user_profile = np.mean(rated_vectors, axis=0).reshape(1, -1)

    # Content-based σκορ
    content_scores = cosine_similarity(user_profile, item_features).flatten()
    content_scores = MinMaxScaler().fit_transform(content_scores.reshape(-1, 1)).flatten()

    # Φόρτωση SVD
    if not os.path.exists('svd_model.pkl'):
        return ["Δεν έχει εκπαιδευτεί ακόμα το SVD μοντέλο."]
    with open('svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)

    svd_scores = np.zeros(len(items_df))
    for i, item_id in enumerate(items_df['item_id']):
        try:
            svd_scores[i] = svd_model.predict(user_id, item_id).est
        except:
            svd_scores[i] = 0.0
    svd_scores = MinMaxScaler().fit_transform(svd_scores.reshape(-1, 1)).flatten()

    hybrid_scores = 0.5 * content_scores + 0.5 * svd_scores

    # Αφαίρεση ήδη βαθμολογημένων
    recommendations = [
        (item_id, score) for item_id, score in zip(items_df['item_id'], hybrid_scores)
        if item_id not in rated_item_ids
    ]
    recommendations.sort(key=lambda x: x[1], reverse=True)

    top_items = [
        items_df[items_df['item_id'] == item_id]['title'].values[0]
        for item_id, _ in recommendations[:top_n]
    ]
    return top_items
