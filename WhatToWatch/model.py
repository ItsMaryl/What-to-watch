import pickle
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

def train_svd_model(ratings_df, model_path='svd_model.pkl'):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def evaluate_svd_model(ratings_df, model_path='svd_model.pkl'):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD()
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return rmse
