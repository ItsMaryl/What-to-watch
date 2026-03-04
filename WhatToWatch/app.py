import streamlit as st
import pandas as pd
from model import train_svd_model, evaluate_svd_model
from recommend import hybrid_recommendations
str
# Δεδομένα
ratings = pd.read_csv('data/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
items = pd.read_csv(
    'data/u.item',
    sep='|',
    encoding='latin-1',
    names=[
        'item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
)

st.title("🎬 What To Watch Tonight")

# Custom background and styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }

    .stApp {
        background-image: url("https://cdn.financebuzz.com/filters:quality(75)/images/2024/01/01/private_screening_room.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }

    h1 {
        color: #ffe347;
        text-shadow: 2px 2px 4px #000000;
    }

    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }

    .stSelectbox label {
        color: white;
    }

    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.9);
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


user_id = st.selectbox("Επίλεξε Χρήστη", sorted(ratings['user_id'].unique()))

if st.button("Εκπαίδευση SVD Μοντέλου"):
    train_svd_model(ratings)
    st.success("✅ Το μοντέλο SVD εκπαιδεύτηκε και αποθηκεύτηκε.")

if st.button("Προτάσεις για Χρήστη"):
    results = hybrid_recommendations(user_id, ratings, items)
    if isinstance(results[0], str):
        st.warning(results[0])
    else:
        st.subheader(f"Προτάσεις για Χρήστη {user_id}")
        for title in results:
            st.markdown(f"🎥 **{title}**")

if st.button("Υπολογισμός RMSE"):
    rmse = evaluate_svd_model(ratings)
    st.info(f"📊 RMSE: **{rmse:.4f}**")
