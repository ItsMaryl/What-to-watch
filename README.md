# What-to-watch
Intelligent Movie Recommendation System combining collaborative filtering (SVD) and content-based modeling (Python & Streamlit)
# 🎬 Hybrid Movie Recommendation System

An end-to-end intelligent movie recommendation web application built with Python and Streamlit.  
The system combines collaborative filtering (SVD) with content-based filtering to generate personalized movie suggestions.

---

## 🚀 Project Overview

This project implements a hybrid recommendation engine using the MovieLens dataset.  
Users can select a user ID and receive personalized movie recommendations based on:

- Collaborative filtering (SVD)
- Content-based filtering (genre similarity)
- Hybrid score combination

The model performance is evaluated using RMSE.

---

## 🧠 Technical Architecture

- **Frontend:** Streamlit
- **Backend:** Python
- **ML Model:** SVD (Surprise library)
- **Content-Based Filtering:** Cosine similarity
- **Evaluation Metric:** RMSE
- **Dataset:** MovieLens 100K

---

## ⚙️ How It Works

1. Load MovieLens dataset
2. Train SVD collaborative filtering model
3. Build user content profile based on rated movie genres
4. Compute cosine similarity between user profile and movies
5. Combine SVD and content-based scores (hybrid approach)
6. Recommend top-N unseen movies

Hybrid Score Formula:Hybrid Score = 0.5 * Content Score + 0.5 * SVD Score


---

## 📊 Model Evaluation

The system evaluates performance using Root Mean Squared Error (RMSE) on a 80/20 train-test split.

---

## 🖥️ Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
---
📂 Project Structure
.
├── app.py
├── model.py
├── recommend.py
├── data/
│   ├── u.data
│   └── u.item
├── requirements.txt
└── README.md
