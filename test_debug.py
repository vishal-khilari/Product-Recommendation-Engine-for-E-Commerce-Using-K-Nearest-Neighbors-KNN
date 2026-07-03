import pandas as pd
import pickle
import numpy as np
from scipy import sparse

print("Loading data...")
df = pd.read_csv('amazon_products_subset.csv')
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
tfidf_matrix = sparse.load_npz('tfidf_matrix.npz')

def debug_query(item_query):
    print(f"\n--- Query: {item_query} ---")
    query_vec = vectorizer.transform([item_query.lower()])
    tfidf_sims = tfidf_matrix.dot(query_vec.T).toarray().flatten()
    
    match_mask = tfidf_sims > 0.05
    if match_mask.any():
        candidates = df[match_mask].copy()
        candidates['tfidf_sim'] = tfidf_sims[match_mask]
        
        # Category Mass Voting
        cat_mass = candidates.groupby('category_name')['svd_norm'].sum()
        winning_category = cat_mass.idxmax()
        print(f"Winning category: {winning_category}")
        
        candidates['anchor_score'] = candidates['tfidf_sim'] + (candidates['svd_norm'] * 0.1)
        
        # Sort by anchor score in winning category
        cat_cands = candidates[candidates['category_name'] == winning_category].sort_values('anchor_score', ascending=False)
        for _, row in cat_cands.head(3).iterrows():
            print(f"[{row['asin']}] {row['title'][:60]} | TFIDF: {row['tfidf_sim']:.4f} | SVD: {row['svd_norm']:.4f} | Anchor: {row['anchor_score']:.4f}")

debug_query("Home First Aid Kit")
debug_query("Burn Care Kit")
