#!/usr/bin/env python
# coding: utf-8

# **Data Cleaning and Transformation**
# 
# Remove Duplicates: If there are any duplicate rows, remove them.
# 
# Handle Missing Values: Handle any missing values in columns like category and about_product that may be useful for content-based filtering.
# 
# Standardize Categories: Normalize text fields (convert to lowercase, remove extra spaces, etc.).

# In[34]:

import sys
sys.stdout.reconfigure(encoding='utf-8')


import pandas as pd
import numpy as np

# Load the dataset
products = pd.read_csv("amazon_products.csv", nrows=10000)
categories = pd.read_csv("amazon_categories.csv")

# Merge
data = products.merge(categories, left_on='category_id', right_on='id', how='left')

# Synthesize dummy user_id for collaborative filtering
np.random.seed(42)
data['user_id'] = ['user_' + str(i) for i in np.random.randint(1, 100, size=len(data))]

print("Data Shape:", data.shape)
print(data.columns)

# Cleaning
data = data.drop_duplicates()
data.fillna('', inplace=True)

data['category_name'] = data['category_name'].astype(str).str.lower().str.strip()
data['title'] = data['title'].astype(str).str.lower().str.strip()
data['combined_features'] = data['category_name'] + " " + data['title']

print(data.head())


# **Collaborative Filtering**
# 
# We will now create a User-Item Interaction Matrix based on implicit feedback (browsing history or purchase interactions). Since the dataset doesn't have explicit ratings, we will treat interactions as binary values (i.e., user has interacted with the product or not).

# In[35]:


interaction_matrix = pd.pivot_table(data, index='user_id', columns='asin', aggfunc='count', fill_value=0)
print(interaction_matrix.head())


# **Singular Value Decomposition (SVD) for Collaborative Filtering**
# 
# We'll apply Singular Value Decomposition (SVD) to factorize the interaction matrix and use it to generate user-product
# recommendations based on similarity.

# In[36]:


from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Apply SVD for dimensionality reduction
svd = TruncatedSVD(n_components=20, random_state=42)
svd_matrix = svd.fit_transform(interaction_matrix)

# Reconstruct the matrix with the reduced dimensions
reconstructed_matrix = np.dot(svd_matrix, svd.components_)

# Calculate similarity between users
user_similarity = cosine_similarity(reconstructed_matrix)


print(user_similarity[:5])


# **Generate Collaborative Filtering Recommendations**
# 
# Using the similarity between users, we will recommend products to a user based on what similar users have interacted with.

# In[37]:


def get_collaborative_recommendations(user_id, top_n=5):
    # Get the index of the user in the interaction matrix
    user_idx = interaction_matrix.index.get_loc(user_id)
    similarity_scores = user_similarity[user_idx]
    similar_user_indices = similarity_scores.argsort()[-(top_n + 1):-1][::-1]

    recommended_products = set()

    for idx in similar_user_indices:
        similar_user = interaction_matrix.index[idx]
        similar_user_interactions = interaction_matrix.loc[similar_user]

        # Recommend products that similar users interacted
        recommended_products.update(similar_user_interactions[similar_user_interactions > 0].index)

    return list(recommended_products)[:top_n]



# **Content-Based Filtering**
# 
# Now, let’s implement Content-Based Filtering using TF-IDF vectorization. We will use the combined_features column (which includes both category and about_product) to compute the similarity between products.

# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])
content_similarity_matrix = cosine_similarity(tfidf_matrix)


print(content_similarity_matrix[0])


# **Hybrid Recommendation System**
# 
# Now that we have both collaborative and content-based recommendations, we can combine the two methods to form a Hybrid Recommendation System. The idea is to take the union of both sets of recommendations or give weights to each. bold text

# In[39]:


def get_content_based_recommendations(product_id, top_n=5):
    product_idx = data[data['asin'] == product_id].index[0]
    similarities = content_similarity_matrix[product_idx]
    similar_product_idx = similarities.argsort()[-(top_n + 1):-1][::-1]

    recommended_product_ids = data.iloc[similar_product_idx]['asin'].tolist()

    return recommended_product_ids

# Example:
product_id = "B014TMV5YE"
content_based_recommendations = get_content_based_recommendations(product_id)
print(f"Content-based recommendations for product {product_id}: {content_based_recommendations}")


# **Real-Time Recommendations**
# 
# For real-time updates, you can simulate real-time browsing or interaction and immediately generate recommendations based on the recent activity.

# In[40]:


def get_hybrid_recommendations(user_id, product_id, top_n=5, alpha=0.6):
    # collaborative filtering
    collaborative_recs = get_collaborative_recommendations(user_id, top_n=top_n)

    # content-based filtering
    content_recs = get_content_based_recommendations(product_id, top_n=top_n)

    # Combine recommendations: a weighted average of both
    combined_recs = set(collaborative_recs) | set(content_recs)

    # Return top N unique recommendations
    return list(combined_recs)[:top_n]


# **Evaluation**
# 
# We can evaluate the recommendation system using Precision, Recall, or Mean Average Precision (MAP) by comparing the generated recommendations with actual user preferences.

# In[41]:


recently_browsed_product = "B014TMV5YE"
recent_user_id = "user_1"  # Example user who viewed the product

# Function to get product names for a list of product_ids
def get_product_names(product_ids):
    # Assuming 'data' is the original dataset loaded with product details
    product_names = data[data['asin'].isin(product_ids)]['title'].tolist()
    return product_names

# real-time recommendations based on the recent activity (product IDs)
real_time_recommendations_ids = get_hybrid_recommendations(user_id=recent_user_id, product_id=recently_browsed_product, top_n=5)

# Get product names for the recommended product IDs
real_time_recommendations_names = get_product_names(real_time_recommendations_ids)

print(f"Real-time recommendations for user {recent_user_id} after viewing product {recently_browsed_product}: {real_time_recommendations_names}")


# In[42]:


# ============================================================
# FINAL VERSION: Tiered Recommendation System
# With cross-category diversity for Low tier
# ============================================================

def get_tiered_recommendations(product_id, top_n=5):
    matches = data[data['asin'] == product_id]
    if matches.empty:
        print(f"❌ Product ID '{product_id}' not found.")
        return

    product_idx = matches.index[0]
    product_name = data.loc[product_idx, 'title']
    selected_category = data.loc[product_idx, 'category_name'].split("|")[0].strip()

    similarities = content_similarity_matrix[product_idx]
    similar_indices = similarities.argsort()[::-1]

    high, medium, low = [], [], []
    seen_names = set()

    for idx in similar_indices:
        if idx == product_idx:
            continue

        score = similarities[idx]
        name = data.loc[idx, 'title']
        raw_category = data.loc[idx, 'category_name']
        top_category = raw_category.split("|")[0].strip()
        display_category = raw_category.split("|")[-1].strip()[:35]

        short_name = name[:35].lower()
        if short_name in seen_names:
            continue
        seen_names.add(short_name)

        if score == 1.0 and name[:30].lower() == product_name[:30].lower():
            continue

        entry = {
            "Product": name[:68],
            "Category": display_category,
            "Score": round(score, 3)
        }

        # Force cross-category items into Low tier regardless of score
        if top_category != selected_category:
            if len(low) < top_n:
                low.append(entry)
        elif score >= 0.50:
            if len(high) < top_n:
                high.append(entry)
        elif score >= 0.15:
            if len(medium) < top_n:
                medium.append(entry)

        if len(high) >= top_n and len(medium) >= top_n and len(low) >= top_n:
            break

    # ── Display ──────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  🛒  Selected Product:")
    print(f"      {product_name}")
    print(f"  📂  Category: {selected_category}")
    print(f"{'='*75}")

    print(f"\n🟢  HIGH Recommendation — Strongly Related / Frequently Bought Together")
    print(f"    (Same category, similarity score ≥ 0.50)")
    print(f"    {'─'*68}")
    if high:
        for i, item in enumerate(high, 1):
            bar = "█" * int(item['Score'] * 20)
            print(f"    {i}. {item['Product']}")
            print(f"       Score: {item['Score']}  {bar}")
    else:
        print("    (None found)")

    print(f"\n🟡  MEDIUM Recommendation — Moderately Related / Useful Add-ons")
    print(f"    (Same category, similarity score 0.15–0.49)")
    print(f"    {'─'*68}")
    if medium:
        for i, item in enumerate(medium, 1):
            bar = "█" * int(item['Score'] * 20)
            print(f"    {i}. {item['Product']}")
            print(f"       Score: {item['Score']}  {bar}")
    else:
        print("    (None found)")

    print(f"\n🔴  LOW Recommendation — Loosely Related / General Items")
    print(f"    (Different top-level category)")
    print(f"    {'─'*68}")
    if low:
        for i, item in enumerate(low, 1):
            bar = "█" * int(item['Score'] * 20)
            print(f"    {i}. {item['Product']}")
            print(f"       Category: {item['Category']}  |  Score: {item['Score']}  {bar}")
    else:
        print("    (None found — all products in dataset share the same top category)")

    print(f"\n  📊 Summary: {len(high)} High  |  {len(medium)} Medium  |  {len(low)} Low")
    print(f"{'='*75}\n")


# ── Run it ────────────────────────────────────────────────────────────────
get_tiered_recommendations("B014TMV5YE")


# In[43]:


# Test 3 different products back to back
for pid in ["B07GDLCQXV", "B07XSCCZYG", "B08MVFKGJM"]:
    get_tiered_recommendations(pid)


# In[44]:


import pandas as pd

def get_recommendation_table(product_id, top_n=5):
    matches = data[data['asin'] == product_id]
    if matches.empty:
        print(f"❌ Product ID '{product_id}' not found.")
        return

    product_idx = matches.index[0]
    product_name = data.loc[product_idx, 'title']
    selected_category = data.loc[product_idx, 'category_name'].split("|")[0].strip()

    similarities = content_similarity_matrix[product_idx]
    similar_indices = similarities.argsort()[::-1]

    rows = []
    seen_names = set()

    for idx in similar_indices:
        if idx == product_idx:
            continue

        score = similarities[idx]
        name = data.loc[idx, 'title']
        raw_category = data.loc[idx, 'category_name']
        top_category = raw_category.split("|")[0].strip()
        leaf_category = raw_category.split("|")[-1].strip()

        short_name = name[:35].lower()
        if short_name in seen_names:
            continue
        seen_names.add(short_name)

        if score == 1.0 and name[:30].lower() == product_name[:30].lower():
            continue

        # Assign tier
        if top_category != selected_category:
            tier = "🔴 Low"
            tier_reason = "Different category"
        elif score >= 0.50:
            tier = "🟢 High"
            tier_reason = "Strongly related"
        elif score >= 0.15:
            tier = "🟡 Medium"
            tier_reason = "Moderately related"
        else:
            continue

        rows.append({
            "Tier": tier,
            "Product Name": name[:55] + "..." if len(name) > 55 else name,
            "Category": leaf_category[:25],
            "Score": round(score, 3),
            "Reason": tier_reason
        })

        high_count = sum(1 for r in rows if r["Tier"] == "🟢 High")
        med_count  = sum(1 for r in rows if r["Tier"] == "🟡 Medium")
        low_count  = sum(1 for r in rows if r["Tier"] == "🔴 Low")
        if high_count >= top_n and med_count >= top_n and low_count >= top_n:
            break

    print(f"\n🛒 Selected: {product_name[:80]}")
    print(f"📂 Category: {selected_category}\n")

    df_result = pd.DataFrame(rows)
    df_result = df_result.sort_values(
        by="Tier",
        key=lambda x: x.map({"🟢 High": 0, "🟡 Medium": 1, "🔴 Low": 2})
    ).reset_index(drop=True)
    df_result.index += 1

    return df_result

# Run it — output is a clean DataFrame table
result_df = get_recommendation_table("B014TMV5YE")
result_df


# In[45]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_recommendation_chart(product_id, top_n=5):
    matches = data[data['asin'] == product_id]
    if matches.empty:
        print("Product not found.")
        return

    product_idx = matches.index[0]
    product_name = data.loc[product_idx, 'title'][:60]
    selected_category = data.loc[product_idx, 'category_name'].split("|")[0].strip()

    similarities = content_similarity_matrix[product_idx]
    similar_indices = similarities.argsort()[::-1]

    names, scores, colors, tiers = [], [], [], []
    seen_names = set()

    color_map = {"🟢 High": "#2ecc71", "🟡 Medium": "#f39c12", "🔴 Low": "#e74c3c"}

    for idx in similar_indices:
        if idx == product_idx:
            continue

        score = similarities[idx]
        name = data.loc[idx, 'title']
        top_category = data.loc[idx, 'category_name'].split("|")[0].strip()

        short_name = name[:35].lower()
        if short_name in seen_names:
            continue
        seen_names.add(short_name)

        if score == 1.0 and name[:30].lower() == product_name[:30].lower():
            continue

        if top_category != selected_category:
            tier = "🔴 Low"
        elif score >= 0.50:
            tier = "🟢 High"
        elif score >= 0.15:
            tier = "🟡 Medium"
        else:
            continue

        high_c  = tiers.count("🟢 High")
        med_c   = tiers.count("🟡 Medium")
        low_c   = tiers.count("🔴 Low")

        if tier == "🟢 High"   and high_c >= top_n: continue
        if tier == "🟡 Medium" and med_c  >= top_n: continue
        if tier == "🔴 Low"    and low_c  >= top_n: continue

        label = name[:40] + "..." if len(name) > 40 else name
        names.append(label)
        scores.append(round(score, 3))
        colors.append(color_map[tier])
        tiers.append(tier)

        if high_c + (1 if tier=="🟢 High" else 0) >= top_n and \
           med_c  + (1 if tier=="🟡 Medium" else 0) >= top_n and \
           low_c  + (1 if tier=="🔴 Low" else 0) >= top_n:
            break

    # Sort: High → Medium → Low
    order = {"🟢 High": 0, "🟡 Medium": 1, "🔴 Low": 2}
    combined = sorted(zip(tiers, names, scores, colors), key=lambda x: order[x[0]])
    tiers_s, names_s, scores_s, colors_s = zip(*combined)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(names_s)), scores_s, color=colors_s,
                   edgecolor='white', linewidth=0.8, height=0.65)

    # Score labels on bars
    for bar, score in zip(bars, scores_s):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score}", va='center', fontsize=9, color='#333333')

    # Threshold lines
    ax.axvline(x=0.50, color='#2ecc71', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axvline(x=0.15, color='#f39c12', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(0.50, len(names_s) - 0.3, ' High ≥ 0.50', color='#2ecc71', fontsize=8)
    ax.text(0.15, len(names_s) - 0.3, ' Medium ≥ 0.15', color='#f39c12', fontsize=8)

    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cosine Similarity Score", fontsize=11)
    ax.set_xlim(0, 1.12)
    ax.set_title(f"Product Recommendation Tiers\n🛒 {product_name}",
                 fontsize=13, fontweight='bold', pad=15)

    # Legend
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in color_map.items()]
    ax.legend(handles=patches, loc='lower right', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("recommendation_chart.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Chart saved as recommendation_chart.png")

# Run it
plot_recommendation_chart("B014TMV5YE")


# In[46]:


def evaluate_precision(product_id, relevant_category_keyword, top_n=5):
    """
    Precision = how many of the recommended products are actually relevant.
    We define 'relevant' as: product name or category contains the keyword.
    """
    df_result = get_recommendation_table(product_id, top_n=top_n)
    if df_result is None or df_result.empty:
        return

    high_recs = df_result[df_result["Tier"] == "🟢 High"]["Product Name"].tolist()

    relevant = [p for p in high_recs
                if relevant_category_keyword.lower() in p.lower()]

    precision = len(relevant) / len(high_recs) if high_recs else 0

    print(f"\n📐 Evaluation — Precision@{top_n}")
    print(f"   Keyword checked : '{relevant_category_keyword}'")
    print(f"   High-tier recs  : {len(high_recs)}")
    print(f"   Relevant matches: {len(relevant)}")
    print(f"   Precision Score : {precision:.2f} ({precision*100:.0f}%)")

    if precision >= 0.8:
        print("   ✅ Excellent precision!")
    elif precision >= 0.5:
        print("   🟡 Moderate precision.")
    else:
        print("   🔴 Low precision — consider tuning thresholds.")

# Evaluate: for a Luggage product, how many High recs are also Luggage?
evaluate_precision("B014TMV5YE", relevant_category_keyword="Luggage", top_n=5)

