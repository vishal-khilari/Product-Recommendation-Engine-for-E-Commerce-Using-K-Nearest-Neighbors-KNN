# E-Commerce Product Recommendation System using K-Neighbor Nearest Algorithm

A high-performance, real-time hybrid recommendation system built using Python, Flask, and scikit-learn, powered by an optimized **K-Nearest Neighbors (KNN)** architecture. The system processes extensive e-commerce metadata (1.4 million products) and calculates product similarities using **Cosine Distance** on combined text-behavioral vectors, returning dynamic, tiered recommendations. Additionally, it integrates a generative AI project planner powered by Google Gemini.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Algorithmic Architecture (KNN Paradigm)](#-algorithmic-architecture-knn-paradigm)
3. [Generative AI Features](#-generative-ai-features)
4. [Key Features](#-key-features)
5. [Project Structure](#-project-structure)
6. [Tech Stack & Dependencies](#-tech-stack--dependencies)
7. [Installation & Setup](#-installation--setup)
8. [How to Run the Project](#-how-to-run-the-project)
9. [API Endpoints Reference](#-api-endpoints-reference)
10. [License & Disclaimer](#-license--disclaimer)

---

## 🌟 Project Overview
In modern e-commerce platforms, delivering accurate and contextually relevant product suggestions is critical to user engagement and cross-selling. This system integrates content features (textual title and category names) with collaborative/behavioral characteristics (synthetic user interaction and item popularity matrices) into a single hybrid recommendation engine. 

The recommendation core is built around the **K-Nearest Neighbors (KNN)** retrieval paradigm, calculating spatial proximity across high-dimensional TF-IDF and SVD spaces. The results are classified and partitioned into three distinct consumption tiers:
*   **High Tier (Immediate Need / Substitutes)**
*   **Medium Tier (Useful Add-ons / Ecosystem)**
*   **Low Tier (General / Cross-Sell)**

---

## 📐 Algorithmic Architecture (KNN Paradigm)

Traditional KNN models use Euclidean distance, which is inefficient and inaccurate in high-dimensional text token spaces. This system implements an optimized, unsupervised KNN search using **Cosine Similarity** on multi-dimensional coordinate spaces.

```
Anchor Product ──> Vectorization (TF-IDF & SVD) ──> Cosine Similarity Lookup ──> K-Nearest Sorting ──> Rule-Based Classification ──> Tiered Output
```

### 1. Spatial Coordinate Mapping (Feature Vectors)
Every product is represented as a coordinates vector in a combined mathematical space:
*   **Textual Representation (TF-IDF):** Product titles and category strings are tokenized and vectorized using Term Frequency-Inverse Document Frequency (TF-IDF). This captures the language signature of each item.
*   **Behavioral Representation (SVD):** Latent vectors are generated using Singular Value Decomposition (SVD) to represent behavioral user interactions, categorizing products into root category clusters and popularity weights.

### 2. Similarity Metric (Cosine Distance)
We use Cosine Similarity to find the angular distance between the active "Anchor" product vector ($\vec{u}$) and all other product candidate vectors ($\vec{v}$):

$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$

Since the coordinates are pre-normalized, the cosine similarity simplifies to a fast, matrix dot-product operation:

$$\text{Score} = \vec{u} \cdot \vec{v}$$

### 3. Finding the K-Nearest Neighbors
When a query or product is chosen:
1. It becomes the **Anchor Point** in the coordinate space.
2. The distance (similarity) is computed from this anchor to all items.
3. The system executes a high-speed sort (`numpy.argsort()[::-1]`) to retrieve the top **$K=500$ Nearest Neighbors**.

### 4. Heuristic Tier Classification
The retrieved nearest neighbors are classified into recommendation tiers based on geometric similarity boundaries and product properties:
*   **Tier 1 (High Relevance):** Neighbor similarity $\ge 0.25$, textual similarity $< 0.3$, and price $\le$ anchor price (offering cost-effective substitute items).
*   **Tier 2 (Medium Relevance):** Neighbors sharing the same top-level root category as the anchor.
*   **Tier 3 (Low / Cross-Sell):** Neighbors in different root categories (outliers) to promote discovery.

---

## 🤖 Generative AI Features

The system integrates a Generative AI layer using the official Google GenAI SDK (`google-genai`):

1.  **AI Project Builder:** Expects a natural language goal from the user (e.g., *"Build a backyard vegetable garden"*). Google Gemini generates a list of 4–8 required item categories, and our local KNN model finds the best-matched products from the Amazon catalog to construct a complete cart.
2.  **Marketing Campaign Copywriter:** Generates targeted marketing materials (Email Blasts, Social Media Posts, or Ad Copies) for any product in the inventory.

---

## ⚡ Key Features
*   **Hybrid Recommendation Model:** Joins Content-Based filtering (TF-IDF) and Collaborative/Behavioral signals (SVD).
*   **Real-time Analytics Dashboard:** Features a luxury dark UI with charts showing similarity distributions and precision rates.
*   **Interactive Category Browser:** Search or select from thousands of categories.
*   **Fast Informational Retrieval:** Returns recommendations in milliseconds.
*   **Responsive UX:** Built using Tailwind CSS, vanilla JavaScript, and modern design principles.

---

## 📂 Project Structure
```text
E-Commerce-Product-Recommendation-System/
│
├── app.py                      # Core Flask API backend & KNN calculations
├── requirements.txt            # Python environment packages
├── .gitignore                  # Prevents committing large files, venv, and keys
├── .env                        # Local variables (e.g. GEMINI_API_KEY)
├── LICENSE                     # Project license file
├── README.md                   # Complete developer manual
├── detail.txt                  # Model implementation summary
│
├── amazon_products.csv         # Main product catalog database (~375MB, excluded from Git)
├── amazon_categories.csv       # Categories structure database
│
├── frontend/                   # Client application folder
│   ├── index.html              # Modern Dashboard UI layout
│   ├── css/
│   │   └── style.css           # Custom styles & micro-animations
│   └── js/
│       └── script.js           # Client-side routing, API fetching, PapaParse
```

---

## 🛠️ Tech Stack & Dependencies
*   **Backend:** Python 3.8+, Flask, Flask-CORS
*   **Machine Learning:** NumPy, Pandas, scikit-learn (TfidfVectorizer, SVD)
*   **AI Integration:** Google GenAI SDK (`google-genai`), python-dotenv
*   **Frontend:** HTML5, Tailwind CSS (via CDN), Vanilla JavaScript, PapaParse

---

## 💾 Installation & Setup

### Prerequisites
*   Python 3.8 or higher installed on your machine.
*   A Google Gemini API key (obtainable from [Google AI Studio](https://aistudio.google.com/)).

### Step 1: Clone and Navigate
Navigate to the project root directory in your command terminal.

### Step 2: Set Up Virtual Environment
Create and activate a virtual environment to manage project packages:
```bash
# Create
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 3: Install Required Packages
Install all required libraries specified in [requirements.txt](file:///c:/Users/Vishal%20P%20Khilari/Desktop/E-Commerce-Product-Recommendation-System%20-%20Copy/requirements.txt):
```bash
pip install -r requirements.txt
```

### Step 4: Configuration File
Create a `.env` file in the root folder (or inspect the existing [.env](file:///c:/Users/Vishal%20P%20Khilari/Desktop/E-Commerce-Product-Recommendation-System%20-%20Copy/.env) file) and insert your Gemini API Key:
```env
GEMINI_API_KEY=your_google_gemini_api_key
```

---

## 🚀 How to Run the Project

### 1. Launch the Backend API
Start the Flask application server from your active terminal:
```bash
python app.py
```
*   The backend will parse dataset metadata, build the mathematical vector matrices, and initialize the similarity search index.
*   Once initialized, the message **`Initialization complete! API is ready.`** will appear, and the server will host at `http://127.0.0.1:5000`. Keep this window open.

### 2. Run the Frontend Client
*   Open the [frontend](file:///c:/Users/Vishal%20P%20Khilari/Desktop/E-Commerce-Product-Recommendation-System%20-%20Copy/frontend) directory.
*   Simply open the [index.html](file:///c:/Users/Vishal%20P%20Khilari/Desktop/E-Commerce-Product-Recommendation-System%20-%20Copy/frontend/index.html) file directly in any modern browser by double-clicking it.
*   Alternatively, run it using a local static file server (such as the VS Code Live Server extension) for a smoother development workflow.

---

## 📡 API Endpoints Reference

### 1. Get Categories
*   **Route:** `/api/categories`
*   **Method:** `GET`
*   **Response:** List of categorized directories.
```json
[
  {
    "id": 112,
    "category_name": "Electronics | Accessories | Cables"
  }
]
```

### 2. Get KNN Recommendations
*   **Route:** `/api/recommend`
*   **Method:** `GET`
*   **Parameters:**
    *   `q` (string): Text query to search.
    *   `asin` (string): ASIN identifier of the anchor item.
    *   `category_id` (integer): Filter search by category.
*   **Response:** Target product metadata and a tiered dictionary of nearest neighbors.
```json
{
  "target": {
    "asin": "B003L1A7S6",
    "name": "Amazon Basics USB Cable",
    "category": "Cables",
    "price": "399.00"
  },
  "recommendations": {
    "high": [...],
    "medium": [...],
    "low": [...]
  },
  "precision": 92
}
```

### 3. Generate AI Project Plan
*   **Route:** `/api/project_recommend`
*   **Method:** `POST`
*   **Headers:** `Content-Type: application/json`
*   **Body:**
```json
{
  "goal": "Build a study desk set-up"
}
```
*   **Response:** List of recommended items matching the requested goal.

---

## 📜 License & Disclaimer
*   This codebase is distributed under the **MIT License**. See the `LICENSE` file for more details.
*   This recommendation engine is developed for educational and demonstration purposes. It simulates recommendation behaviors using Amazon product metadata and generative AI.