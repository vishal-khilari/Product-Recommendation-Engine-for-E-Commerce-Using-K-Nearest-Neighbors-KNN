# E-Commerce Product Recommendation System using K-Neighbor Nearest Algorithm

## 1. Project Overview
The project is a hybrid E-Commerce Product Recommendation System using the K-Neighbor Nearest (KNN) algorithm designed to accurately predict and recommend relevant products to a user based on their active browsing or search queries. It operates on a comprehensive Amazon dataset comprising products, prices, ratings, and categories. The backend logic is powered by Python and Flask, seamlessly delivering high-performance, real-time tiered recommendations to a modern frontend interface.

The system leverages Machine Learning techniques to surface:
- **Substitute Products** (Exact or very similar items)
- **Complementary Products** (Add-ons in the same ecosystem)
- **Cross-Sell Items** (Loosely related products in different categories)

Additionally, it integrates a Generative AI feature (via Google Gemini) that functions as an "AI Project Builder", which listens to natural language goals (e.g., "Build a gaming PC") and recommends a cart of required products.

## 2. Features
- **Hybrid Recommendation Engine:** Combines Content-Based Filtering (TF-IDF) and Collaborative/Behavioral Scoring (SVD).
- **Tiered Recommendations:** Categorizes recommendations into High Tier (Immediate Need), Medium Tier (Useful Add-ons), and Low Tier (General/Cross-Sell).
- **AI Project Builder:** Generates a list of recommended items based on a natural language project goal using the Gemini API.
- **Modern User Interface:** A responsive, dark luxury aesthetic frontend built with HTML, CSS, and JavaScript.
- **Real-Time Performance:** Flask API processes requests and returns data efficiently.

## 3. Project Structure
- `app.py`: The main Flask backend application that handles the API routes (`/api/categories`, `/api/recommend`, `/api/project_recommend`).
- `Product_Recommendation_System.py` / `Product_Recommendation_System.ipynb`: Initial ML research, data cleaning, and modeling scripts.
- `frontend/`: Contains the user interface (HTML, CSS, JS).
- `amazon_products.csv` & `amazon_categories.csv`: The datasets used for generating recommendations.
- `requirements.txt`: Python dependencies.

## 4. Detailed Implementation

### Data Processing & Feature Engineering
- **Cleaning:** The dataset is stripped of duplicates and normalized (lowercase strings, handling missing values).
- **Textual Features:** The product title and category names are combined into a single feature set for TF-IDF.
- **Behavioral Vectors:** Generates a synthetic user interaction matrix using Singular Value Decomposition (SVD) to group products into behavioral clusters based on root categories and popularity.

### Recommendation Engine Architecture (KNN Paradigm)
While using matrix dot products for performance, the logic implements an optimized **K-Nearest Neighbors (KNN)** approach using **Cosine Similarity**:
1. **Map Feature Space:** Products are mapped into a multi-dimensional space using TF-IDF (textual) and SVD (behavioral) coordinates.
2. **Calculate Distance:** Cosine Similarity determines the angle/distance between the target (anchor) product and others.
3. **Sort Neighbors:** Sorts products to retrieve the top similar candidates.
4. **Rule-Based Tiering:** Classifies the nearest neighbors into High, Medium, or Low tiers based on similarity scores, price, category, and title length.

## 5. Installation

### Prerequisites
- Python 3.8 or higher
- Modern Web Browser

### Setup Steps
1. **Clone or Download the Repository:**
   Navigate to the project directory in your terminal.

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   ```
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. **Install Dependencies:**
   Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables (For AI Project Builder):**
   To use the Gemini AI Project Builder feature, you need a Google Gemini API Key.
   - Create a `.env` file in the root directory.
   - Add your API key:
     ```env
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

## 6. How to Run the Project

1. **Start the Backend Server:**
   Run the Flask application from the root directory:
   ```bash
   python app.py
   ```
   The backend API will start on `http://127.0.0.1:5000` (or `http://localhost:5000`). Wait until you see `Initialization complete! API is ready.` in the console.

2. **Launch the Frontend:**
   - Open the `frontend` directory.
   - You can simply double-click the `index.html` file to open it in your web browser.
   - Alternatively, you can use a local web server extension (like Live Server in VS Code) to serve the `frontend` folder for a better development experience.

3. **Using the Application:**
   - Use the search bar to find products by name or category.
   - Browse the categories dropdown.
   - Click on the "AI Builder" in the navigation bar to try the Gemini generative recommendations.
   - Click on any product to see its tiered recommendations.

## 7. License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 8. Disclaimer
This project is solely for educational purposes to demonstrate advanced recommendation algorithms, Machine Learning workflows, and full-stack web integration.