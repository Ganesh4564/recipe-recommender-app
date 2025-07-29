from flask import Flask, render_template_string, request
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)

# --- Load CSV files ---
recipes = pd.read_csv("RAW_recipes.csv")
interacts = pd.read_csv("RAW_interactions.csv")

# --- Build tag_string ---
def to_list(val):
    if isinstance(val, list):
        return [str(v) for v in val]
    s = str(val).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    return [x.strip(" '\"") for x in s.split(',') if x.strip()]

def build_tag_string(row):
    return " ".join(to_list(row.get('ingredients', '')) + to_list(row.get('tags', '')))

recipes['tag_string'] = recipes.apply(build_tag_string, axis=1)

# --- Merge ratings ---
INTERACT_KEY = 'recipe_id' if 'recipe_id' in interacts.columns else 'id'
avg_rating = interacts.groupby(INTERACT_KEY)['rating'].mean().reset_index()

recipes = recipes.merge(avg_rating, how='left', left_on='id', right_on=INTERACT_KEY)
recipes['rating'] = recipes['rating'].fillna(recipes['rating'].mean())

# --- Load model & index ---
model = SentenceTransformer("recipe_semantic_model")  # Replace with custom if needed
index = faiss.read_index("recipe_index.faiss")

titles = recipes['name'].tolist()
ids = recipes['id'].tolist()
ratings = recipes['rating'].tolist()

# --- Recommend logic ---
def recommend(query: str, top_k: int = 5):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        results.append({
            'title': titles[idx],
            'rating': float(ratings[idx]),
            'score': float(score)
        })
    return results

# --- HTML Template ---
template = """
<!doctype html>
<html>
<head>
    <title>🍽️ Recipe Recommender</title>
    <style>
        body { font-family: Arial; margin: 40px; background-color: #f8f9fa; }
        h1 { color: #2c3e50; }
        input[type=text] { width: 300px; padding: 10px; margin: 10px 0; }
        input[type=submit] { padding: 10px 20px; }
        .result { background: white; padding: 15px; margin: 10px 0; border-radius: 8px; box-shadow: 0 0 5px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <h1>🍽️ Recipe Recommender</h1>
    <form method="post">
        <input type="text" name="query" placeholder="e.g. chicken basil spicy" required />
        <input type="submit" value="Search" />
    </form>

    {% if results %}
        <h2>Top 5 Results</h2>
        {% for r in results %}
            <div class="result">
                <strong>{{ r.title }}</strong><br>
                ⭐ Rating: {{ r.rating }}<br>
                🔍 Score: {{ r.score }}
            </div>
        {% endfor %}
    {% endif %}
</body>
</html>
"""

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    if request.method == "POST":
        query = request.form["query"]
        results = recommend(query)
    return render_template_string(template, results=results)

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
