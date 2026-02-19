from flask import Flask, request
import numpy as np
from sklearn.cluster import AgglomerativeClustering

app = Flask(__name__)

# Original dataset
original_data = np.array([
    [45, 150],
    [50, 155],
    [55, 160],
    [60, 162],
    [65, 165],
    [70, 170],
    [75, 172],
    [80, 175],
    [85, 178],
    [90, 180]
])

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        weight = float(request.form["weight"])
        height = float(request.form["height"])

        new_dataset = np.vstack([original_data, [weight, height]])

        model = AgglomerativeClustering(n_clusters=3)
        clusters = model.fit_predict(new_dataset)

        cluster_number = clusters[-1]

        # Friendly labels
        if cluster_number == 0:
            prediction = "Cluster 0 → Light Weight Group"
        elif cluster_number == 1:
            prediction = "Cluster 1 → Medium / Normal Weight Group"
        else:
            prediction = "Cluster 2 → Higher Weight Group"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Weight & Height Clustering</title>
        <style>
            body {{
                margin: 0;
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #74ebd5, #9face6);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }}

            .container {{
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 15px 30px rgba(0,0,0,0.2);
                text-align: center;
                width: 350px;
            }}

            h2 {{
                margin-bottom: 25px;
                color: #333;
            }}

            input {{
                width: 100%;
                padding: 12px;
                margin: 10px 0;
                border-radius: 8px;
                border: 1px solid #ccc;
                font-size: 14px;
            }}

            button {{
                width: 100%;
                padding: 12px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                cursor: pointer;
                transition: 0.3s;
            }}

            button:hover {{
                background-color: #45a049;
            }}

            .result {{
                margin-top: 20px;
                padding: 15px;
                border-radius: 10px;
                background-color: #f0f8ff;
                font-weight: bold;
                color: #333;
            }}
        </style>
    </head>
    <body>

        <div class="container">
            <h2>Weight & Height Clustering</h2>

            <form method="POST">
                <input type="number" name="weight" placeholder="Enter Weight (kg)" required>
                <input type="number" name="height" placeholder="Enter Height (cm)" required>
                <button type="submit">Predict Cluster</button>
            </form>

            {f'<div class="result">{prediction}</div>' if prediction else ""}

        </div>

    </body>
    </html>
    """
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

