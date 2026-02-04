from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
df = pd.read_csv("products.csv")
df = df[[
    "product_title",
    "product_category",
    "discounted_price"
]].dropna()

df["original_price"] = df["discounted_price"].copy()

df["product_category"] = df["product_category"].astype("category")
category_mapping = dict(enumerate(df["product_category"].cat.categories))
df["category_code"] = df["product_category"].cat.codes

scaler = MinMaxScaler()
df[["discounted_price"]] = scaler.fit_transform(df[["discounted_price"]])

features = df[["category_code", "discounted_price"]]


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    selected_category = None
    selected_price = None

    if request.method == "POST":
        selected_category = request.form.get("category")
        selected_price = request.form.get("price")
        max_price = float(selected_price)

        category_code = list(category_mapping.keys())[list(category_mapping.values()).index(selected_category)]
        price_scaled = scaler.transform([[float(max_price)]])[0][0]

        user_vector = [[category_code, price_scaled]]

        filtered_df = df[
            (df["product_category"] == selected_category) &
            (df["original_price"] <= max_price)
        ]
        
        if len(filtered_df) > 0:
            filtered_features = filtered_df[["category_code", "discounted_price"]]
            similarity_scores = cosine_similarity(user_vector, filtered_features)[0]
            filtered_df = filtered_df.copy()
            filtered_df["similarity"] = similarity_scores
        
            recommendations = filtered_df.sort_values(by="similarity", ascending=False).head(5)

    return render_template(
        "index.html",
        categories=category_mapping.values(),
        recommendations=recommendations,
        selected_category=selected_category,
        selected_price=selected_price
    )


if __name__ == "__main__":
    app.run(debug=True)
