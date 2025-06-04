from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

CSV_FILE = 'expense_data.csv'

# ----------------------------
# Train models from data
# ----------------------------
def train_models():
    if not os.path.exists(CSV_FILE):
        return None, None

    df = pd.read_csv(CSV_FILE)
    if len(df) < 10:
        return None, None  # Not enough data

    X = df[['income', 'num_dependents', 'num_transactions', 'avg_transaction', 'prev_expense']]

    # Regression model
    y_reg = df['monthly_expense']
    reg_model = LinearRegression()
    reg_model.fit(X, y_reg)

    # Logistic model
    y_log = df['is_overspending']
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X, y_log)

    # Save models
    with open('regression_model.pkl', 'wb') as f:
        pickle.dump(reg_model, f)
    with open('logistic_model.pkl', 'wb') as f:
        pickle.dump(log_model, f)

    return reg_model, log_model

# ----------------------------
# Load models (if exists)
# ----------------------------
def load_models():
    try:
        with open('regression_model.pkl', 'rb') as f:
            reg_model = pickle.load(f)
        with open('logistic_model.pkl', 'rb') as f:
            log_model = pickle.load(f)
        return reg_model, log_model
    except:
        return train_models()

# ----------------------------
# API Routes
# ----------------------------
@app.route('/')
def index():
    return "Expense ML API is running."

@app.route('/save-expense', methods=['POST'])
def save_expense():
    data = request.json
    df_new = pd.DataFrame([data])

    if os.path.exists(CSV_FILE):
        df_new.to_csv(CSV_FILE, mode='a', index=False, header=False)
    else:
        df_new.to_csv(CSV_FILE, index=False)

    return jsonify({"message": "Expense data saved successfully."})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([
        data['income'],
        data['num_dependents'],
        data['num_transactions'],
        data['avg_transaction'],
        data['prev_expense']
    ]).reshape(1, -1)

    reg_model, log_model = load_models()
    if not reg_model or not log_model:
        return jsonify({"error": "Not enough data to train models yet."}), 400

    predicted_expense = reg_model.predict(features)[0]
    overspending_prob = log_model.predict_proba(features)[0][1]
    overspending_pred = int(overspending_prob >= 0.5)

    return jsonify({
        "predicted_monthly_expense": round(predicted_expense, 2),
        "overspending_probability": round(overspending_prob, 3),
        "will_overspend": overspending_pred
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

