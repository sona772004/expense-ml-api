from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

CSV_FILE = 'expense_data.csv'

def train_models():
    if not os.path.exists(CSV_FILE):
        return None, None

    df = pd.read_csv(CSV_FILE)
    if len(df) < 10:
        return None, None

    X = df[['income', 'num_dependents', 'num_transactions', 'avg_transaction', 'prev_expense']]
    y_reg = df['monthly_expense']
    y_log = df['is_overspending']

    reg_model = LinearRegression()
    reg_model.fit(X, y_reg)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X, y_log)

    with open('regression_model.pkl', 'wb') as f:
        pickle.dump(reg_model, f)
    with open('logistic_model.pkl', 'wb') as f:
        pickle.dump(log_model, f)

    return reg_model, log_model

def train_savings_model():
    if not os.path.exists(CSV_FILE):
        return None

    df = pd.read_csv(CSV_FILE)
    if len(df) < 10:
        return None

    if 'monthly_expense' not in df.columns or 'income' not in df.columns:
        return None

    df['savings'] = df['income'] - df['monthly_expense']
    X = df[['income', 'monthly_expense']]
    y = df['savings']

    model = LinearRegression()
    model.fit(X, y)

    with open('savings_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def load_models():
    try:
        with open('regression_model.pkl', 'rb') as f:
            reg_model = pickle.load(f)
        with open('logistic_model.pkl', 'rb') as f:
            log_model = pickle.load(f)
        return reg_model, log_model
    except:
        return train_models()

def load_savings_model():
    try:
        with open('savings_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        return train_savings_model()

@app.route('/')
def home():
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

@app.route('/predict-savings', methods=['POST'])
def predict_savings():
    data = request.json
    income = data.get('income')
    expense = data.get('monthly_expense')

    if income is None or expense is None:
        return jsonify({'error': 'Missing income or monthly_expense'}), 400

    model = load_savings_model()
    if not model:
        return jsonify({'error': 'Not enough data to train savings model'}), 400

    features = np.array([[income, expense]])
    predicted_savings = model.predict(features)[0]

    return jsonify({
        'predicted_savings': round(predicted_savings, 2)
    })

@app.route('/expense-trends', methods=['GET'])
def expense_trends():
    try:
        df = pd.read_csv(CSV_FILE)
        if 'prev_expense' not in df.columns:
            return jsonify({'error': 'Missing prev_expense column in CSV'}), 400

        df['fake_month'] = (df.index // 5) + 1
        monthly_expense = df.groupby('fake_month')['prev_expense'].sum()

        plt.figure(figsize=(8, 4))
        plt.plot(monthly_expense.index, monthly_expense.values, marker='o', color='blue')
        plt.title('Simulated Monthly Expense Trend')
        plt.xlabel('Fake Month Number')
        plt.ylabel('Total Expense')
        plt.grid(True)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)

        return send_file(img, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/monthly-expense-bar', methods=['GET'])
def monthly_expense_bar():
    try:
        df = pd.read_csv(CSV_FILE)
        df['fake_month'] = (df.index // 5) + 1
        monthly_expense = df.groupby('fake_month')['monthly_expense'].sum()

        plt.figure(figsize=(8, 4))
        plt.bar(monthly_expense.index, monthly_expense.values, color='skyblue')
        plt.title('Monthly Total Expenses')
        plt.xlabel('Fake Month')
        plt.ylabel('Total Monthly Expense')
        plt.grid(axis='y')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/overspending-pie', methods=['GET'])
def overspending_pie():
    try:
        df = pd.read_csv(CSV_FILE)
        counts = df['is_overspending'].value_counts()
        labels = ['Not Overspending', 'Overspending']
        values = [counts.get(0, 0), counts.get(1, 0)]

        plt.figure(figsize=(6, 6))
        plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
        plt.title('Overspending Distribution')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.seek(0)
        return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/income-vs-expense-scatter', methods=['GET'])
def income_vs_expense_scatter():
    try:
        df = pd.read_csv(CSV_FILE)
        plt.figure(figsize=(8, 5))
        plt.scatter(df['income'], df['monthly_expense'], alpha=0.7, c='orange')
        plt.title('Income vs Monthly Expense')
        plt.xlabel('Income')
        plt.ylabel('Monthly Expense')
        plt.grid(True)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/avg-transaction-histogram', methods=['GET'])
def avg_transaction_histogram():
    try:
        df = pd.read_csv(CSV_FILE)
        plt.figure(figsize=(8, 4))
        plt.hist(df['avg_transaction'], bins=10, color='purple', edgecolor='black')
        plt.title('Average Transaction Distribution')
        plt.xlabel('Average Transaction Value')
        plt.ylabel('Frequency')
        plt.grid(True)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/savings-goal-status', methods=['POST'])
def savings_goal_status():
    data = request.json
    required_fields = ['income', 'num_dependents', 'num_transactions', 'avg_transaction', 'prev_expense']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing fields in input"}), 400

    saving_goal = data.get('monthly_saving_goal', 10000)

    features = np.array([
        data['income'],
        data['num_dependents'],
        data['num_transactions'],
        data['avg_transaction'],
        data['prev_expense']
    ]).reshape(1, -1)

    reg_model, _ = load_models()
    if not reg_model:
        return jsonify({"error": "Not enough data to train models yet."}), 400

    predicted_expense = reg_model.predict(features)[0]
    predicted_savings = data['income'] - predicted_expense

    difference = predicted_savings - saving_goal

    if difference >= 0:
        status = "On track to meet or exceed your savings goal!"
    else:
        status = f"You're likely to fall short of your savings goal by {abs(difference):.2f}."

    return jsonify({
        "predicted_savings": round(predicted_savings, 2),
        "savings_goal": saving_goal,
        "difference": round(difference, 2),
        "status": status
    })

@app.route('/retrain-models', methods=['POST'])
def retrain_models():
    reg_model, log_model = train_models()
    savings_model = train_savings_model()

    if not reg_model or not log_model or not savings_model:
        return jsonify({"error": "Not enough data to retrain models."}), 400

    return jsonify({"message": "Models retrained successfully."})

@app.route('/realtime-expense', methods=['POST'])
def realtime_expense():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        required_fields = ['amount', 'income']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        amount = float(data['amount'])
        income = float(data['income'])
        monthly_saving_goal = float(data.get('monthly_saving_goal', 10000))
        num_dependents = int(data.get('num_dependents', 0))
        num_transactions = int(data.get('num_transactions', 0))
        avg_transaction = float(data.get('avg_transaction', 0))
        prev_expense = float(data.get('prev_expense', 0))

        reg_model, log_model = load_models()
        if not reg_model or not log_model:
            return jsonify({"error": "Not enough data to train models yet."}), 400

        features = np.array([[income, num_dependents, num_transactions, avg_transaction, prev_expense]])
        predicted_expense = reg_model.predict(features)[0]

        total_monthly_expense = predicted_expense + amount
        savings_after_expense = income - total_monthly_expense

        features_new = np.array([[income, num_dependents, num_transactions + 1, avg_transaction, total_monthly_expense]])
        overspending_prob = log_model.predict_proba(features_new)[0][1]
        is_overspending = int(overspending_prob >= 0.5)

        goal_met = bool(savings_after_expense >= monthly_saving_goal)
        alert = "Good job! You are on track with your savings." if goal_met else "Alert: You have overspent your income!"

        return jsonify({
            "total_monthly_expense": round(total_monthly_expense, 2),
            "savings_after_expense": round(savings_after_expense, 2),
            "monthly_saving_goal": monthly_saving_goal,
            "goal_met": goal_met,
            "alert": alert,
            "is_overspending": is_overspending
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
