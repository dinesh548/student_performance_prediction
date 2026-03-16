"""
Student Performance Prediction — Flask Web Application
Run: python app.py
Open: http://127.0.0.1:5000
"""

import os, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, jsonify

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
DATA_PATH  = os.path.join(BASE_DIR, 'student_performance.csv')

# Load model once at startup
model = joblib.load(MODEL_PATH)

# Load dataset for dashboard stats
df = pd.read_csv(DATA_PATH)
df.fillna(df.median(numeric_only=True), inplace=True)

FEATURES = ['study_hours','attendance','assignments_completed',
            'previous_grade','participation','sleep_hours','internet_usage']

# ─── Helper ───────────────────────────────────────────────────────────────────
def get_grade(score):
    if score >= 90: return 'A+', 'Outstanding'
    if score >= 80: return 'A',  'Excellent'
    if score >= 70: return 'B',  'Good'
    if score >= 60: return 'C',  'Average'
    if score >= 50: return 'D',  'Below Average'
    return 'F', 'Needs Improvement'

def get_tips(data):
    tips = []
    if data['study_hours'] < 4:
        tips.append("📚 Increase study hours to at least 4–6 hours per day.")
    if data['attendance'] < 75:
        tips.append("🏫 Improve class attendance — aim for 80%+.")
    if data['assignments_completed'] < 7:
        tips.append("📝 Complete more assignments for better scores.")
    if data['sleep_hours'] < 6:
        tips.append("😴 Get adequate sleep (7–8 hrs) for better focus.")
    if data['internet_usage'] > 5:
        tips.append("📵 Reduce recreational internet usage during study time.")
    if not tips:
        tips.append("🎉 Keep up the excellent habits!")
    return tips

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    stats = {
        'total_students': len(df),
        'avg_score':      round(df['final_score'].mean(), 1),
        'top_score':      round(df['final_score'].max(), 1),
        'pass_rate':      round((df['final_score'] >= 50).mean() * 100, 1),
    }
    return render_template('index.html', stats=stats)


@app.route('/dashboard')
def dashboard():
    # Dataset overview
    overview = {
        'rows':    len(df),
        'columns': len(df.columns),
        'missing': int(pd.read_csv(DATA_PATH).isnull().sum().sum()),
        'avg_score': round(df['final_score'].mean(), 2),
        'std_score': round(df['final_score'].std(), 2),
    }
    # Grade distribution
    bins   = [0, 50, 60, 70, 80, 90, 101]
    labels = ['F (<50)', 'D (50-60)', 'C (60-70)', 'B (70-80)', 'A (80-90)', 'A+ (90+)']
    df['grade_bin'] = pd.cut(df['final_score'], bins=bins, labels=labels, right=False)
    grade_dist = df['grade_bin'].value_counts().sort_index().to_dict()

    # Model metrics (hard-coded from training for display)
    metrics = {
        'Linear Regression': {'R2': 0.8957, 'MAE': 3.31, 'MSE': 18.07},
        'Decision Tree':     {'R2': 0.6959, 'MAE': 5.64, 'MSE': 52.70},
        'Random Forest':     {'R2': 0.8592, 'MAE': 3.88, 'MSE': 24.39},
    }
    return render_template('dashboard.html', overview=overview,
                           grade_dist=json.dumps(grade_dist), metrics=metrics)


@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html')


@app.route('/result', methods=['POST'])
def result():
    try:
        data = {
            'study_hours':           float(request.form['study_hours']),
            'attendance':            float(request.form['attendance']),
            'assignments_completed': float(request.form['assignments_completed']),
            'previous_grade':        float(request.form['previous_grade']),
            'participation':         float(request.form['participation']),
            'sleep_hours':           float(request.form['sleep_hours']),
            'internet_usage':        float(request.form['internet_usage']),
        }

        X = pd.DataFrame([data], columns=FEATURES)
        score = float(np.clip(model.predict(X)[0], 0, 100))
        grade, label = get_grade(score)
        tips = get_tips(data)

        # Percentile rank in dataset
        percentile = round((df['final_score'] < score).mean() * 100, 1)

        return render_template('result.html',
                               score=round(score, 1),
                               grade=grade,
                               label=label,
                               tips=tips,
                               percentile=percentile,
                               inputs=data)
    except Exception as e:
        return render_template('predict.html', error=f"Error: {str(e)}")


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Student Performance Prediction Web App")
    print("  Starting server...\n")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
