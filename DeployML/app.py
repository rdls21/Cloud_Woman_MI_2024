from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
# Columns
features = ['Smoking', 'Stroke', 'DiffWalking', 'AgeCategory', 'Diabetic', 'GenHealth', 'Asthma', 'KidneyDisease']

## Variables
age_categories = ['18-24', '25-29',
'30-34', '35-39',
'40-44', '45-49',
'50-54', '55-59',
'60-64', '65-69',
'70-74', '75-79',
'80 or older']
health_categories = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
diabetic_categories = ['No', 'No, borderline diabetes', 'Yes (during pregnancy)', 'Yes']
general_categories = ['No', 'Yes']

# Model
model = joblib.load('w_heart_b_svc.pkl')

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html', age_categories=age_categories, health_categories=health_categories, diabetic_categories=diabetic_categories)

@app.route('/', methods=['POST'])
def predict():
     # Get form data
    smoking = request.form['smoking']
    stroke = request.form['stroke']
    diffwalking = request.form['diffwalking']
    agecategory = request.form['agecategory']
    diabetic = request.form['diabetic']
    genhealth = request.form['genhealth']
    asthma = request.form['asthma']
    kidneydisease = request.form['kidneydisease']
    # Create de DF
    df = pd.DataFrame([[smoking, stroke, diffwalking, agecategory, diabetic, genhealth, asthma, kidneydisease]], columns=features)
    # Predict
    prediction = model.predict(df)
    # Return the prediction to the user
    return render_template('index.html', prediction=prediction,  age_categories=age_categories, health_categories=health_categories, diabetic_categories=diabetic_categories)

@app.route('/api', methods=['POST'])
def predict():
     # Get form data
    smoking = request.form['smoking']
    stroke = request.form['stroke']
    diffwalking = request.form['diffwalking']
    agecategory = request.form['agecategory']
    diabetic = request.form['diabetic']
    genhealth = request.form['genhealth']
    asthma = request.form['asthma']
    kidneydisease = request.form['kidneydisease']
    # Create de DF
    df = pd.DataFrame([[smoking, stroke, diffwalking, agecategory, diabetic, genhealth, asthma, kidneydisease]], columns=features)
    # Predict
    prediction = model.predict(df)
    # Return the prediction to the user
    return prediction


if __name__ == '__main__':
    app.run(port=3000, debug=True)