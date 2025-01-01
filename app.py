from flask import Flask, render_template, request

import joblib

import pandas as pd

import numpy as np

import altair as alt

import json



# Load the pre-trained model

pipe_lr = joblib.load(open("model/mental_health_rf.pkl", "rb"))



# Define prediction functions

def predict_conditions(docx):

    results = pipe_lr.predict([docx])

    return results[0]



def get_prediction_proba(docx):

    results = pipe_lr.predict_proba([docx])

    return results



# Initialize Flask application

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])

def home():

    prediction = None

    probability = None

    raw_text = None

    proba_df_clean = None



    if request.method == 'POST':

        raw_text = request.form['raw_text']



        # Get the prediction and probability

        prediction = predict_conditions(raw_text)

        probability = get_prediction_proba(raw_text)



        # Prepare the prediction probabilities for visualization

        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)

        proba_df_clean = proba_df.T.reset_index()

        proba_df_clean.columns = ["conditions", "probability"]



        # Max probability for confidence

        confidence = np.max(probability)



        # Render the page with results

        return render_template(

            'index.html',

            raw_text=raw_text,

            prediction=prediction,

            confidence=confidence,

            proba_df_clean=proba_df_clean.to_dict(orient='records')  # Pass probabilities to the template

        )



    # If it's a GET request, just render the empty form

    return render_template('index.html')



# Run the app

if __name__ == '__main__':

    app.run(debug=True)

