from flask import Flask, render_template, request
import pickle
import requests

app = Flask(__name__)

# Load the Random Forest model using pickle
model = pickle.load(open('randomf.pkl', 'rb'))

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "6iE7d9U_k5ivmIOA05WcPb8Qnu_Ea3p2pWdNIlWF_utv"

def get_access_token(api_key):
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": api_key, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    return token_response.json()["access_token"]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Get input from the form
            input_features = [float(request.form['Deal_value']),
                              float(request.form['Weighted_amount']),
                              float(request.form['Internal_rating']),
                              float(request.form['Pitch']),
                              float(request.form['Fund_category']),
                              float(request.form['Lead_revenue']),
                              float(request.form['Resource']),
                              float(request.form['Lead_source']),
                              float(request.form['Level_of_meeting']),
                             ]
            
            # Make predictions
            prediction = model.predict([input_features])[0]
            print(prediction)  # Debug line
            
            # Set up scoring request to IBM Watson Machine Learning
            access_token = get_access_token(API_KEY)
            header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + access_token}
            payload_scoring = {"input_data": {"fields": ['Deal_value', 'Weighted_amount', 'Internal_rating', 'Pitch', 'Fund_category', 'Lead_revenue', 'Resource', 'Lead_source', 'Level_of_meeting'], "values": [input_features]}}
            
            response_scoring = requests.post('https://eu-gb.ml.cloud.ibm.com/ml/v4/deployments/b6df13e7-c47e-40ff-9487-f3fe3ba54cde/predictions?version=2021-05-01', json=payload_scoring, headers=header)
            print("Scoring response:")
            print(response_scoring.json())
            
        except Exception as e:
            print("Error:", e)  # Debug line

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)


