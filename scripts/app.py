from flask import Flask, request, render_template, jsonify
import pickle  # To load your trained model
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and column information
model = pickle.load(open('model.sav', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/churn', methods=['POST'])
def churn():
    try:
        # Retrieve input data from the form and ensure they are correctly cast
        input_data = [
            int(request.form['SeniorCitizen']),
            float(request.form['MonthlyCharges']),
            float(request.form['TotalCharges']),
            request.form['gender'],
            request.form['Partner'],
            request.form['Dependents'],
            request.form['PhoneService'],
            request.form['MultipleLines'],
            request.form['InternetService'],
            request.form['OnlineSecurity'],
            request.form['OnlineBackup'],
            request.form['DeviceProtection'],
            request.form['TechSupport'],
            request.form['StreamingTV'],
            request.form['StreamingMovies'],
            request.form['Contract'],
            request.form['PaperlessBilling'],
            request.form['PaymentMethod'],
            int(request.form['tenure'])
        ]
        print(input_data)

        # Create a DataFrame from the input data
        column_names = [
            'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner',
            'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'tenure'
        ]

        input_df = pd.DataFrame([input_data], columns=column_names)

        # Group the tenure in bins of 12 months
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        input_df['tenure_group'] = pd.cut(input_df['tenure'], range(1, 80, 12), right=False, labels=labels)

        # Drop unnecessary columns (ensure 'customerID' isn't in the input)
        input_df.drop(['tenure'], axis=1, inplace=True)

        # One-hot encode the categorical features
        new_df = pd.get_dummies(input_df)
        print(new_df.columns)
        dummies_df = new_df.reindex(columns=model_columns, fill_value=0)
        print(dummies_df.values)
        
        #return model.predict(dummies_df)
        # Align the new DataFrame with the columns used during model training
        #new_df = new_df.reindex(columns=model_columns, fill_value=0)
        #print(new_df)
        #print(model.predict(dff))
        # Make prediction
        prediction = model.predict(dummies_df)
        print('this is prediction',prediction)
        prediction_result = 'Churn' if prediction[0] == 1 else 'No Churn'

        return jsonify({"Prediction": prediction_result})

    except Exception as e:
        return jsonify({"error": str(e)})
    
    
# @app.route('/predict_csv', methods=['POST'])
# def predict_csv():
#     try:
#         # Check if the file is in the request
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part in the request"})

#         file = request.files['file']

#         # If no file is selected
#         if file.filename == '':
#             return jsonify({"error": "No selected file"})

#         # Read the uploaded CSV file into a DataFrame
#     #     if file:
#     #         input_df = pd.read_csv(file)

#     #         # Preprocess and align the input data with the model columns
#     #         input_df['tenure_group'] = pd.cut(input_df['tenure'], range(1, 80, 12), right=False,
#     #                                           labels=["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)])
#     #         input_df.drop(['tenure'], axis=1, inplace=True)

#     #         new_df = pd.get_dummies(input_df, drop_first=True)
#     #         new_df = new_df.reindex(columns=model_columns, fill_value=0)
#     #         print('this is input df',new_df)

#     #         # Make predictions
#     #         predictions = model.predict(new_df)
#     #         input_df['Prediction'] = ['Churn' if pred == 1 else 'No Churn' for pred in predictions]

            

#     #         # Return the predictions as a JSON response
#     #         return input_df[['Prediction']].to_json(orient="records")

#     except Exception as e:
#         return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

