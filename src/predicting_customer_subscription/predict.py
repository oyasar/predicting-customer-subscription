import datetime
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import OneHotEncoder
import datetime

date = datetime.datetime.now().strftime('%Y-%m-%d')
app = Flask(__name__)

# Load the XGBoost model
with open('models/xgb_model.pkl', 'rb') as f:
    xgb_model, features = pickle.load(f)

def encode(df):
    """
    One-hot encode the categorical columns and return the DataFrame.
    Args:
        df: dataset to be scored.

    Returns:
        DataFrame with one-hot encoded columns.

    """
    # One-hot encoding of the categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_cat = pd.DataFrame(enc.fit_transform(df[cat_cols]), index=df.index)
    df_cat.columns = enc.get_feature_names_out(cat_cols)
    df = pd.concat([df.drop(cat_cols, axis=1), df_cat], axis=1)

    return df

def match_cols(df_score, features):
    """
    Check if encoding results in the same columns
     as the training data. if not, add the missing columns.

    Args:
        df_score: DataFrame to match columns.
        df_train: DataFrame with columns to match.

    Returns:
        DataFrame with columns matched.
    """

    missing_cols = (set(features)- set(df_score.columns))
    for col in missing_cols:
        df_score[col] = 0

    # reorder:
    df_score = df_score[features]

    return df_score

@app.route('/')
def index():
    return '''
        <html>
          <head>
            <title>Upload CSV for Prediction</title>
          </head>
          <body>
            <h1>Upload CSV for Prediction</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
              <input type="file" name="file">
              <br><br>
              <input type="submit" value="Submit">
            </form>
          </body>
        </html>
        '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    df = pd.read_excel(file)

    df = encode(df)

    df = match_cols(df, features)

    # Make predictions
    predictions = xgb_model.predict(df)

    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])

    # for local implementation use 'output/predictions.csv',
    predictions_df.to_csv(f'/app/output/predictions_{date}.csv', index=False)

    return jsonify({'message': f'Predictions saved to output/predictions_{date}.csv'})

if __name__ == '__main__':
    app.run(debug=True)