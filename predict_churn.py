import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def make_predictions(df):
    model = load_model('Logistic Regression')
    predictions = predict_model(model, data=df)
    return predictions['prediction_label']

if __name__ == '__main__':
    df = load_data('new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions: ')
    print(predictions)