import pandas as pd
from pycaret.classification import predict_model, load_model

class autoML:
    def __init__(self, df):
        self.df = df
        
    def load_data(self):
        df = pd.read_csv(self.df)
        return df
    
    def make_predictions(self):
        model = load_model('Logistic Regression')
        predictions = predict_model(model, data=self.load_data())
        return predictions['prediction_label']

    def print_self(self):
        predictions = self.make_predictions()
        print('predictions: ')
        print(predictions)

filepath = input('What is the name of the file?')
automl = autoML(df=filepath)
automl.print_self()