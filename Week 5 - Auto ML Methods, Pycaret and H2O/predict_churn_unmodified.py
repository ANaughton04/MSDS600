import pandas as pd
from pycaret.classification import predict_model, load_model
from sklearn.preprocessing import LabelEncoder

class autoML:
    def __init__(self, df):
        self.df = df
        
    def load_data(self):
        df = pd.read_csv(self.df)
        df['charge_per_tenure'] = df['TotalCharges']/df['tenure']
        label_encoder = LabelEncoder()
        df['PhoneService'].replace({'Yes':1, 'No':0}, inplace=True)
        df['Contract'] = label_encoder.fit_transform(df['Contract'])
        df['PaymentMethod'] = label_encoder.fit_transform(df['PaymentMethod'])
        return df
    
    def make_predictions(self):
        model = load_model('Logistic Regression')
        predictions = predict_model(model, data=self.load_data())
        return predictions['prediction_label']

    def print_self(self):
        predictions = self.make_predictions()
        print('predictions: ')
        print(predictions)

filepath = input('What is the name of your file?')
automl = autoML(df=filepath)
automl.print_self()


    