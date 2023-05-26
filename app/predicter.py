import pandas as pd
import numpy as np
import joblib

MODEL_DIR = 'app/models/pipeline.joblib'
price_dict = {'0-250000':0, '250000-350000':1,'350000-450000':2, '450000-650000':3,'650000+':4}
price_dict2 = dict(zip(price_dict.values(), price_dict.keys()))

def json_to_dataframe(json):
    df = pd.read_json(json)
    return df

def preprocessing(df):
    df['age'] = df['yearBuilt'].max() - df['yearBuilt']
    df['len_description'] = df['description'].str.len()
    df.drop(columns=['uid', 'description'], inplace=True)
    df['numOfBathrooms'] = df['numOfBathrooms'].astype('int64')
    df['hasSpa'] = df['hasSpa'].map({True:1, False:0})
    df.dropna(inplace=True)
    return df

def load_model(path):
    model = joblib.load(path)
    return model

def make_prediction(json):
    df = json_to_dataframe(json)
    df = preprocessing(df)
    model = load_model(MODEL_DIR)
    features_in = model.feature_names_in_.tolist()
    pred = model.predict(df[features_in]).tolist()
    return pred

def translate(pred, dict_):
    preds = []
    for i in pred:
        preds.append(dict_[int(i)])
    return preds


def predict(json):
    pred = make_prediction(json)
    preds = translate(pred, price_dict2)
    return preds