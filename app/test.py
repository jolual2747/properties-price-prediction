import pandas as pd
import numpy as np
from predicter import predict
import requests

df = pd.read_csv('https://fsl-assessment-public-files.s3.amazonaws.com/ai-challenge/train.csv')
json_df = df.head(10).to_json()

r = requests.post(url='http://127.0.0.1:5000/predict', json=json_df)
preds = r.json()
print(preds)