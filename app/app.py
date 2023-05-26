from flask import Flask, request, jsonify
import numpy as np
from predicter import predict

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def get_prediction():
    features = request.get_json()
    pred = predict(features)
    ids = np.arange(len(pred))
    return jsonify({'ids':ids.tolist(), 'class':pred})

if __name__ == '__main__':
    app.run(debug=True, port = 5000)