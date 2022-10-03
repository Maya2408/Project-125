from flask import Flask, jsonify, request
from classifier import prediction

app = Flask(__name__)

@app.route("/predict-alphabet", methods = ["POST"])
def predictAlphabet():
    image = request.files.get('alphabet')
    prediction = prediction(image)

    return jsonify({
        "prediction": prediction
    })

if __name__ == "__main__": 
    app.run(debug = True)