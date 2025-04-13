import pickle
from flask import Flask, request, jsonify

# Load the model
(dv, model) = pickle.load(open('model.bin', 'rb'))

app = Flask("churn")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = dv.transform(data)
    prediction = model.predict(X)
    print ("prediction = ", prediction)

    if prediction[0] == 1:
        return jsonify({'churn': True})
    else:
        return jsonify({'churn': False})

if __name__ == '__main__':
    app.run(debug=True, port=5000)