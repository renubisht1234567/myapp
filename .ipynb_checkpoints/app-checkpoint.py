from flask import Flask, request, jsonify
import pickle
model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message')
    link = request.form.get('link')
    result = {'message': message, 'link': link}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
