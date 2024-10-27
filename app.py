from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to PAL-AI model api server!"
   
@app.route('/predict')
def predict():
   
    return "Hello World"

if __name__ == '__main__':
    app.run(debug=True)
