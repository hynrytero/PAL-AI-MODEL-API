from flask import Flask, request, jsonify
from yolo_model import predict  # Assuming `predict` processes an image and returns results

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to PAL-AI model api server!"

@app.route("/predict", methods=["POST"])
def predict_image():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    image_base64 = data["image"]

    # Process image through YOLO model
    predictions = predict(image_base64)  # Your YOLO model's prediction function

    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run()
