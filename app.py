from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import os
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load YOLO model
model_path = os.path.join(os.path.dirname(__file__), 'pal-ai-model', 'palai_model.pt')
model = YOLO(model_path)

def process_base64_image(base64_string):
    try:
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image, None
    except Exception as e:
        return None, str(e)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "API is running. Send POST requests to /predict"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Check if image data is present
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image data provided',
                'message': 'Request must include image field with base64 image data'
            }), 400
        
        # Get the base64 image data
        image_data = data['image']
        
        # Process the base64 image
        image, error = process_base64_image(image_data)
        if error:
            return jsonify({
                'error': 'Failed to process image',
                'message': error
            }), 400
            
        # Run YOLO prediction
        results = model(image)
        
        # Process predictions
        predictions = []
        for result in results:
            for box in result.boxes:
                prediction = {
                    "xmin": float(box.xyxy[0][0].item()),
                    "ymin": float(box.xyxy[0][1].item()),
                    "xmax": float(box.xyxy[0][2].item()),
                    "ymax": float(box.xyxy[0][3].item()),
                    "confidence": float(box.conf[0].item()),
                    "class": int(box.cls[0].item())
                }
                predictions.append(prediction)
        
        # Return the predictions
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)