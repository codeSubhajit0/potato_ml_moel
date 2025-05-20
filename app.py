from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import logging
import traceback

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model
try:
    model = tf.keras.models.load_model('model1.keras')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def preprocess(image_bytes):
    """Preprocess the image for model prediction"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((256, 256))  # Adjust based on your model input
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict class from uploaded image"""
    # Very detailed request logging
    logger.debug(f"Request method: {request.method}")
    logger.debug(f"Request headers: {request.headers}")
    logger.debug(f"Request files: {request.files}")
    logger.debug(f"Request form: {request.form}")
    logger.debug(f"Content type: {request.content_type}")
    
    # Check if the model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Alternative 1: Try to get the file from request.files
        if request.files and 'file' in request.files:
            logger.info("Found file in request.files['file']")
            file = request.files['file']
            img_bytes = file.read()
        
        # Alternative 2: Try to get raw data if file upload fails
        elif request.data:
            logger.info("Using raw request.data")
            img_bytes = request.data
            
        # Alternative 3: Try to load from a local file if specified in form data
        elif request.form and 'filename' in request.form:
            filename = request.form['filename']
            logger.info(f"Loading from local file: {filename}")
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    img_bytes = f.read()
            else:
                return jsonify({'error': f'Local file not found: {filename}'}), 400
        
        # Alternative 4: Try to get the first file regardless of name
        elif request.files:
            # Get the first file in the request, ignoring the field name
            for key in request.files:
                logger.info(f"Using first file found with key: {key}")
                file = request.files[key]
                img_bytes = file.read()
                break
        else:
            return jsonify({'error': 'No image data found in request'}), 400
        
        # Check if we have image data
        if not img_bytes:
            return jsonify({'error': 'Empty image data'}), 400
            
        img_array = preprocess(img_bytes)
        if img_array is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        logger.info("Starting prediction")
        prediction = model.predict(img_array)
        logger.info(f"Raw prediction output: {prediction}")
        
        # For debugging - log the shape and type of the prediction
        logger.info(f"Prediction shape: {prediction.shape}")
        logger.info(f"Prediction type: {type(prediction)}")
        
        # Handle different prediction output formats
        if isinstance(prediction, list):
            # If predict returns a list (e.g., multiple outputs)
            prediction = prediction[0]  # Take the first output if it's a list
            
        # Convert prediction to a standard format
        if hasattr(prediction, 'numpy'):
            # Convert TensorFlow tensor to numpy array if needed
            prediction_np = prediction.numpy()
        else:
            prediction_np = np.array(prediction)
            
        # Log the prediction array for debugging
        logger.info(f"Processed prediction: {prediction_np}")
        
        # Get predicted class and confidence
        predicted_class = int(np.argmax(prediction_np))
        confidence = float(np.max(prediction_np))
        
        logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")
        
        # Create response with detailed information
        response = {
            'prediction': predicted_class,
            'confidence': confidence,
            'message': 'Prediction successful',
            'prediction_array': prediction_np.tolist()  # Include the full prediction array
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

# Add a simple test endpoint that accepts any kind of request
@app.route('/test_upload', methods=['POST'])
def test_upload():
    """Test endpoint to verify file uploads are working"""
    try:
        result = {
            'headers': dict(request.headers),
            'content_type': request.content_type,
            'has_files': len(request.files) > 0,
            'files_keys': list(request.files.keys()) if request.files else [],
            'form_keys': list(request.form.keys()) if request.form else [],
            'has_data': len(request.data) > 0 if request.data else False,
        }
        
        if request.files and 'file' in request.files:
            file = request.files['file']
            result['filename'] = file.filename
            result['file_content_type'] = file.content_type
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)