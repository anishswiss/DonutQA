"""
Flask web service to run Donut QA model on VM
Deploy this on your compute instance or any VM
"""
import os
import json
import base64
import io
from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

app = Flask(__name__)

# Global variables for model
model = None
processor = None

def load_model():
    """Load the model once at startup"""
    global model, processor
    
    # Model path - adjust based on where you place the model on VM
    model_dir = os.environ.get("MODEL_PATH", "./donut_qa_model/donutQA/outputs/donut-lora")
    
    print(f"Loading model from: {model_dir}")
    
    # Check possible paths
    possible_paths = [
        model_dir,
        os.path.join(model_dir, "donutQA", "outputs", "donut-lora"),
        os.path.join(model_dir, "donut-lora"),
    ]
    
    actual_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
            actual_path = path
            break
    
    if actual_path is None:
        # Search recursively
        for root, dirs, files in os.walk(model_dir):
            if "config.json" in files:
                actual_path = root
                break
    
    if actual_path is None:
        raise FileNotFoundError(f"Could not find model in {model_dir}")
    
    print(f"Found model at: {actual_path}")
    
    # Load processor and model
    processor = DonutProcessor.from_pretrained(actual_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = VisionEncoderDecoderModel.from_pretrained(actual_path)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/score', methods=['POST'])
def score():
    """Main scoring endpoint"""
    global model, processor
    
    if model is None or processor is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        image_bytes = data.get("image")
        question = data.get("question", "")
        
        if not image_bytes:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        image = Image.open(io.BytesIO(base64.b64decode(image_bytes))).convert("RGB")
        
        # Create prompt for DocVQA
        prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        
        # Preprocess
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )
        
        # Decode answer
        sequence = processor.batch_decode(outputs.sequences)[0]
        answer = sequence.split("<s_answer>")[1].split("</s_answer>")[0] if "<s_answer>" in sequence else sequence
        
        return jsonify({"answer": answer})
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in score(): {str(e)}")
        print(error_trace)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Starting Flask server...")
    # Run on all interfaces, port 5000
    # In production, use a proper WSGI server like gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)

