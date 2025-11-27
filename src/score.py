import os
import json
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

model = None
processor = None

def init():
    global model, processor

    # Azure ML sets AZUREML_MODEL_DIR to the model path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(repo_root, "donut_qa_model", "donutQA", "outputs", "donut-lora")
    model_dir = os.environ.get("AZUREML_MODEL_DIR", model_path)
    model_dir = os.path.join(model_dir, "outputs", "donut-lora")
    print(f"Loading model from: {model_dir}")
    
    # List files in model directory for debugging
    if os.path.exists(model_dir):
        print(f"Files in model directory: {os.listdir(model_dir)}")
    else:
        print(f"WARNING: Model directory does not exist: {model_dir}")

    try:
        # Load processor (tokenizer + image preprocessing)
        processor = DonutProcessor.from_pretrained(model_dir)
        print("Processor loaded successfully")

        # Load Donut QA model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def run(raw_data):
    global model, processor

    try:
        # Azure ML passes data as dict or JSON string
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data
            
        image_bytes = data.get("image")  # assume base64-encoded image
        question = data.get("question", "")

        # Convert base64 to PIL image
        from PIL import Image
        import io, base64
        image = Image.open(io.BytesIO(base64.b64decode(image_bytes))).convert("RGB")

        # Create prompt for DocVQA
        prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        
        # Preprocess
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

        # Inference
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            use_cache=True,
        )
        
        # Decode answer
        generated = outputs.sequences if hasattr(outputs, "sequences") else outputs
        sequence = processor.batch_decode(generated)[0]
        answer = sequence.split("<s_answer>")[1].split("</s_answer>")[0] if "<s_answer>" in sequence else sequence

        return json.dumps({"answer": answer})

    except Exception as e:
        error_msg = str(e)
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in run(): {error_msg}")
        print(error_trace)
        return json.dumps({"error": error_msg})
