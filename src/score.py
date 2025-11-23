import os
import json
import torch
from transformers import AutoProcessor, AutoModelForVisionTextGeneration

model = None
processor = None

def init():
    global model, processor

    model_dir = os.environ.get("AZUREML_MODEL_DIR", "./donut_qa_model/donutQA/outputs/donut-lora")  

    # Load processor (tokenizer + image preprocessing)
    processor = AutoProcessor.from_pretrained(model_dir)

    # Load Donut QA model
    model = AutoModelForVisionTextGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto"  # automatically uses GPU if available
    )
    model.eval()

def run(raw_data):
    global model, processor

    try:
        data = json.loads(raw_data)
        image_bytes = data.get("image")  # assume base64-encoded image
        question = data.get("question", "")

        # Convert base64 to PIL image
        from PIL import Image
        import io, base64
        image = Image.open(io.BytesIO(base64.b64decode(image_bytes)))

        # Preprocess
        inputs = processor(images=image, text=question, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        # Inference
        outputs = model.generate(**inputs)
        answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return json.dumps({"answer": answer})

    except Exception as e:
        return json.dumps({"error": str(e)})
