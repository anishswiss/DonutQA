import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model
from transformers import Trainer


class DonutDataset(Dataset):
    def __init__(self, processor, image_dir, label_dir):
        self.processor = processor
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # assume labels have same filenames but .txt extension
        self.labels = [os.path.splitext(f)[0] + ".txt" for f in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import PIL.Image as Image

        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        image = Image.open(image_path).convert("RGB")
        with open(label_path, "r") as f:
            label = f.read().strip()

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        max_length = 128  # Set this for DocVQA and similar tasks
        input_ids = self.processor.tokenizer(
            label,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).input_ids.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": input_ids,
        }


class DonutCollator:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}



def main():
    #model_id = "naver-clova-ix/donut-base"
    model_id = "naver-clova-ix/donut-base-finetuned-docvqa"

    # Load processor and model
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
        
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Apply LoRA
    peft_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]
        
    )

    #model = get_peft_model(model, peft_config)

    # Data
    train_dataset = DonutDataset(processor, "data/train/images", "data/train/labels")
    val_dataset = DonutDataset(processor, "data/val/images", "data/val/labels")

    print("train_dataset[0].keys()")
    print(train_dataset[0].keys())

    collator = DonutCollator(processor)

    # Training args
    output_dir = os.environ.get("AZUREML_OUTPUT_DIR", "outputs")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        save_strategy="steps",
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        learning_rate=5e-5,
        predict_with_generate=True,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        data_collator=collator,
    )


    '''
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    '''

    trainer.train()

    # Save model + processor
    save_dir = os.path.join(output_dir, "donut-lora")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
