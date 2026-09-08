"""
Hugging Face Fine-Tuning Pipeline Module.

Demonstrates fine-tuning pretrained Transformer models (DistilBERT / BERT)
on text classification datasets using Hugging Face Trainer & PyTorch.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def fine_tune_transformer_classifier(
    texts: List[str],
    labels: List[int],
    model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
    output_dir: str = "./output/hf_model",
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Fine-tune pretrained Transformer classifier on text dataset."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dataset = TextDataset(encodings, labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels)))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        logging_steps=10,
        save_strategy="epoch",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    return model, tokenizer
