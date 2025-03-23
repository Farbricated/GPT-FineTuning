# GPT Fine-Tuning with Hugging Face 🚀

## 📌 Overview
This project fine-tunes a GPT model using **Hugging Face Transformers** and **datasets**. It trains a custom language model for text generation tasks using the **IMDB dataset** for sentiment analysis.

## 🛠️ Technologies Used
- Python
- Hugging Face Transformers
- PyTorch
- Datasets (Hugging Face Hub)
- Jupyter Notebook

## 🚀 Installation & Setup
### 1️⃣ Install Dependencies
```bash
pip install transformers datasets huggingface_hub torch --quiet
```

### 2️⃣ Load & Fine-Tune GPT Model
```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load IMDB dataset
dataset = load_dataset("imdb")

# Load GPT model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

### 3️⃣ Train the Model
```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
```

## 📊 Results
- Fine-tuned GPT model generates **sentiment-based text outputs**.
- Model is trained for **3 epochs** with batch size **4**.
- Can be further improved using **LoRA or PEFT techniques**.

## 📌 Future Improvements
- Use **larger datasets** for better generalization.
- Apply **LoRA (Low-Rank Adaptation)** for efficient fine-tuning.
- Deploy model using **Gradio or Hugging Face Spaces**.


