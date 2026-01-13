# 🧠 Hybrid BERT-RoBERTa Ensemble for Emotion Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![Transformers](https://img.shields.io/badge/Transformers-4.x-yellow.svg)](https://huggingface.co/docs/transformers/index)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)  

---

## 🚀 Overview

This project implements a **Hybrid BERT-RoBERTa Ensemble** designed for high-granularity **Text Emotion Classification (TEC)**. While traditional models are often limited to basic emotions, this architecture captures 13 nuanced human states including **sarcasm, confusion, and shame**.

By concatenating the 768-dimensional CLS feature embeddings from both `bert-base-uncased` and `roberta-base`, the model creates a rich **1536-dimensional feature vector** for superior classification performance.

---

## ✨ Features

- 🎭 **Nuanced Detection** → Trained on a 13-label dataset including Sarcasm, Guilt, and Confusion.
- 🤖 **Ensemble Architecture** → Parallel fine-tuning of BERT and RoBERTa models.
- 📊 **Superior Performance** → Achieves a validation accuracy of **69.93%**, outperforming single-model baselines.
- 🧪 **Research-Backed** → Based on methodology developed at **Bennett University**.

---

## 🏗️ Architecture

The model processes text through dual pipelines, concatenating the final hidden states (CLS tokens) before passing them through a linear classifier.

1. **Preprocessing**: Dual tokenization using BERT and RoBERTa tokenizers.
2. **Feature Extraction**: Parallel processing through pre-trained Transformer layers.
3. **Fusion**: Concatenation of embeddings into a 1536-dimensional vector.
4. **Classification**: Final linear layer for 13-class emotion prediction.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone [https://github.com/MohitJaiswal2507/Emotion-Detection-from-Text](https://github.com/MohitJaiswal2507/Emotion-Detection-from-Text)
cd Emotion-Detection-from-Text
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 3️⃣ Install Dependencies 

```bash
pip install -r requirements.txt

```

### 4️⃣ Training (train.ipynb)
Open and run the train.ipynb notebook. This script:

* Loads the Emotions Dataset.

* Performs dual tokenization for BERT and RoBERTa.

* Trains the hybrid model for N(any number of your choice) epochs on any GPU 

* Saves the final model weights to the models/ directory.

### 5️⃣ Once Training is complete, Run :
   ```bash
python app.py

```


## 🛠 Tech Stack

* **Frameworks**: PyTorch, Hugging Face Transformers.


* **Models**: BERT (bert-base-uncased) & RoBERTa (roberta-base).


* **Hardware**: Training performed on **Nvidia RTX 4060 GPU**.


* **Data**: Emotions Dataset by boltuix (130k+ samples).


---

## 📂 Project Structure

```
├── train.ipynb           # Model training & architecture logic
├── app.py                # Inference script/Application
├── requirements.txt      # Project dependencies
├── data/                 # Dataset directory
├── models/               # Saved model weights
├── screenshots/          # Performance graphs & Confusion Matrix
├── .gitignore            # Files to ignore (e.g., __pycache__)
└── Emotion_Detection_from_text.pdf # Research Paper

```

---

## 👨‍💻 Author

**Mohit Jaiswal** 📧 [mohitjaiswal2507@gmail.com](mailto:mohitjaiswal2507@gmail.com)
---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](https://www.google.com/search?q=./LICENSE) file for details.

