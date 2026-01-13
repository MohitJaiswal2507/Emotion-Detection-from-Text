import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer, BertModel, RobertaModel
import warnings
import os

# --- Configuration ---
BERT_MODEL_NAME = 'bert-base-uncased'
ROBERTA_MODEL_NAME = 'roberta-base'
# This path must match the one in your training script
MODEL_PATH = os.path.join('models', 'best_model.pth') 
MAX_LEN = 128

# --- 1. Define the 13 Emotion Labels ---
# This MUST be the same as the dictionary printed by your training notebook
EMOTION_LABELS = {
    0: 'anger',
    1: 'confusion',
    2: 'desire',
    3: 'disgust',
    4: 'fear',
    5: 'guilt',
    6: 'happiness',
    7: 'love',
    8: 'neutral',
    9: 'sadness',
    10: 'sarcasm',
    11: 'shame',
    12: 'surprise'
}
NUM_CLASSES = len(EMOTION_LABELS)

# --- 2. Define the Model Class ---
# This class MUST be identical to the one in your training script
# so that torch.load can reconstruct the model.
class HybridBertRoberta(nn.Module):
    def __init__(self, num_classes):
        super(HybridBertRoberta, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.roberta = RobertaModel.from_pretrained(ROBERTA_MODEL_NAME)
        self.classifier = nn.Linear(768 * 2, num_classes)

    def forward(self, bert_ids, bert_mask, roberta_ids, roberta_mask):
        bert_output = self.bert(input_ids=bert_ids, attention_mask=bert_mask)
        bert_cls_embedding = bert_output.pooler_output

        roberta_output = self.roberta(input_ids=roberta_ids, attention_mask=roberta_mask)
        roberta_cls_embedding = roberta_output.pooler_output

        combined_embedding = torch.cat((bert_cls_embedding, roberta_cls_embedding), dim=1)
        logits = self.classifier(combined_embedding)
        return logits

# --- 3. Load Model and Tokenizers ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model_and_tokenizers():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizers
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    roberta_tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)

    # Initialize model
    model = HybridBertRoberta(num_classes=NUM_CLASSES).to(device)
    
    # Load the saved model weights
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            st.error(f"Please make sure '{MODEL_PATH}' is the correct file and you have run the training.")
            return None, None, None, None
    else:
        st.error(f"Model file not found at '{MODEL_PATH}'")
        st.error("Please run the training script (`train_local.ipynb`) to create the model file.")
        return None, None, None, None
    
    model.eval() # Set model to evaluation mode
    print("Model and tokenizers loaded successfully.")
    return model, bert_tokenizer, roberta_tokenizer, device

# Load everything
model, bert_tokenizer, roberta_tokenizer, device = load_model_and_tokenizers()

# --- 4. Streamlit UI ---
st.title("Hybrid BERT-RoBERTa Emotion Classifier")
st.markdown(f"This app uses a hybrid model trained on **{NUM_CLASSES} emotions**, including nuances like sarcasm and confusion.")

# Stop the app if the model failed to load
if model is None:
    st.stop()

# Get user input
user_text = st.text_area("Enter your text here:", "I can't believe you would do that!")

if st.button("Predict Emotion"):
    if user_text:
        # --- 5. Prediction Logic ---
        try:
            # Tokenize for BERT
            bert_inputs = bert_tokenizer.encode_plus(
                user_text, None, add_special_tokens=True, max_length=MAX_LEN,
                padding='max_length', return_token_type_ids=False,
                truncation=True, return_attention_mask=True, return_tensors='pt'
            )
            
            # Tokenize for RoBERTa
            roberta_inputs = roberta_tokenizer.encode_plus(
                user_text, None, add_special_tokens=True, max_length=MAX_LEN,
                padding='max_length', return_token_type_ids=False,
                truncation=True, return_attention_mask=True, return_tensors='pt'
            )

            # Move inputs to device
            bert_ids = bert_inputs['input_ids'].to(device)
            bert_mask = bert_inputs['attention_mask'].to(device)
            roberta_ids = roberta_inputs['input_ids'].to(device)
            roberta_mask = roberta_inputs['attention_mask'].to(device)

            # Get prediction
            with torch.no_grad():
                outputs = model(
                    bert_ids=bert_ids,
                    bert_mask=bert_mask,
                    roberta_ids=roberta_ids,
                    roberta_mask=roberta_mask
                )
                
                # --- GET PROBABILITIES ---
                # Apply Softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # Get the main prediction index
                prediction_index = torch.argmax(probabilities, dim=1).item()
                # Get the label name
                prediction_label = EMOTION_LABELS.get(prediction_index, "Unknown")
                
                # Convert probabilities to a list of percentages
                probs_list = probabilities.cpu().numpy().flatten() * 100

            # --- 6. Display Results ---
            st.subheader("Prediction:")
            st.success(f"The model predicts the emotion is: **{prediction_label.upper()}**")

            st.subheader("Emotion Breakdown (Confidence %):")
            # Create a DataFrame for the chart
            df_probs = pd.DataFrame({
                'Emotion': [EMOTION_LABELS[i] for i in range(NUM_CLASSES)],
                'Percentage': probs_list
            })
            df_probs = df_probs.sort_values(by='Percentage', ascending=False)
            df_probs = df_probs.set_index('Emotion')
            
            # Display bar chart
            st.bar_chart(df_probs, height=350)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter some text to predict.")