Assignment 2 - Deep Learning
Name: Palakala Siddharth
Roll Number: 160122737190 College: CBIT (Chaitanya Bharathi Institute of Technology)
Course: Deep Learning
Assignment:Seq2Seq transliteration model using RNN-based architectures on the Dakshina dataset.


🎯 Goal
Build a Seq2Seq transliteration model using RNN-based architectures on the Dakshina dataset.

Fine-tune GPT-2 to generate English song lyrics.

📂 Contents
seq2seq_model.py: Main script for transliteration model

gpt2_finetune.py: Fine-tuning GPT-2 with Huggingface

🔁 Task 1: Seq2Seq Transliteration Model
✅ Features:
Configurable embedding dimension, RNN cell (RNN/LSTM/GRU), and number of layers

Encoder-decoder architecture

Character-level tokenization

💻 Run:
python seq2seq_model.py
🔢 Customization:
Adjust the config dictionary for:

embedding_dim: character embedding size

hidden_units: number of units in RNN cell

cell_type: "RNN", "LSTM", or "GRU"

num_layers: layers in encoder and decoder

📊 Output:
Training accuracy
Test accuracy

Sample test predictions

🎵 Task 2: Fine-tune GPT-2 for Lyrics
📂 Dataset Suggestions:
Lyrics Dataset

Poetry Corpus

🧪 Run:
python gpt2_finetune.py --data_path path/to/lyrics.txt --output_dir ./gpt2_lyrics_model
⚙️ Requirements:
pip install transformers datasets
