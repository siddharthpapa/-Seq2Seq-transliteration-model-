
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import transformers
from datasets import load_dataset


# CONFIGURATION (flexible hyperparameters)
config = {
    "embedding_dim": 128,
    "hidden_units": 256,
    "cell_type": "LSTM",  # Options: RNN, LSTM, GRU
    "num_layers": 1,
    "batch_size": 64,
    "epochs": 20
}


# Load and preprocess data (Dakshina dataset - transliteration task)
def load_dakshina_dataset():
    # Placeholder logic; user must replace with actual file loading
    # x: Latin characters, y: Devanagari characters
    # Should return: input_texts, target_texts
    return ["ghar"], ["\u0918\u0930"]  # Example: ghar -> घर


# Tokenization and Vectorization
def vectorize_data(input_texts, target_texts, input_tokenizer, target_tokenizer, max_len=20):
    encoder_input_data = input_tokenizer.texts_to_sequences(input_texts)
    decoder_input_data = target_tokenizer.texts_to_sequences(["\t" + txt for txt in target_texts])
    decoder_target_data = target_tokenizer.texts_to_sequences([txt + "\n" for txt in target_texts])

    encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_len, padding='post')
    decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_len, padding='post')
    decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_len, padding='post')

    return encoder_input_data, decoder_input_data, decoder_target_data


# Build Seq2Seq model
def build_seq2seq_model(config, vocab_in_size, vocab_out_size):
    embedding_dim = config["embedding_dim"]
    hidden_units = config["hidden_units"]
    cell_type = config["cell_type"]
    num_layers = config["num_layers"]

    # Encoder
    encoder_inputs = Input(shape=(None,))
    x = Embedding(vocab_in_size, embedding_dim)(encoder_inputs)
    for _ in range(num_layers):
        if cell_type == "LSTM":
            x, state_h, state_c = LSTM(hidden_units, return_state=True, return_sequences=False)(x)
            encoder_states = [state_h, state_c]
        elif cell_type == "GRU":
            x, state_h = GRU(hidden_units, return_state=True, return_sequences=False)(x)
            encoder_states = [state_h]
        else:
            x, state_h = SimpleRNN(hidden_units, return_state=True, return_sequences=False)(x)
            encoder_states = [state_h]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    y = Embedding(vocab_out_size, embedding_dim)(decoder_inputs)
    if cell_type == "LSTM":
        decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(y, initial_state=encoder_states)
    elif cell_type == "GRU":
        decoder_gru = GRU(hidden_units, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_gru(y, initial_state=encoder_states)
    else:
        decoder_rnn = SimpleRNN(hidden_units, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_rnn(y, initial_state=encoder_states)

    decoder_dense = Dense(vocab_out_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


# GPT2 Fine-tuning
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def finetune_gpt2(data_path, output_dir):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

