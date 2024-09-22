#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:02:36 2024

@author: divya
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import argparse

# Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing the data: normalize and one-hot encode labels
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Global variables for input shape
input_shape = (28, 28)
input_shape_cnn = (28, 28, 1)

# Plot accuracy function
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# MLP Model
def build_mlp():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# CNN Model
def build_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape_cnn))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# RNN with LSTM Model
def build_rnn():
    model = models.Sequential()
    model.add(layers.LSTM(128, input_shape=input_shape))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Transformer Encoder function
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    return x + res

# Transformer Model
def build_transformer():
    inputs = layers.Input(shape=(28, 28))
    x = transformer_encoder(inputs, head_size=256, num_heads=4, ff_dim=256, dropout=0.2)
    x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=256, dropout=0.2)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

# Compile and Train the Model
def compile_and_train(model, x_train, y_train, x_test, y_test, epochs=10):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_data=(x_test, y_test))
    return history

def main(args):


     # Function to train and evaluate a specific model, return evaluation accuracy
    def train_and_plot(model_name, model_func, x_train, y_train, x_test, y_test):
        print(f"Training {model_name}...")
        model = model_func()
        history = compile_and_train(model, x_train, y_train, x_test, y_test, epochs=args.epochs)
        plot_accuracy(history)

        # Evaluate the model on the test set
        loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}\n")

        # Return test accuracy
        return test_accuracy

    if args.model == 'mlp':
        train_and_plot('MLP', build_mlp, x_train, y_train, x_test, y_test)
    
    elif args.model == 'cnn':
        # Reshape for CNN (add channel dimension)
        x_train_cnn = x_train.reshape(-1, 28, 28, 1)
        x_test_cnn = x_test.reshape(-1, 28, 28, 1)
        train_and_plot('CNN', build_cnn, x_train_cnn, y_train, x_test_cnn, y_test)

    elif args.model == 'rnn':
        train_and_plot('RNN (LSTM)', build_rnn, x_train, y_train, x_test, y_test)
    
    elif args.model == 'transformer':
        train_and_plot('Transformer', build_transformer, x_train, y_train, x_test, y_test)

    # If 'all' is chosen, we train all models and track their test accuracies
    elif args.model == 'all':
        results = {}  # Dictionary to store results in the format {model_name: test_accuracy}

        # Train and evaluate MLP
        results['MLP'] = train_and_plot('MLP', build_mlp, x_train, y_train, x_test, y_test)

        # Train and evaluate CNN (reshape input)
        x_train_cnn = x_train.reshape(-1, 28, 28, 1)
        x_test_cnn = x_test.reshape(-1, 28, 28, 1)
        results['CNN'] = train_and_plot('CNN', build_cnn, x_train_cnn, y_train, x_test_cnn, y_test)

        # Train and evaluate RNN (LSTM)
        results['RNN (LSTM)'] = train_and_plot('RNN (LSTM)', build_rnn, x_train, y_train, x_test, y_test)

        # Train and evaluate Transformer
        results['Transformer'] = train_and_plot('Transformer', build_transformer, x_train, y_train, x_test, y_test)

        # Find the model with the minimum and maximum test accuracy
        model_with_min_acc = min(results, key=results.get)  # Model with the lowest test accuracy
        model_with_max_acc = max(results, key=results.get)  # Model with the highest test accuracy

        # Print out the models with min/max test accuracies
        summary = (
            f"\n--- Summary ---\n"
            f"Model with Minimum Test Accuracy: {model_with_min_acc} - {results[model_with_min_acc]:.4f}\n"
            f"Model with Maximum Test Accuracy: {model_with_max_acc} - {results[model_with_max_acc]:.4f}\n"
        )

        # Print the summary
        print(summary)

        # Save the summary to a text file
        with open("output.txt", "w") as f:
            f.write(summary)
        
if __name__ == "__main__":
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Train different models on the MNIST dataset")
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'cnn', 'rnn', 'transformer', 'all'], 
                        help="Choose the model to train: 'mlp', 'cnn', 'rnn', 'transformer', or 'all'")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model")
    args = parser.parse_args()
    main(args)
