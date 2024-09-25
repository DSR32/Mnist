# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:07:06 2024

@author: divya

This script intends to build a deep-learning model from scratch to classify the MNIST Data.
A Multi-Layer Perceptron (MLP) is a feedforward artificial neural network consisting of multiple layers: 
an input layer, one or more hidden layers, and an output layer. Each layer is fully connected to the next,
and information flows from the input to the output. MLPs are typically used for supervised learning tasks like classification.

In the context of our MLP classifier for the MNIST dataset, which consists of 28x28 grayscale images of digits (0–9),
the goal is to classify each image into one of 10 possible categories (digits).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

class MLPClassifier:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the MLP Classifier with input size, hidden layer size, and output size.
        
        We randomly initialize the weights and set biases to zero. 
        Small random weights help break symmetry, so neurons can learn different features.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Random initialization of weights using a normal distribution (helps with symmetry breaking)
        np.random.seed(42)  # Seed for reproducibility - It can be an random integer.
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1. / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))  # Bias initialized to 0
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1. / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))  # Bias initialized to 0

    def relu(self, Z):
        """
        ReLU activation function. ReLU helps introduce non-linearity into the model,
        allowing it to learn more complex functions.
        
        ReLU simply returns 0 for negative inputs and the input itself for positive values.
        """
        return np.maximum(0, Z)

    def softmax(self, Z):
        """
        Softmax activation function. It converts raw output scores into probabilities that sum up to 1. 
        This is why we use softmax in the output layer for classification tasks.
        """
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Numerical stability trick
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    def forward_propagation(self, X):
        """
        Forward Propagation:
        
        1. Compute the linear combination Z1 = X * W1 + b1
        2. Apply ReLU activation to get A1
        3. Compute the linear combination Z2 = A1 * W2 + b2
        4. Apply softmax activation to get the final output A2 (probabilities)
        """
        # Step 1: Compute hidden layer's linear output
        Z1 = np.dot(X, self.W1) + self.b1
        # Step 2: Apply ReLU to introduce non-linearity
        A1 = self.relu(Z1)
        # Step 3: Compute the output layer's linear output
        Z2 = np.dot(A1, self.W2) + self.b2
        # Step 4: Apply softmax to get probabilities
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def compute_loss(self, Y_true, Y_pred):
        """
        Cross-Entropy Loss:
        Measures how well the predicted probabilities (Y_pred) match the true labels (Y_true).
        A lower cross-entropy loss means better predictions.
        """
        m = Y_true.shape[0]  # Number of training examples
        # To avoid log(0), we add a small constant 1e-8 for numerical stability.
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m
        return loss

    def backpropagation(self, X, Y_true, Z1, A1, Z2, A2):
        """
        Backpropagation:

        We calculate the gradients of the loss with respect to each parameter (W1, b1, W2, b2).
        This helps us figure out how to adjust the weights and biases to minimize the loss.
        
        We go backwards from the output to the input, applying the chain rule to compute these gradients.
        """
        m = X.shape[0]  # Number of examples

        # Step 1: Error at output layer (how far our prediction is from the actual value)
        dZ2 = A2 - Y_true
        # Step 2: Gradients for W2 and b2
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Step 3: Error at hidden layer (backpropagate the error to the hidden layer)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (Z1 > 0)  # Derivative of ReLU: 1 if Z1 > 0, else 0

        # Step 4: Gradients for W1 and b1
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        """
        Gradient Descent:
        We update the weights and biases in the direction that reduces the loss.
        The learning rate controls the step size in each update.
        """
        # Update weights and biases using the learning rate
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X_train, y_train, epochs=1000):
        """
        Training Loop:

        For each epoch, we:
        1. Perform forward propagation to get predictions.
        2. Compute the loss to evaluate how well our model is performing.
        3. Perform backpropagation to compute gradients.
        4. Update parameters (weights and biases) using gradient descent.
        """
        for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
            # Step 1: Forward propagation
            Z1, A1, Z2, A2 = self.forward_propagation(X_train)

            # Step 2: Compute loss for this epoch
            loss = self.compute_loss(y_train, A2)

            # Step 3: Backpropagation (compute gradients)
            dW1, db1, dW2, db2 = self.backpropagation(X_train, y_train, Z1, A1, Z2, A2)

            # Step 4: Update parameters
            self.update_parameters(dW1, db1, dW2, db2)

            # Logging the loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        """
        Prediction:
        After training, we can use the model to predict the class labels for unseen data.
        The predicted class is the one with the highest probability.
        """
        # Forward propagate to get probabilities
        _, _, _, A2 = self.forward_propagation(X)
        # Return the index of the class with the highest probability
        return np.argmax(A2, axis=1)

    def accuracy(self, X, y_true):
        """
        Accuracy Calculation:
        The accuracy is the percentage of correct predictions.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == np.argmax(y_true, axis=1)) * 100
        return accuracy

def main():
    # Load and preprocess the MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Flatten the images from 28x28 to 784-dimensional vectors
    X_train = X_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype('float32') / 255.0
    
    # One-hot encode the labels
    one_hot = OneHotEncoder(sparse_output=False)
    y_train = one_hot.fit_transform(y_train.reshape(-1, 1))
    y_test = one_hot.transform(y_test.reshape(-1, 1))
    
    # Define the MLP architecture
    input_size = 784    # Each image is 28x28, so we have 784 input neurons
    hidden_size = 128   # Number of neurons in the hidden layer
    output_size = 10    # There are 10 classes (digits 0-9)
    
    # Create an instance of MLPClassifier
    mlp = MLPClassifier(input_size, hidden_size, output_size, learning_rate=0.01)
    
    # Train the MLP on the MNIST dataset
    mlp.train(X_train, y_train, epochs=1000)
    
    # Evaluate the MLP's accuracy on the training and test sets
    train_accuracy = mlp.accuracy(X_train, y_train)
    test_accuracy = mlp.accuracy(X_test, y_test)
    
    print(f"Training Accuracy: {train_accuracy}%")
    print(f"Test Accuracy: {test_accuracy}%")

if __name__ == "__main__":
    main()
