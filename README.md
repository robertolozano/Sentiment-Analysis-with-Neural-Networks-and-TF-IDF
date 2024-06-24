# Sentiment Analysis with Neural Networks and TF-IDF

This project focuses on sentiment analysis using a neural network model implemented in PyTorch, leveraging TF-IDF vectorization for text feature extraction.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Examples](#examples)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Validation](#training-and-validation)
- [Prediction](#prediction)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates an end-to-end workflow of sentiment analysis, from text preprocessing to model training, evaluation, and prediction. It provides insights into how neural networks can be effectively used for natural language processing tasks.

## Dataset

- **Training Data:** `x_train.csv` and `y_train.csv`
- **Test Data:** `x_test.csv`
- The text data is preprocessed by converting it to lowercase and then transformed into TF-IDF vectors for feature extraction.
- The dataset consists of reviews from websites such as Amazon, Yelp, and IMDb.
  - Examples:
    - **"There is no plot here to keep you going in the first place."**
    - **"Of all the dishes, the salmon was the best, but all were great."**

# Examples

**Examples of the sentiment analysis predictions:**

- Negative Sentiment

  - Input: "There is no plot here to keep you going in the first place."
  - Output: 0

- Positive Sentiment
  - Input: "Of all the dishes, the salmon was the best, but all were great."
  - Output: 1

## Requirements

- Python 3.6+
- PyTorch
- scikit-learn
- pandas
- numpy

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository

```python
git clone https://github.com/yourusername/sentiment-analysis-tfidf-nn.git
cd sentiment-analysis-tfidf-nn
```

2. Prepare your data:
   Ensure that x_train.csv, y_train.csv, and x_test.csv are in the project directory.

3. Run the script

```python
python sentiment_analysis.py
```

## Model Architecture

The neural network model consists of:

Two hidden layers with ReLU activations and dropout regularization.
An output layer with a sigmoid activation function for binary classification.

## Training and Validation

The model is trained using K-Fold Cross-Validation with 5 splits.
The training process includes:
Loading and preparing text data.
Transforming text data into TF-IDF vectors.
Defining the neural network architecture.
Training the model with Adam optimizer and BCELoss.
Evaluating model performance using validation accuracy.

## Prediction

After training, the model is used to predict sentiments on the test data.
The predictions are saved to y_prediction.txt.
The model will predict **1** if the sentiment is positive and **0** if the sentiment is negative.

## Results

The model's performance is evaluated using the average validation accuracy across all folds.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
