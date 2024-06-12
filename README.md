# Hierarchical Transformer and CNN Models for Time-Series Forecasting

## Overview
This repository contains various Python scripts for implementing and evaluating different machine learning models for time-series forecasting. The models include Convolutional Neural Networks (CNN), Long Short-Term Memory networks (LSTM), and Transformer models. The scripts are designed to forecast data at different time intervals: 10 minutes, 30 minutes, and 60 minutes.

## Repository Structure
- `Auto Data.py`: Script for data preprocessing and automated data handling.
- `CNN10 Minutes.py`: CNN model for 10-minute interval forecasting.
- `CNN30 Minutes.py`: CNN model for 30-minute interval forecasting.
- `CNN60 Minutes.py`: CNN model for 60-minute interval forecasting.
- `Hierarchal Transformer-Auto 10 Minutes-CNN.py`: Hierarchical Transformer-Auto model with CNN for 10-minute interval forecasting.
- `Hierarchal Transformer-Auto 30 Minutes-CNN.py`: Hierarchical Transformer-Auto model with CNN for 30-minute interval forecasting.
- `Hierarchal Transformer-Auto 60 Minutes-CNN.py`: Hierarchical Transformer-Auto model with CNN for 60-minute interval forecasting.
- `LSTM-Simple 10 Minutes.py`: Simple LSTM model for 10-minute interval forecasting.
- `LSTM-Simple 30 Minutes.py`: Simple LSTM model for 30-minute interval forecasting.
- `LSTM-Simple 60 Minutes.py`: Simple LSTM model for 60-minute interval forecasting.
- `Transformer-Auto 10 Minutes.py`: Transformer-Auto model for 10-minute interval forecasting.
- `Transformer-Auto 30 Minutes.py`: Transformer-Auto model for 30-minute interval forecasting.
- `Transformer-Auto 60 Minutes.py`: Transformer-Auto model for 60-minute interval forecasting.
- `Transformer-Autoencoder-ED.py`: Transformer model with autoencoder for Encoder-Decoder structure.
- `Transformer-Simple 10 Minutes.py`: Simple Transformer model for 10-minute interval forecasting.
- `Transformer-Simple 30 Minutes.py`: Simple Transformer model for 30-minute interval forecasting.
- `Transformer-Simple 60 Minutes.py`: Simple Transformer model for 60-minute interval forecasting.

## Getting Started

### Prerequisites
Ensure you have the following libraries installed:
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Plotly

You can install them using pip:
```bash
pip install torch numpy pandas scikit-learn matplotlib plotly
```

### Running the Models
1. **Data Preparation**: Use `Auto Data.py` to preprocess your dataset.
2. **Training the Models**: Run the respective Python script for the model and time interval you wish to train. For example, to train a CNN model for 10-minute interval forecasting, use:
    ```bash
    python CNN10\ Minutes.py
    ```
3. **Evaluation**: Each script will train the model and provide evaluation metrics on the test set.

### Sample Code Explanation

The following is a brief explanation of the key components in the sample code:

1. **Library Imports**:
    - The code starts by importing all necessary libraries, including PyTorch, NumPy, Pandas, and others for data manipulation, model building, and visualization.

2. **Seed Initialization**:
    - A function `reset_random_seeds` is defined to ensure reproducibility by setting the seed for random number generators.

3. **Data Loading**:
    - Data is loaded from a pickle file using `pickle.load`, and then it is split into training and testing sets.

4. **Normalization Function**:
    - Functions `reconstructData` and `norm` are defined to preprocess and normalize the data.

5. **Model Definition**:
    - Various classes are defined to build the models, including `PositionalEmbedding`, `MultiHeadAttention`, `TransformerBlock`, `TransformerEncoder`, `LSTMModel`, `Classifier`, `HierarchicalTransformerEncoder`, and `HierarchicalTransformerDecoder`.

6. **Training Loop**:
    - A training loop is defined to train the models. It includes forward passes, loss calculation, backpropagation, and evaluation.

7. **Evaluation**:
    - The models are evaluated on the validation and test sets, and performance metrics like accuracy, precision, recall, F1 score, and AUC are printed.

## Models

### CNN Models
- `CNN10 Minutes.py`
- `CNN30 Minutes.py`
- `CNN60 Minutes.py`

### LSTM Models
- `LSTM-Simple 10 Minutes.py`
- `LSTM-Simple 30 Minutes.py`
- `LSTM-Simple 60 Minutes.py`

### Transformer Models
- `Transformer-Simple 10 Minutes.py`
- `Transformer-Simple 30 Minutes.py`
- `Transformer-Simple 60 Minutes.py`
- `Transformer-Auto 10 Minutes.py`
- `Transformer-Auto 30 Minutes.py`
- `Transformer-Auto 60 Minutes.py`
- `Transformer-Autoencoder-ED.py`

### Hierarchical Transformer Models with CNN
- `Hierarchal Transformer-Auto 10 Minutes-CNN.py`
- `Hierarchal Transformer-Auto 30 Minutes-CNN.py`
- `Hierarchal Transformer-Auto 60 Minutes-CNN.py`

## Contributing
Feel free to contribute by submitting a pull request. Please ensure your changes are well-documented and tested.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or inquiries, please contact bluecodeindia@gmail.com.
