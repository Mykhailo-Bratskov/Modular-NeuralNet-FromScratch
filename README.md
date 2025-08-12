# Modular-NeuralNet-FromScratch
This Modular-NeuralNet-FromScratch is a highly customizable Multilayer Perceptron. The network can take different forms as the number of layers and the number of nodes within each layer can be adjusted to a specific purpose. Additionally, the network involves L2 regularization to prevent overfitting, where the user can define lambda themselves. 

# Key Features
- Fully customizable architecture: choose layers, sizes, and activations
- Supports multiple loss functions: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- L2 regularization with user-defined lambda
- Pure NumPy — no external deep learning frameworks
- Parameter extraction and prediction methods for evaluation

# Performance Notes
- On small/artificial datasets, the model performs well and converges quickly.
  
- On real-world datasets (e.g., Kaggle housing prices), results are more sensitive to:

          - Learning rate (alpha)
          - Regularization strength (lambda)
          - Model size (number of layers and neurons)
          - Proper feature scaling (normalization or standardization)

- Without L2, the model can overfit quickly; L2 helps keep weights small and improves generalization.
# Usage Example 
## Example data

X = np.random.randn(20, 1000)  # 20 features, 1000 samples

y = np.random.randn(1, 1000)   # regression target

## Define layers

layers = [

    Layer(in_features=20, out_features=12, activation=ReluAct()),
    
    Layer(in_features=12, out_features=8, activation=ReluAct()),
    
    Layer(in_features=8, out_features=1, activation=LinearAct()),
]

## Initialize loss with L2 regularization

loss = MeanRegressionError(m=X.shape[1], y=y, lamda=0.01, layers=layers)

## Build and train network
nn = NeuralNetwork(alpha=0.01, data=X, loss=loss, layers=layers)

nn.fit()

## Make predictions
predictions = nn.predict(X)

## Evaluation 
nn.meansquarederror(prediction, y_test) 

## See neuralnetwork_check.ipynb for example training and testing runs.

# Dataset
The real-world testing was performed on the Kaggle dataset:
"Housing Price Prediction" by Harish Kumardatalab
Link: https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction/data

# License
This project is licensed under the MIT License.

