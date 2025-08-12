# Modular-NeuralNet-FromScratch

Highly customizable Multilayer Perceptron (MLP) built entirely with NumPy.  
Define any architecture, choose activations and loss functions, and apply L2 regularization to control overfitting.

---

##  Features
- Customizable network architecture: layers, hidden sizes, activations
- Activation functions: Linear, Sigmoid, RELU, Softmax
- Loss functions: MSE, Binary/Categorical Cross-Entropy  
- L2 regularization with user-defined λ  
- Pure NumPy implementation — no TensorFlow/PyTorch  
- Tools for predictions, parameter extraction, and model evaluation

---

## Evaluation & Experimentation
- Verify training via train/test MSE.
- Tweak hyperparameters: alpha (learning rate), lambda (regularization), and layer dimensions.
- Ensure your continuous data inputs and target y are properly normalized (mean 0, std 1) to prevent exploding costs.
- Full performance walkthroughs are in demo.ipynb.
- Complete code for the model is in nn_constructor.py
  
---

## Dataset for demo.ipynb
https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction/data

---
## Model Example
```python
import numpy as np
from nn_constructor import NeuralNetwork, Layer, ReluAct, LinearAct, MeanRegressionError

# Sample data
X = np.random.randn(20, 500)  # (features, samples)
y = np.random.randn(1, 500) * 100 + 200  # regression target

# create Layers
layers = [
    Layer(20, 16, ReluAct()),
    Layer(16, 12, ReluAct()),
    Layer(12, 1, LinearAct())
]

# making model
loss = MeanRegressionError(m=X.shape[1], y=y, lamda=0.01, layers=layers)
nn = NeuralNetwork(alpha=0.005, data=X, loss=loss, layers=layers)

#training model and making predictions owith trained model
nn.fit()
print("Predictions:", nn.predict(X)[:, :5])


