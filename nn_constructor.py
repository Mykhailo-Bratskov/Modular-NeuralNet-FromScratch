import numpy as np 

class Layer(): 
    def __init__(self, in_features, out_features, activation):
        # in_features is the number of features getting passed to the layer
        # out_features is the size of the vector (k, 1) 
        # activation is the function 
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation 
        self.weights = self.initial_weight()
        self.b = self.initial_b()


    def initial_weight(self): 
        # He initialization (good default for ReLU hidden layers)
        size_input = self.in_features
        size_of_layer = self.out_features
        weights_vector = np.random.randn(size_of_layer, size_input) * np.sqrt(2 / size_input)
        return weights_vector

    def initial_b(self): 
        # Start biases at zero
        size_of_layer = self.out_features
        b_vector = np.zeros((size_of_layer, 1))
        return b_vector
    
    def z(self, X): 
        z = np.matmul(self.weights, X) + self.b
        return z
    #the data is getting passed to each layer , node 
    def forwardpass(self, X): 
        self.X = X
        self.Z = self.z(X)
        self.A = self.activation.function(self.Z)
        return self.A
            
    def backward(self, dA, lamda = 0.0) : 
        # this method calculates the chain of derivatives for a specific Layer
        if isinstance(self.activation, SoftmaxAct):
            dZ =dA # softmax activation function is being due to its complexity
        else:  # the derivative will incure the misshape 
            dZ = dA * self.activation.derivative(self.Z)   # (out, m)

        m = self.X.shape[1]
        dW = np.matmul(dZ, self.X.T)                     # (out, in)
        if lamda:
            dW += (lamda / m) * self.weights           # L2 on weights only
        db = np.sum(dZ, axis=1, keepdims=True)          # (out, 1)
        dA_prev = np.matmul(self.weights.T,  dZ)             # (in, m)

        return dA_prev, dW, db
#activation  
class LinearAct(): 
    def function(self, z): 
        return z 
    def derivative(self, z): 
        return np.ones_like(z)
#activation      
class SigmoidAct(): 
    def function(self, z):
        return 1 / (1 + np.exp(-z))
    def derivative(self, z): 
        s = self.function(z)
        return s * (1 - s)
#activation     
class ReluAct(): 
    def function(self, z):
        return np.maximum(0, z)
    def derivative(self, z): 
        return np.where(z > 0, 1, 0)
    
class SoftmaxAct():
    def function(self, z):
        z_stable = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
# the main (central) class for loss computing. This class collects necessary variables to compute costs    
class Loss(): 
    def __init__(self, m, y, lamda): 
        self.lamda = lamda
        #self.yhat = yhat
        self.y = y 
        self.m = m
    
# The costs are regularized to prevent overfitting 
# for binary ourtput
class BinaryCrossEntropy(Loss):
    #Regularized 
    def cost_function(self, yhat, weights): 
        regularization = (self.lamda / (2 * self.m)) * sum(np.sum(i**2) for i in weights)
        eps = 1e-8
        loss = -self.y * np.log(yhat+ eps) - (1 - self.y) * np.log(1 - yhat + eps)
        return (np.sum(loss) / self.m) + regularization
    def cost_derivative(self, yhat): 
        #the derivative is slightly modified to avoid division by 0, consequently, an error
        eps = 1e-8
        loss_derivative = -(self.y / yhat + eps) + ((1 - self.y) / (1 - yhat + eps)) 
        return loss_derivative / self.m
# for linear output    
class MeanRegressionError(Loss): 
    #Regularized
    def cost_function(self, yhat, weights): 
        regularization =  (self.lamda / (2 * self.m)) * sum(np.sum(i**2) for i in weights)
        return np.sum((yhat - self.y)**2) / (2 * self.m) + regularization
    def cost_derivative(self, yhat):
        loss_derivative =  yhat - self.y
        return loss_derivative/self.m
# for softmax output
class CategoricalCrossEntropy(Loss): 
    #Regularized
    def cost_function(self, yhat, weights):
        regularization =  (self.lamda / (2 * self.m)) * sum(np.sum(i**2) for i in weights)
        eps = 1e-8
        log_likelihood = -np.log(yhat[np.argmax(self.y, axis=0), range(self.m)] + eps)
        return (np.sum(log_likelihood) / self.m) + regularization
    def cost_derivative(self, yhat): 
        loss_derivative = yhat - self.y 
        return loss_derivative/self.m 

# this is the class with main computations like compiling, predicting , etc. 
class NeuralNetwork():
    def __init__(self,alpha, data, loss, layers = None): 
        self.alpha = alpha
        self.data = data
        self.loss = loss 
        self.layers = list(layers) if layers is not None else []
    
    # the layers passing data to each other 
    def layer_connection(self): 
        lst = []
        first_hidden = self.layers[0].forwardpass(self.data)
        lst.append(first_hidden)
        for i in range(1, len(self.layers)):
            datapassing = self.layers[i].forwardpass(lst[i - 1])
            lst.append(datapassing)
        # the last item in list is the output layer
        return lst 
    # we save the output layer
    def output_layer(self): 
        output_layer = self.layer_connection()[-1]
        return output_layer
    # the costs are computed by calling the Loss class
    def cost(self): 
        output_layer = self.output_layer()
        # The way to save the weights for a particular iteration of fitting
        # Because cost function gets iterated in compiling, the weights get renewed as well, which enables the regularization 
        weights_lst = []
        lst_of_layers = self.layers
        for layer in lst_of_layers: 
            weights_lst.append(layer.weights)
        
        cost = self.loss.cost_function(output_layer, weights_lst)
        return cost
    
    # implementing gradient_descent
    def gradient_descent(self): 
        yhat = self.output_layer() # extracting the output layer

    # 2) Upstream gradient from the chosen loss (output layer only)
        dA = self.loss.cost_derivative(yhat)

    # 3 Backprop layer-by-layer (last → first), update after each backward
        for i in range(len(self.layers) - 1, -1, -1): # we start from (e.g from 2) instead of 3, with the limit of 0, and the step of -1
            dA, dW, db = self.layers[i].backward(dA, lamda=self.loss.lamda)
            self.layers[i].weights -= self.alpha * dW
            self.layers[i].b       -= self.alpha * db
    
    # the fit model is being rounded to 4 decimals after comma (as further precision improvement will not make a huge difference)
    #max_epochs to prevent training last hours or just too long
    def fit(self, max_epochs=1000):
        statement = True
        best_value = float('inf') 
        count = 0
        
        while statement and count < max_epochs:
            cost = np.round(self.cost(), 4)

            if cost < best_value:
                best_value = cost
                self.gradient_descent()
                count += 1
                self.print_loss_breakdown(count)
            #if we hit plateu, iterate 20 more times to see if the cost can be improved. If not -> stop
            else: 
                lst = []
                for i in range(20):
                    self.gradient_descent()
                    cost = np.round(self.cost(), 4)
                    self.print_loss_breakdown(count + i + 1)
                    lst.append(cost)
                if min(lst) < best_value:
                    statement = True
                else:
                    statement = False
                count += 20
        print("Training stopped.")
        
    # the parameters of compiled model will be organised and possible to extract and review 
    def get_params(self):
        params = []
        for layer in self.layers:
            params.append((layer.weights.copy(), layer.b.copy()))
        return params
    
    # the compiled model making prediction on a new data to see how well does model generalize
    def predict_raw(self, x):
        lst = []
        first_hidden = self.layers[0].forwardpass(x)
        lst.append(first_hidden)
        for i in range(1, len(self.layers)):
            datapassing = self.layers[i].forwardpass(lst[i - 1])
            lst.append(datapassing)
        output_layer = lst[-1]
        return output_layer

    # predictions are adjusted to the output activations as linear use different from linear
    def predict_classes(self, X):
        yhat = self.predict_raw(X)

        # Binary classification
        if yhat.shape[0] == 1 and isinstance(self.loss, BinaryCrossEntropy):
            return (yhat >= 0.5).astype(int)

        # Multiclass classification
        if isinstance(self.loss, CategoricalCrossEntropy):
            return np.argmax(yhat, axis=0)

        raise ValueError("predict_classes() is only for classification tasks.")

    def predict(self, X):
        # For regression or generic use
        return self.predict_raw(X)
    
    # these two last methods are simply used to check how accurate the predictions are on new data. 
    # They are greatly simplified and do not use the most optimized mathematics to obtain best possible results. 
    # Anyways, for simple models they are more than sufficient 
 
    # to check the costs for the test linear data 
    def meansquarederror(self, x , y): 
        m = x.size
        sumation = np.sum((x - y)**2)
        return sumation/m
    # to check the costs for the test binary or categorical data (simplified)
    def binaryerror(self, prediction, y): 
        count = np.sum(prediction == y)
        accuracy = (count / len(y)) * 100
        return f"The accuracy is {accuracy:.2f}%"
    # computing mean squared error (more modular to separate in several functions instead of one)
    def mse_loss(self, yhat, y):
        yhat = np.asarray(yhat)
        y = np.asarray(y)
        return np.mean((yhat - y) ** 2)
    # this is an addition to computing costs, as the code implements regularization
    # making it separate from main loss computing functions for error tractability reasons
    def l2_reg_loss(self, m=None):
        # m = number of samples in current batch/dataset
        if m is None:
            m = self.data.shape[1]

        lam = getattr(self.loss, "lamda", 0.0)
        if lam is None or lam == 0:
            return 0.0

        weight_sum = 0.0
        for layer in self.layers:
            # adapt attribute name if you use layer.w instead of layer.weights
            W = layer.weights
            weight_sum += np.sum(W ** 2)

        return (lam / (2 * m)) * weight_sum

    # adding all the costs for the neural network
    def total_loss(self, yhat, y, m=None):
        data_loss = self.mse_loss(yhat, y)
        reg_loss = self.l2_reg_loss(m=m)
        return data_loss + reg_loss, data_loss, reg_loss
    
    # more detailed breakdown of training for tracing errors in training 
    def print_loss_breakdown(self, epoch_num):
        yhat = self.forward_propagation(self.data) if hasattr(self, "forward_propagation") else self.output_layer()
        y_true = self.loss.y
        m = self.data.shape[1]

        total, data_loss, reg_loss = self.total_loss(yhat, y_true, m=m)

        print(
            f"epoch {epoch_num} - "
            f"data_loss(MSE): {data_loss:.6f} | "
            f"reg_loss: {reg_loss:.6f} | "
            f"total: {total:.6f}"
        )