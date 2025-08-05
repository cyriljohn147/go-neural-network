# Go Neural Network from Scratch

A simple neural network implementation in Go that demonstrates the fundamentals of artificial neural networks, including forward propagation, backpropagation, and training. This project successfully solves the XOR problem, which is a classic non-linearly separable problem requiring a hidden layer.

## 🎯 Features

- **Pure Go Implementation**: No external dependencies - built entirely from scratch
- **Backpropagation Training**: Implements the gradient descent algorithm for learning
- **XOR Problem Solver**: Successfully learns the XOR logical function
- **Configurable Architecture**: Currently supports 2-2-1 architecture (2 inputs, 2 hidden, 1 output)
- **Loss Tracking**: Monitors training progress with mean squared error
- **Sigmoid Activation**: Uses sigmoid activation function with derivative for backpropagation

## 🏗️ Architecture

```
Input Layer (2 nodes) → Hidden Layer (2 nodes) → Output Layer (1 node)
```

The network uses:
- **Activation Function**: Sigmoid (σ(x) = 1/(1 + e^(-x)))
- **Loss Function**: Mean Squared Error (MSE)
- **Learning Algorithm**: Gradient Descent with Backpropagation
- **Learning Rate**: 0.5

## 📊 Performance

Training on the XOR dataset:
- **Initial Loss**: 0.149030
- **Final Loss**: 0.000188 (after 10,000 epochs)
- **Accuracy**: 98%+ on all XOR patterns

### XOR Results:
| Input | Target | Output | Accuracy |
|-------|--------|--------|----------|
| [0,0] | 0      | 0.0197 | ✅ 98.0% |
| [0,1] | 1      | 0.9828 | ✅ 98.3% |
| [1,0] | 1      | 0.9827 | ✅ 98.3% |
| [1,1] | 0      | 0.0181 | ✅ 98.2% |

## 🚀 Quick Start

### Prerequisites
- Go 1.23.4 or later

### Installation & Running
```bash
# Clone the repository
git clone https://github.com/cyriljohn147/go-neural-network.git
cd go-neural-network

# Run the neural network
go run .
```

### Expected Output
```
Neural Net from Scratch - Now with Training!
Training the network...
Epoch 0, Average Loss: 0.149030
Epoch 1000, Average Loss: 0.097772
Epoch 2000, Average Loss: 0.006698
...
Epoch 9000, Average Loss: 0.000188

Testing trained network:
Input: [0 0], Target: 0, Output: 0.0197
Input: [0 1], Target: 1, Output: 0.9828
Input: [1 0], Target: 1, Output: 0.9827
Input: [1 1], Target: 0, Output: 0.0181
```

## 📁 Project Structure

```
go-neural-network/
├── main.go          # Main neural network implementation
├── intial.go        # Utility functions (matrix/vector operations)
├── go.mod           # Go module file
└── README.md        # This file
```

## 🧠 How It Works

### 1. **Forward Propagation**
```go
// Calculate hidden layer activations
for i := range hiddenLayer {
    sum := dotProduct(weightsInputHidden[i], inputs) + biasesHidden[i]
    hiddenLayer[i] = sigmoid(sum)
}

// Calculate output
outputSum := dotProduct(weightsHiddenOutput[0], hiddenLayer) + biasesOutput[0]
output := sigmoid(outputSum)
```

### 2. **Backpropagation**
```go
// Calculate errors and update weights
outputError := target - output
outputDelta := outputError * sigmoidDerivative(output)

// Update weights using gradient descent
weightsHiddenOutput[0][i] += learningRate * outputDelta * hiddenLayer[i]
```

### 3. **Training Loop**
The network trains for 10,000 epochs, processing all training examples in each epoch and adjusting weights to minimize error.

## 🔧 Key Functions

| Function | Purpose |
|----------|---------|
| `forwardPass()` | Makes predictions using current weights |
| `forwardPassWithHidden()` | Forward pass that returns hidden layer values |
| `backpropagate()` | Updates weights using gradient descent |
| `sigmoid()` | Activation function |
| `sigmoidDerivative()` | Derivative for backpropagation |
| `meanSquaredError()` | Loss function |
| `dotProduct()` | Vector dot product calculation |

## 📈 Learning Curve

The network's loss decreases exponentially during training:
- Epochs 0-1000: Rapid initial learning
- Epochs 1000-3000: Significant improvement
- Epochs 3000-10000: Fine-tuning and convergence

## 🎓 Educational Value

This project demonstrates:
- **Neural Network Fundamentals**: Forward/backward propagation
- **Gradient Descent**: How networks learn from errors
- **Non-linear Problem Solving**: XOR requires hidden layers
- **Go Programming**: Slices, functions, and mathematical operations
- **Machine Learning Concepts**: Training, epochs, loss functions

## 🔮 Future Enhancements

Potential improvements for this project:
- [ ] Configurable network architecture (variable layers/nodes)
- [ ] Multiple activation functions (ReLU, tanh, etc.)
- [ ] Different optimizers (Adam, RMSprop)
- [ ] Batch training support
- [ ] Cross-validation
- [ ] More complex datasets
- [ ] Network serialization/deserialization
- [ ] Visualization of training progress

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!
