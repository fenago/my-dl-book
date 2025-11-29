# From Neurons to Networks: Understanding the Building Blocks

## Introduction

Deep learning has revolutionized artificial intelligence, powering everything from voice assistants to self-driving cars. But before we dive into complex neural networks, we need to understand their fundamental building blocks. This chapter explores how biological neurons inspired artificial neural networks and introduces the mathematical foundations that make deep learning possible.

:::{tip}
As you read through this chapter, have a Python environment ready to experiment with the code examples. Understanding comes from doing!
:::

## The Biological Inspiration

### How Biological Neurons Work

The human brain contains approximately 86 billion neurons, each connected to thousands of other neurons. These neurons communicate through electrical and chemical signals, forming the basis of all our thoughts, memories, and actions.

```{mermaid}
flowchart LR
    A[Dendrites] --> B[Cell Body/Soma]
    B --> C[Axon]
    C --> D[Synapses]
    D --> E[Other Neurons]
```

A biological neuron consists of several key components:

1. **Dendrites**: Receive signals from other neurons
2. **Cell Body (Soma)**: Processes incoming signals
3. **Axon**: Transmits output signals to other neurons
4. **Synapses**: Connection points where signals pass between neurons

The neuron fires (sends a signal) only when the accumulated input signals exceed a certain threshold. This "all-or-nothing" behavior inspired the first artificial neurons.

:::{note}
While artificial neural networks are inspired by biological neurons, they are highly simplified models. Real biological neurons are far more complex and not fully understood.
:::

### From Biology to Mathematics

Warren McCulloch and Walter Pitts created the first mathematical model of a neuron in 1943. Their model captured the essential idea: a neuron receives multiple inputs, processes them, and produces an output based on whether the combined signal exceeds a threshold.

This simple idea forms the foundation of all modern neural networks.

## The Perceptron: The First Artificial Neuron

### Mathematical Foundation

The perceptron, introduced by Frank Rosenblatt in 1958, was the first practical artificial neuron. It takes multiple inputs, multiplies each by a weight, sums them up, and applies a decision rule.

Mathematically, a perceptron computes:

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$

Where:
- $x_i$ are the input values
- $w_i$ are the weights (importance of each input)
- $b$ is the bias term (shifts the decision boundary)
- $f$ is the activation function
- $y$ is the output

:::{table} Perceptron Components
:name: tbl-perceptron-components

| Component | Symbol | Description |
|-----------|--------|-------------|
| Inputs | $x_1, x_2, ..., x_n$ | Features or data points |
| Weights | $w_1, w_2, ..., w_n$ | Learned parameters |
| Bias | $b$ | Offset term |
| Activation | $f$ | Non-linear function |
| Output | $y$ | Prediction |
:::

### Implementing a Simple Perceptron

Let's implement a basic perceptron from scratch in Python:

```python
import numpy as np

class Perceptron:
    def __init__(self, num_inputs):
        """Initialize perceptron with random weights and bias."""
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def activate(self, x):
        """Step activation function."""
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        """Make a prediction for given inputs."""
        # Calculate weighted sum
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply activation function
        return self.activate(weighted_sum)
    
    def train(self, training_inputs, labels, learning_rate=0.1, epochs=100):
        """Train the perceptron using the perceptron learning rule."""
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # Update weights based on error
                error = label - prediction
                self.weights += learning_rate * error * inputs
                self.bias += learning_rate * error

# Example: Learning the AND function
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1])

perceptron = Perceptron(num_inputs=2)
perceptron.train(X_train, y_train)

# Test the trained perceptron
print("Testing AND function:")
for inputs, label in zip(X_train, y_train):
    prediction = perceptron.predict(inputs)
    print(f"Input: {inputs}, Expected: {label}, Predicted: {prediction}")
```

:::{dropdown} Understanding the Code
The `Perceptron` class implements three key methods:
- `__init__`: Initializes random weights and bias
- `predict`: Computes the weighted sum and applies the activation function
- `train`: Updates weights using the perceptron learning rule

The perceptron learning rule is simple: if the prediction is wrong, adjust the weights in the direction that would have made the prediction correct.
:::

### Limitations of the Perceptron

In 1969, Marvin Minsky and Seymour Papert published a book highlighting a critical limitation: a single perceptron cannot learn non-linearly separable functions.

:::{exercise}
:label: ex-xor

Try to modify the perceptron code above to learn the XOR function:
- Inputs: [0,0], [0,1], [1,0], [1,1]
- Outputs: 0, 1, 1, 0

What happens? Why does it fail?
:::

:::{solution} ex-xor
:class: dropdown

```python
# XOR function - a single perceptron will fail
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

perceptron_xor = Perceptron(num_inputs=2)
perceptron_xor.train(X_xor, y_xor, epochs=1000)

print("Testing XOR function:")
correct = 0
for inputs, label in zip(X_xor, y_xor):
    prediction = perceptron_xor.predict(inputs)
    print(f"Input: {inputs}, Expected: {label}, Predicted: {prediction}")
    if prediction == label:
        correct += 1

print(f"Accuracy: {correct}/{len(y_xor)}")
```

The perceptron fails because XOR is not linearly separable. You cannot draw a single straight line that separates the positive examples (0,1 and 1,0) from the negative examples (0,0 and 1,1). This limitation led to the development of multi-layer networks.
:::

## Activation Functions: Adding Non-Linearity

### Why Activation Functions Matter

Activation functions introduce non-linearity into neural networks. Without them, no matter how many layers you stack, the network would still only compute linear transformations—essentially just a complicated way of doing linear regression.

:::{warning}
A neural network without non-linear activation functions can only learn linear relationships, severely limiting its power!
:::

### Common Activation Functions

Let's explore the most important activation functions used in modern deep learning:

::::{tab-set}

:::{tab-item} Step Function
The simplest activation function, used in the original perceptron:

$$
f(x) = \begin{cases} 
1 & \text{if } x \geq 0 \\
0 & \text{if } x < 0
\end{cases}
$$

```python
def step_function(x):
    return np.where(x >= 0, 1, 0)
```

**Pros**: Simple and interpretable  
**Cons**: Not differentiable, cannot be used for gradient-based learning
:::

:::{tab-item} Sigmoid
Maps inputs to a range between 0 and 1:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

**Pros**: Smooth, differentiable, outputs can be interpreted as probabilities  
**Cons**: Vanishing gradient problem for large/small values
:::

:::{tab-item} Tanh
Maps inputs to a range between -1 and 1:

$$
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

```python
def tanh(x):
    return np.tanh(x)
```

**Pros**: Zero-centered, stronger gradients than sigmoid  
**Cons**: Still suffers from vanishing gradients
:::

:::{tab-item} ReLU
The most popular activation function in modern deep learning:

$$
f(x) = \max(0, x)
$$

```python
def relu(x):
    return np.maximum(0, x)
```

**Pros**: Computationally efficient, helps with vanishing gradient problem  
**Cons**: "Dying ReLU" problem when neurons get stuck outputting zero
:::

::::

### Visualizing Activation Functions

Let's visualize these activation functions to understand their behavior:

```python
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(-5, 5, 100)

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def step(x):
    return np.where(x >= 0, 1, 0)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, step(x))
axes[0, 0].set_title('Step Function')
axes[0, 0].grid(True)

axes[0, 1].plot(x, sigmoid(x))
axes[0, 1].set_title('Sigmoid')
axes[0, 1].grid(True)

axes[1, 0].plot(x, tanh(x))
axes[1, 0].set_title('Tanh')
axes[1, 0].grid(True)

axes[1, 1].plot(x, relu(x))
axes[1, 1].set_title('ReLU')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

:::{tip}
ReLU has become the default choice for hidden layers in most neural networks. For output layers, the choice depends on your problem: sigmoid for binary classification, softmax for multi-class classification, or linear for regression.
:::

## Building Multi-Layer Networks

### The Architecture of Neural Networks

To overcome the limitations of single perceptrons, we stack multiple layers of neurons together. This creates a **multi-layer perceptron (MLP)** or **feedforward neural network**.

```{mermaid}
graph LR
    subgraph Input Layer
        I1((x1))
        I2((x2))
        I3((x3))
    end
    
    subgraph Hidden Layer
        H1((h1))
        H2((h2))
        H3((h3))
        H4((h4))
    end
    
    subgraph Output Layer
        O1((y))
    end
    
    I1 --> H1
    I1 --> H2
    I1 --> H3
    I1 --> H4
    I2 --> H1
    I2 --> H2
    I2 --> H3
    I2 --> H4
    I3 --> H1
    I3 --> H2
    I3 --> H3
    I3 --> H4
    H1 --> O1
    H2 --> O1
    H3 --> O1
    H4 --> O1
```

A typical neural network consists of:

1. **Input Layer**: Receives the raw data
2. **Hidden Layer(s)**: Processes the data through learned representations
3. **Output Layer**: Produces the final prediction

:::{note}
"Deep learning" refers to neural networks with multiple hidden layers. The term "deep" comes from the depth (number of layers) of the network.
:::

### Forward Propagation

Forward propagation is the process of computing the output of a neural network given an input. The data flows forward through the network, layer by layer.

For a network with one hidden layer:

$$
\mathbf{h} = f_1(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
$$

$$
\mathbf{y} = f_2(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)
$$

Where:
- $\mathbf{x}$ is the input vector
- $\mathbf{W}_1, \mathbf{W}_2$ are weight matrices
- $\mathbf{b}_1, \mathbf{b}_2$ are bias vectors
- $f_1, f_2$ are activation functions
- $\mathbf{h}$ is the hidden layer output
- $\mathbf{y}$ is the final output

### Implementing a Multi-Layer Network

Let's implement a simple two-layer neural network:

```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize a two-layer neural network."""
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward propagation."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output, learning_rate):
        """Backward propagation."""
        m = X.shape[0]
        
        # Calculate gradients
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate):
        """Train the neural network."""
        losses = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss (Mean Squared Error)
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y, output, learning_rate)
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions."""
        return self.forward(X)

# Example: Learning XOR function with a neural network
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Create and train network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
losses = nn.train(X_xor, y_xor, epochs=10000, learning_rate=0.5)

# Test the network
print("\nTesting XOR function with Neural Network:")
predictions = nn.predict(X_xor)
for i, (inputs, label) in enumerate(zip(X_xor, y_xor)):
    pred = predictions[i][0]
    print(f"Input: {inputs}, Expected: {label[0]}, Predicted: {pred:.4f}, "
          f"Rounded: {round(pred)}")
```

:::{dropdown} Understanding Backward Propagation
Backward propagation (backprop) is the algorithm that enables neural networks to learn. It calculates how much each weight contributed to the error and adjusts them accordingly.

The key idea:
1. Calculate the error at the output
2. Propagate this error backward through the network
3. Update each weight proportionally to its contribution to the error

We'll dive deeper into the mathematics of backpropagation in Chapter 3.
:::

## Network Architecture Decisions

### How Many Layers?

Choosing the right architecture is more art than science. Here are some guidelines:

:::{table} Network Depth Guidelines
:name: tbl-network-depth

| Problem Type | Suggested Layers | Reasoning |
|--------------|------------------|-----------|
| Simple classification | 1-2 hidden layers | Sufficient for linearly separable or simple non-linear problems |
| Image recognition | 10-100+ layers | Need to learn hierarchical features |
| Natural language | 10-50 layers | Complex patterns in sequential data |
| Tabular data | 2-5 hidden layers | Usually sufficient for structured data |
:::

### How Many Neurons per Layer?

The number of neurons in each layer affects the network's capacity to learn:

- **Too few neurons**: The network may underfit and fail to learn complex patterns
- **Too many neurons**: The network may overfit, memorizing training data instead of learning general patterns

:::{tip}
Start with a reasonable number (e.g., 64 or 128 neurons) and adjust based on performance. Monitor both training and validation accuracy to detect overfitting.
:::

### Universal Approximation Theorem

A remarkable result in neural network theory states that a neural network with a single hidden layer containing enough neurons can approximate any continuous function to arbitrary accuracy.

:::{note}
While theoretically a single layer is sufficient, in practice, **deeper networks** (multiple layers with fewer neurons each) are more efficient and easier to train for complex problems.
:::

## Practical Considerations

### Weight Initialization

Proper weight initialization is crucial for training success:

```python
# Bad: All zeros
weights = np.zeros((input_size, hidden_size))

# Bad: Too large
weights = np.random.randn(input_size, hidden_size) * 10

# Good: Small random values (Xavier/Glorot initialization)
weights = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)

# Good: He initialization (for ReLU)
weights = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
```

:::{warning}
Initializing all weights to zero causes all neurons to learn the same features, severely limiting the network's power!
:::

### Choosing Learning Rate

The learning rate controls how much we adjust weights during training:

- **Too small**: Training is very slow
- **Too large**: Training may diverge or oscillate

```python
# Common learning rates to try
learning_rates = [0.001, 0.01, 0.1, 1.0]

# Often start with 0.01 or 0.001
```

## Exercises

:::{exercise}
:label: ex-activation-comparison

Modify the `NeuralNetwork` class to use ReLU activation in the hidden layer instead of sigmoid. Compare the training speed and final accuracy on the XOR problem.
:::

:::{solution} ex-activation-comparison
:class: dropdown

```python
class NeuralNetworkReLU(NeuralNetwork):
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function."""
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        """Forward propagation with ReLU."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # ReLU for hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)  # Sigmoid for output
        return self.a2
    
    def backward(self, X, y, output, learning_rate):
        """Backward propagation with ReLU."""
        m = X.shape[0]
        
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Train and compare
nn_relu = NeuralNetworkReLU(input_size=2, hidden_size=4, output_size=1)
losses_relu = nn_relu.train(X_xor, y_xor, epochs=10000, learning_rate=0.1)
```

ReLU often trains faster because it doesn't suffer from vanishing gradients as much as sigmoid.
:::

:::{exercise}
:label: ex-network-size

Create a function that trains neural networks with different hidden layer sizes (2, 4, 8, 16 neurons) on the XOR problem. Plot how the final loss varies with network size.
:::

:::{solution} ex-network-size
:class: dropdown

```python
def compare_network_sizes():
    sizes = [2, 4, 8, 16]
    final_losses = []
    
    for size in sizes:
        nn = NeuralNetwork(input_size=2, hidden_size=size, output_size=1)
        losses = nn.train(X_xor, y_xor, epochs=5000, learning_rate=0.5)
        final_losses.append(losses[-1])
        print(f"Hidden size: {size}, Final loss: {losses[-1]:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, final_losses, marker='o')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Final Loss')
    plt.title('Network Size vs Final Loss on XOR Problem')
    plt.grid(True)
    plt.show()

compare_network_sizes()
```

You'll notice that even 2 neurons can solve XOR, but larger networks may train more reliably.
:::

:::{exercise}
:label: ex-multi-layer

Extend the `NeuralNetwork` class to support an arbitrary number of hidden layers. Create a three-layer network (two hidden layers) and train it on a dataset of your choice.
:::

## Summary

In this chapter, we've covered the fundamental building blocks of neural networks:

1. **Biological Inspiration**: Neural networks are inspired by biological neurons but are highly simplified mathematical models
2. **Perceptrons**: The simplest artificial neuron, capable of learning linearly separable functions
3. **Activation Functions**: Non-linear functions that give neural networks their power
4. **Multi-Layer Networks**: Stacking layers allows networks to learn complex, non-linear relationships
5. **Forward Propagation**: Computing outputs by passing data through the network
6. **Architecture Decisions**: Choosing the right number of layers and neurons

:::{important}
Key takeaways:
- A single perceptron can only learn linear decision boundaries
- Activation functions introduce non-linearity
- Multi-layer networks can approximate any continuous function
- ReLU is the most common activation function for hidden layers
- Proper initialization and learning rate selection are crucial for training
:::

In the next chapter, we'll dive deeper into how neural networks learn through the process of gradient descent and backpropagation, the algorithms that make training deep networks possible.

## Further Reading

- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- McCulloch, W. S., & Pitts, W. (1943). "A Logical Calculus of the Ideas Immanent in Nervous Activity"
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" - Chapter 6