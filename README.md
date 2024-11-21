# nn-rs
Just a simple neural network in Rust for learning purposes.

## Project Goals
- [ ] Implement a basic neural network from scratch.
- [ ] Focus on image classification as the application.

## Checklist

### **1. Mathematical Foundations**
   - [ ] **Linear Algebra**:
     - [ ] Implement vector and matrix operations (addition, multiplication, dot product).
     - [ ] Transpose, inverse, determinant for matrices.
   - [ ] **Activation Functions**:
     - [ ] Implement common ones like ReLU, Sigmoid, Tanh.
     - [ ] Possibly Softmax for output layer.

### **2. Core Neural Network Components**
   - [ ] **Neuron**:
     - [ ] Define what constitutes a neuron (weights, bias, activation function).
   - [ ] **Layer**:
     - [ ] Create a structure for layers (input, hidden, output).
     - [ ] Implement forward pass for layers.
   - [ ] **Neural Network**:
     - [ ] Structure to hold layers.
     - [ ] Methods for adding layers, forward pass through the whole network.

### **3. Data Handling**
   - [ ] **Data Loading**:
     - [ ] Implement or use an existing library to load image datasets (e.g., MNIST).
     - [ ] Normalize data.
     - [ ] Prepare labels.
   - [ ] **Data Preprocessing**:
     - [ ] Convert images to grayscale if needed.
     - [ ] Resize images to a uniform size.

### **4. Training**
   - [ ] **Loss Function**:
     - [ ] Implement Cross-Entropy loss or Mean Squared Error.
   - [ ] **Backpropagation**:
     - [ ] Compute gradients of the loss with respect to weights and biases.
     - [ ] Adjust weights and biases using gradient descent or a variant.
   - [ ] **Optimization Algorithms**:
     - [ ] Implement Stochastic Gradient Descent (SGD).
     - [ ] Optionally, more advanced optimizers like Adam, RMSprop.

### **5. Evaluation**
   - [ ] **Accuracy Calculation**:
     - [ ] Function to compute classification accuracy.
   - [ ] **Validation**:
     - [ ] Split dataset into training, validation, and test sets.
     - [ ] Use validation set for hyperparameter tuning.

### **6. Image Classification Specifics**
   - [ ] **Flattening Images**:
     - [ ] Convert images from 2D to 1D for network input.
   - [ ] **Convolutional Layers** (optional for a later stage):
     - [ ] Implement if you plan to enhance your network's capability.

### **7. Testing and Debugging**
   - [ ] **Unit Tests**:
     - [ ] Write tests for basic operations (matrix multiplication, neuron activation, etc.).
   - [ ] **Integration Tests**:
     - [ ] Test the whole pipeline with a few known image sets.

### **8. Documentation and Reporting**
   - [ ] **Documentation**:
     - [ ] Document the structure of your network, how to use it, and the rationale behind design choices.
   - [ ] **Visualization**:
     - [ ] Optionally, create ways to visualize network structure, weights, or learning progress.

### **9. Future Enhancements**
   - [ ] **Advanced Layers**: Implement convolutional layers, pooling, dropout.
   - [ ] **Regularization**: Add L1/L2 regularization to prevent overfitting.
   - [ ] **Hyperparameter Tuning**: Implement or integrate tools for better parameter optimization.

## Usage

Coming in the future when ready..

## License

MIT License.
