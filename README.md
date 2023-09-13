# neural-network-cpp
A neural network (more specifically, a multilayer perceptron) implementation in C++, in addition to a self-implemented multi-threaded matrix math class to vectorize calculations.


## Matrix Class and the Math Namespace
Under the `Math` namespace (see `Matrix.hpp` or `Vector.hpp` in `include`). The these custom structs supports several common operations used in linear algebra as well as some utilities:
- Initialization with C++ `std::vector` of doubles
- `*, -, +, /` and `*=, -=, +=, /=` operators between matrices and doubles
- `*, -, +` and `-=, +=` between matrices
- `&` Hadamard/element-wise products
- A `Math::Vector` class to encapsulate row/column vectors as a special sub-class of matrices
- Random matrix generation (default random engine, not optimized for better numerical distribution)
- Direct type casting into a double or `std::vector` of doubles
- Some other mathematical functions

A `ThreadPool` class is defined for multi-threading optimizations in and outside of the matrix class, the use of which is controlled by benchmarked calculations.


## Neural Network
Under the `NeuralNetwork` namespace consists of several components
- `MultilayerPerceptron` class with fully vectorized calculations
- `Layer` class with fully vectorized calculations and value storage
- `Neuron` deprecated class (replaced by vectorized values stored in `Layer`)
- `ActivationFn::ActivationFn` different activation functions
- `CostFn::CostFn` different cost functions


Other notable components include
- `Data` for reading and storing data in a vectorized manner
- `Threadpool` for optimizing thread usage


## Performance and Accuracy
Training on MNIST
- 99% accuracy and 95% validation accuracy in 40-50 epochs
- Around 1.5 seconds per epoch (60000 data instances), which could use major improvements


## Current issues
- Certain classes need refactoring to improve reliability
- Multithreading does not provide satisfying performance increases for matrix calculations
- Current implementation of cost functions class does not support softmax


## Todo
- Implement softmax rather than using sigmoid
- Automatic differentiation
- CUDA optimizations rather than CPU multi-threading
- Implement configurable metrics, optimizers (momentum, ADAM, etc.)