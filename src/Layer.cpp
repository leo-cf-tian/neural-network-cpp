#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <functional>

#include "ActivationFn.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "Neuron.hpp"
#include "Vector.hpp"

namespace NeuralNetwork
{
    Layer::Layer()
        : neuronCount(1), weightMatrix(Math::Matrix(1, 1)), biasVector(Math::Vector(1)), valueMatrix(Math::Matrix(1, 1)), activationFn(nullptr) {};

    Layer::Layer(std::size_t p_count)
        : neuronCount(p_count), weightMatrix(Math::Matrix(p_count, 1)), biasVector(Math::Vector(p_count)), valueMatrix(Math::Matrix(p_count, 1)), activationFn(nullptr) {};

    Layer::Layer(std::size_t p_count, ActivationFn::ActivationFn *p_fn)
        : neuronCount(p_count), weightMatrix(Math::Matrix(p_count, 1)), biasVector(Math::Vector(p_count)), valueMatrix(Math::Matrix(p_count, 1)), activationFn(p_fn) {};

    Layer::~Layer()
    {
        delete activationFn;
    };

    Layer::Layer(const Layer &p_layer)
        : neuronCount(p_layer.neuronCount), connectionCount(p_layer.connectionCount), weightMatrix(p_layer.weightMatrix), biasVector(p_layer.biasVector), valueMatrix(p_layer.valueMatrix), activationFn(nullptr)
    {
        if (p_layer.activationFn)
            activationFn = p_layer.activationFn->clone();
    };

    void Layer::InitializeConnections(std::size_t count)
    {
        srand(std::chrono::system_clock::now().time_since_epoch().count());

        connectionCount = count;

        weightMatrix = Math::Matrix::RandomMatrix(neuronCount, connectionCount, -0.1, 0.1);
        biasVector = Math::Matrix(neuronCount, 1);
    };

    Math::Matrix Layer::Output()
    {
        if (activationFn)
            return valueMatrix.Apply(activationFn->fn());
        
        return valueMatrix;
    };

    Math::Matrix Layer::CalculateValues(Math::Matrix input)
    {
        if (input.rows != connectionCount)
            throw std::invalid_argument("input dimensions do not match specified dimensions");

        valueMatrix = weightMatrix * input + biasVector;

        return Output();
    };

    void Layer::AdjustNeurons(Math::Matrix weightShiftMatrix, Math::Vector biasShiftVector, double mult)
    {
        if (weightMatrix.rows != weightShiftMatrix.rows || weightMatrix.cols != weightShiftMatrix.cols || biasVector.size() != biasShiftVector.size())
            throw std::invalid_argument("number of adjustments do not match number of neurons");

        weightMatrix += weightShiftMatrix * mult;
        biasVector += biasShiftVector * mult;
    };
}