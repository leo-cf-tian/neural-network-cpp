#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>

#include "ActivationFn.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "Neuron.hpp"

namespace NeuralNetwork
{
    Layer::Layer(std::size_t p_count)
        : activationFn(nullptr)
    {
        neurons = std::vector<Neuron>(p_count, Neuron(0));
    }

    Layer::Layer(std::size_t p_count, ActivationFn::ActivationFn *p_fn)
        : activationFn(p_fn)
    {
        neurons = std::vector<Neuron>(p_count, Neuron(0));
    }

    void Layer::InitializeConnections(std::size_t count)
    {
        srand(std::chrono::system_clock::now().time_since_epoch().count());

        connectionCount = count;

        for (auto &n : neurons) {
            n.weights = std::vector<double>(count, 0);

            for (unsigned int i = 0; i < n.weights.size(); i++) {
                n.weights[i] = rand() / static_cast<double>(RAND_MAX) * 2 - 1;
            }
            
            n.bias = rand() / static_cast<double>(RAND_MAX) * 2 - 1;
        }
    }

    Math::Matrix Layer::WeightMatrix()
    {
        std::vector<std::vector<double>> weights = {};

        for (auto neuron : neurons) {
            weights.push_back(neuron.weights);
        }

        return Math::Matrix(weights);
    }

    std::vector<double> Layer::BiasVector()
    {
        std::vector<double> bias = {};

        for (auto neuron : neurons) {
            bias.push_back(neuron.bias);
        }

        return bias;
    }

    std::vector<double> Layer::OutputVector()
    {
        std::vector<double> output = {};

        for (auto neuron : neurons) {
            if (activationFn != nullptr)
                output.push_back(activationFn->fn(neuron.value));
            else
                output.push_back(neuron.value);
        }

        return output;
    }

    std::vector<double> Layer::CalculateValues(std::vector<double> input)
    {
        if (input.size() != connectionCount)
            throw std::invalid_argument("input dimensions do not match specified dimensions");

        std::vector<double> values = WeightMatrix() * input + BiasVector();

        for (unsigned int i = 0; i < values.size(); i++)
        {
            neurons[i].value = values[i];
        }

        return OutputVector();
    }

    void Layer::AdjustNeurons(std::vector<std::vector<double>> weightShiftVector, std::vector<double> biasShiftVector, double mult)
    {
        if (neurons.size() != weightShiftVector.size() || neurons.size() != biasShiftVector.size())
            throw std::invalid_argument("number of adjustments do not match number of neurons");

        for (unsigned int i = 0; i < neurons.size(); i++) {
            neurons[i].Adjust(weightShiftVector[i], biasShiftVector[i], mult);
        }
    }
}