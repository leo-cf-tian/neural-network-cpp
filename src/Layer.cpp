#include <iostream>
#include <vector>

#include "ActivationFn.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "Neuron.hpp"

namespace NeuralNetwork
{
    Layer::Layer(unsigned int p_count, ActivationFn::ActivationFn *p_fn)
        : activationFn(p_fn)
    {
        neurons = std::vector<Neuron>(p_count, Neuron(0));
    }

    void Layer::InitializeConnections(unsigned int count)
    {
        connectionCount = count;

        for (auto n : neurons) {
            n.weights = std::vector<float>(count, 0);
            n.bias = 0;
        }
    }

    Math::Matrix Layer::WeightMatrix()
    {
        std::vector<std::vector<float>> weights = {};

        for (auto neuron : neurons) {
            weights.push_back(neuron.weights);
        }

        return weights;
    }

    std::vector<float> Layer::BiasVector()
    {
        std::vector<float> bias = {};

        for (auto neuron : neurons) {
            bias.push_back(neuron.bias);
        }

        return bias;
    }

    std::vector<float> Layer::OutputVector()
    {
        std::vector<float> output = {};

        for (auto neuron : neurons) {
            if (activationFn != nullptr)
                output.push_back(activationFn->fn(neuron.value));
            else
                output.push_back(neuron.value);
        }

        return output;
    }

    std::vector<float> Layer::CalculateValues(std::vector<float> input)
    {
        if (input.size() != connectionCount)
            throw std::invalid_argument("input dimensions do not match specified dimensions");
        
        std::vector<float> values = WeightMatrix() * input + BiasVector();

        for (unsigned int i = 0; i < values.size(); i++)
        {
            neurons[i].value = values[i];
        }

        return OutputVector();
    }

    void Layer::AdjustNeurons(std::vector<std::vector<float>> weightShiftVector, std::vector<float> biasShiftVector)
    {
        if (neurons.size() != weightShiftVector.size() || neurons.size() != biasShiftVector.size())
            throw std::invalid_argument("number of adjustments do not match number of neurons");

        for (unsigned int i = 0; i < neurons.size(); i++) {
            neurons[i].Adjust(weightShiftVector[i], biasShiftVector[i]);
        }
    }
}