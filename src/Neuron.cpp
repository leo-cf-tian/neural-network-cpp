#include <iostream>
#include <vector>

#include "Neuron.hpp"

namespace NeuralNetwork
{
    Neuron::Neuron(unsigned int p_connectionCount)
    {
        weights = std::vector<float>(p_connectionCount, 0);
    };
    
    void Neuron::Adjust(std::vector<float> weightShift, float biasShift, float mult = 1)
    {
        if (weightShift.size() != weights.size())
            throw std::invalid_argument("dimensions of adjustment vector do not align with weight vector");

        for (unsigned int i = 0; i < weights.size(); i++) {
            weights[i] += weightShift[i] * mult;
        }

        bias += biasShift * mult;
    }
}