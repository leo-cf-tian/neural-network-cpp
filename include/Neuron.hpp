#pragma once
#include <vector>

namespace NeuralNetwork {
    struct Neuron
    {
        std::vector<float> weights;
        float bias;
        float value;

        Neuron(unsigned int connectionCount);
        
        void Adjust(std::vector<float> weightShift, float biasShift, float mult);
    };
}