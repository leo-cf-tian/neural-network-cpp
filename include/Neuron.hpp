#pragma once
#include <vector>

namespace NeuralNetwork {
    struct Neuron
    {
        std::vector<double> weights;
        double bias;
        double value;

        Neuron(std::size_t connectionCount);
        
        void Adjust(std::vector<double> weightShift, double biasShift, double mult);
    };
}