#pragma once
#include <vector>

#include "ActivationFn.hpp"
#include "Matrix.hpp"
#include "Neuron.hpp"

namespace NeuralNetwork
{
    struct Layer
    {
    public:
        double connectionCount;
        std::vector<Neuron> neurons;
        ActivationFn::ActivationFn* activationFn;

        Layer();
        Layer(std::size_t count);
        Layer(std::size_t count, ActivationFn::ActivationFn* fn);

        void InitializeConnections(std::size_t count);
    
        Math::Matrix WeightMatrix();
        std::vector<double> BiasVector();
        std::vector<double> OutputVector();
        std::vector<double> CalculateValues(std::vector<double> input);
        void AdjustNeurons(std::vector<std::vector<double>> weightShiftVector, std::vector<double> biasShiftVector, double mult = 1);
    };
}