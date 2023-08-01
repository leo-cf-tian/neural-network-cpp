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
        float connectionCount;
        std::vector<Neuron> neurons;
        ActivationFn::ActivationFn* activationFn;

        Layer();
        Layer(unsigned int count);
        Layer(unsigned int count, ActivationFn::ActivationFn* fn);

        void InitializeConnections(unsigned int count);
    
        Math::Matrix WeightMatrix();
        std::vector<float> BiasVector();
        std::vector<float> OutputVector();
        std::vector<float> CalculateValues(std::vector<float> input);
        void AdjustNeurons(std::vector<std::vector<float>> weightShiftVector, std::vector<float> biasShiftVector, float mult);
    };
}