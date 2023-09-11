#pragma once
#include <vector>

#include "ActivationFn.hpp"
#include "Matrix.hpp"
#include "Neuron.hpp"
#include "Vector.hpp"

namespace NeuralNetwork
{
    struct Layer
    {
    public:
        std::size_t neuronCount;
        std::size_t connectionCount;

        Math::Matrix weightMatrix;
        Math::Vector biasVector;
        Math::Matrix valueMatrix;

        ActivationFn::ActivationFn* activationFn;

        Layer();
        Layer(std::size_t count);
        Layer(std::size_t count, ActivationFn::ActivationFn* fn);

        ~Layer();
        Layer(const Layer &p_layer);

        void InitializeConnections(std::size_t count);
        Neuron Neurons(unsigned int i);

        Math::Matrix Output();
        Math::Matrix CalculateValues(Math::Matrix input);
        void AdjustNeurons(Math::Matrix weightShiftMatrix, Math::Vector biasShiftVector, double mult = 1);
    };
}