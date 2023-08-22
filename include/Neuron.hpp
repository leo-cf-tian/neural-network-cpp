#pragma once
#include <vector>

#include "Vector.hpp"

namespace NeuralNetwork {
    struct Neuron
    {
        Math::Vector weights;
        double bias;
        Math::Vector values;
    };
}