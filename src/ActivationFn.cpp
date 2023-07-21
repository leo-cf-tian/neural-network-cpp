#include <cmath>

#include "ActivationFn.hpp"

namespace ActivationFn
{
    std::vector<float> ActivationFn::fn(std::vector<float> vecX)
    {
        std::vector<float> result = std::vector<float>(vecX.size(), 0);

        for (unsigned int i = 0; i < vecX.size(); i++) {
            result[i] = fn(vecX[i]);
        }

        return result;
    }

    std::vector<float> ActivationFn::dx(std::vector<float> vecX)
    {
        std::vector<float> result = std::vector<float>(vecX.size(), 0);

        for (unsigned int i = 0; i < vecX.size(); i++) {
            result[i] = dx(vecX[i]);
        }

        return result;
    }

    float ReLU::fn(float x)
    {
        return x > 0 ? x : 0;
    };

    float ReLU::dx(float x)
    {
        return x > 0 ? 1 : 0;
    };

    float Tanh::fn(float x)
    {
        return tanh(x);
    };

    float Tanh::dx(float x)
    {
        return 1 / cosh(x) / cosh(x);
    };

    float LogisticSigmoid::fn(float x)
    {
        return 1 / (1 + exp(-x));
    };

    float LogisticSigmoid::dx(float x)
    {
        return 1 / (exp(x) + 2 + exp(-x));
    }; 
}
