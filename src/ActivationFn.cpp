#include <cmath>

#include "ActivationFn.hpp"

namespace ActivationFn
{
    std::vector<double> ActivationFn::fn(std::vector<double> vecX)
    {
        std::vector<double> result = std::vector<double>(vecX.size(), 0);

        for (unsigned int i = 0; i < vecX.size(); i++) {
            result[i] = fn(vecX[i]);
        }

        return result;
    }

    std::vector<double> ActivationFn::dx(std::vector<double> vecX)
    {
        std::vector<double> result = std::vector<double>(vecX.size(), 0);

        for (unsigned int i = 0; i < vecX.size(); i++) {
            result[i] = dx(vecX[i]);
        }

        return result;
    }

    double ReLU::fn(double x)
    {
        return x > 0 ? x : 0;
    };

    double ReLU::dx(double x)
    {
        return x > 0 ? 1 : 0;
    };

    double LeakyReLU::fn(double x)
    {
        return x > 0 ? x : x * 0.1;
    };

    double LeakyReLU::dx(double x)
    {
        return x > 0 ? 1 : 0.1;
    };

    double Tanh::fn(double x)
    {
        return tanh(x);
    };

    double Tanh::dx(double x)
    {
        return 1 / cosh(x) / cosh(x);
    };

    double LogisticSigmoid::fn(double x)
    {
        return 1 / (1 + exp(-x));
    };

    double LogisticSigmoid::dx(double x)
    {
        if (x > 5 || x < -5)
            return 0;

        return 1 / (exp(x) + 2 + exp(-x));
    }; 

    double Linear::fn(double x)
    {
        return x;
    };

    double Linear::dx(double x)
    {
        return 1;
    };
}
