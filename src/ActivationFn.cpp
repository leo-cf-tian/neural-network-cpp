#include <cmath>

#include "ActivationFn.hpp"

namespace ActivationFn
{
    ActivationFn::~ActivationFn() {};


    std::function<double(double)> ActivationFn::fn()
    {
        return [this](double x) { return fn(x); };
    };

    std::function<double(double)> ActivationFn::dx()
    {
        return [this](double x) { return dx(x); };
    };

    double ActivationFn::fn(double x) { return x; };
    double ActivationFn::dx(double x) { return 1; };
    ActivationFn* ActivationFn::clone() { return new ActivationFn(); };

    double ReLU::fn(double x)
    {
        return x > 0 ? x : 0;
    };

    double ReLU::dx(double x)
    {
        return x > 0 ? 1 : 0;
    };
    
    ActivationFn* ReLU::clone() { return new ReLU(); };

    double LeakyReLU::fn(double x)
    {
        return x > 0 ? x : x * 0.1;
    };

    double LeakyReLU::dx(double x)
    {
        return x > 0 ? 1 : 0.1;
    };
    
    ActivationFn* LeakyReLU::clone() { return new LeakyReLU(); };

    double Tanh::fn(double x)
    {
        return tanh(x);
    };

    double Tanh::dx(double x)
    {
        return 1 / cosh(x) / cosh(x);
    };
    
    ActivationFn* Tanh::clone() { return new Tanh(); };

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
    
    ActivationFn* LogisticSigmoid::clone() { return new LogisticSigmoid(); };

    double Linear::fn(double x)
    {
        return x;
    };

    double Linear::dx(double x)
    {
        return 1;
    };
    
    ActivationFn* Linear::clone() { return new Linear(); };
}
