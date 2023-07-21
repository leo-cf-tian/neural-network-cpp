#pragma once
#include <cmath>
#include <vector>

namespace ActivationFn
{
    class ActivationFn
    {
    public:
        virtual float fn(float x);
        std::vector<float> fn(std::vector<float> vecX);
        virtual float dx(float x);
        std::vector<float> dx(std::vector<float> vecX);
    };

    class ReLU : ActivationFn
    {
    public:
        float fn(float x) override;
        float dx(float x) override;
    };

    class Tanh : ActivationFn
    {
    public:
        float fn(float x) override;
        float dx(float x) override;
    };

    class LogisticSigmoid : ActivationFn
    {
    public:
        float fn(float x) override;
        float dx(float x) override;
    };
}
