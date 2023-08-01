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

    class ReLU : public ActivationFn
    {
    public:
        float fn(float x) override;
        float dx(float x) override;
    };

    class Tanh : public ActivationFn
    {
    public:
        float fn(float x) override;
        float dx(float x) override;
    };

    class LogisticSigmoid : public ActivationFn
    {
    public:
        float fn(float x) override;
        float dx(float x) override;
    };

    class Linear : public ActivationFn
    {
    public:
        float fn(float x) override;
        float dx(float x) override;
    };
}
