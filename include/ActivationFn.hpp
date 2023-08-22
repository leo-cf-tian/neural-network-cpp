#pragma once
#include <cmath>
#include <vector>
#include <functional>

namespace ActivationFn
{
    class ActivationFn
    {
    public:
        virtual double fn(double x);
        std::function<double(double)> fn();
        virtual double dx(double x);
        std::function<double(double)> dx();
    };

    class ReLU : public ActivationFn
    {
    public:
        double fn(double x) override;
        double dx(double x) override;
    };

    class LeakyReLU : public ActivationFn
    {
    public:
        double fn(double x) override;
        double dx(double x) override;
    };

    class Tanh : public ActivationFn
    {
    public:
        double fn(double x) override;
        double dx(double x) override;
    };

    class LogisticSigmoid : public ActivationFn
    {
    public:
        double fn(double x) override;
        double dx(double x) override;
    };

    class Linear : public ActivationFn
    {
    public:
        double fn(double x) override;
        double dx(double x) override;
    };
}
