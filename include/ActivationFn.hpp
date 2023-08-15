#pragma once
#include <cmath>
#include <vector>

namespace ActivationFn
{
    class ActivationFn
    {
    public:
        virtual double fn(double x);
        std::vector<double> fn(std::vector<double> vecX);
        virtual double dx(double x);
        std::vector<double> dx(std::vector<double> vecX);
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
