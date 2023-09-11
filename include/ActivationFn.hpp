#pragma once
#include <cmath>
#include <vector>
#include <functional>

namespace ActivationFn
{
    class ActivationFn
    {
    public:
        virtual ~ActivationFn();

        virtual double fn(double x);
        std::function<double(double)> fn();
        virtual double dx(double x);
        std::function<double(double)> dx();
        virtual ActivationFn* clone();
    };

    class ReLU : public ActivationFn
    {
    public:
        double fn(double x) override;
        double dx(double x) override;
        ActivationFn* clone() override;
    };

    class LeakyReLU : public ActivationFn
    {
    public:
        double fn(double x) override;
        double dx(double x) override;
        ActivationFn* clone() override;
    };

    class Tanh : public ActivationFn
    {
    public:
        double fn(double x) override;
        double dx(double x) override;
        ActivationFn* clone() override;
    };

    class LogisticSigmoid : public ActivationFn
    {
    public:
        double fn(double x) override;
        double dx(double x) override;
        ActivationFn* clone() override;
    };

    class Linear : public ActivationFn
    {
    public:
        double fn(double x) override;
        double dx(double x) override;
        ActivationFn* clone() override;
    };
}
