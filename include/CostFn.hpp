#pragma once
#include <functional>

namespace CostFn
{
    class CostFn
    {
    public:
        virtual double fn(double value, double target);
        std::function<double(double)> fn(double target);
        std::function<double(double, double)> fn();
        virtual double dx(double value, double target);
        std::function<double(double)> dx(double target);
        std::function<double(double, double)> dx();
    };

    class L2 : public CostFn
    {
    public:
        double fn(double value, double target) override;
        double dx(double value, double target) override;
    };

    class BinaryCrossEntropy : public CostFn
    {
    public:
        double fn(double value, double target) override;
        double dx(double value, double target) override;
    };
}
