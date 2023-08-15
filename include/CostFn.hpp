#pragma once

namespace CostFn
{
    class CostFn
    {
    public:
        virtual double fn(double target, double value);
        virtual double dx(double target, double value);
    };

    class L2 : public CostFn
    {
    public:
        double fn(double target, double value) override;
        double dx(double target, double value) override;
    };

    class BinaryCrossEntropy : public CostFn
    {
    public:
        double fn(double target, double value) override;
        double dx(double target, double value) override;
    };
}
