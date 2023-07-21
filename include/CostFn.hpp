#pragma once

namespace CostFn
{
    class CostFn
    {
    public:
        virtual float fn(float target, float value);
        virtual float dx(float target, float value);
    };

    class L2 : CostFn
    {
    public:
        float fn(float target, float value) override;
        float dx(float target, float value) override;
    };
}
