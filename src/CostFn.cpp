#include <cmath>

#include "CostFn.hpp"

namespace CostFn
{
    std::function<double(double)> CostFn::fn(double target)
    {
        return [this, target](double prediction) { return fn(prediction, target); };
    };

    std::function<double(double, double)> CostFn::fn()
    {
        return [this](double prediction, double target) { return fn(prediction, target); };
    };

    std::function<double(double)> CostFn::dx(double target)
    {
        return [this, target](double prediction) { return dx(prediction, target); };
    };

    std::function<double(double, double)> CostFn::dx()
    {
        return [this](double prediction, double target) { return dx(prediction, target); };
    };

    double L2::fn(double value, double target)
    {
        return (value - target) * (value - target);
    };

    double L2::dx(double value, double target)
    {
        return 2 * (value - target);
    };
}

