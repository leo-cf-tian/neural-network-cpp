#include <cmath>

#include "CostFn.hpp"

namespace CostFn
{
    double L2::fn(double target, double value)
    {
        return (value - target) * (value - target);
    };

    double L2::dx(double target, double value)
    {
        return 2 * (value - target);
    };
}

