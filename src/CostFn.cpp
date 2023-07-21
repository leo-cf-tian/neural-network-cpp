#include <cmath>

#include "CostFn.hpp"

namespace CostFn
{
    float L2::fn(float target, float value)
    {
        return (value - target) * (value - target);
    };

    float L2::dx(float target, float value)
    {
        return 2 * (value - target);
    };
}

