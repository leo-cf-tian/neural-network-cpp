#pragma once
#include <cmath>
#include <vector>

#include "Matrix.hpp"

namespace Math
{
    /**
     * @brief A mathematical vector / column matrix
     */
    struct Vector : Matrix
    {
    public:
        Vector(std::size_t p_size, bool isColumn = true);
        Vector(std::size_t p_size, double value, bool isColumn = true);
        Vector(std::vector<double> values, bool isColumn = true);
        Vector(Matrix values);

        Matrix Transpose();
        std::size_t size() const;

        double operator[](std::size_t i) const;
        double &operator[](std::size_t i);
        double at(std::size_t i) const;
        double &at(std::size_t i);
    };
}



