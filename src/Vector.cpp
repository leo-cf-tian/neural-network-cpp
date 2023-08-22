#include <cmath>
#include <iostream>
#include <vector>
#include "Matrix.hpp"
#include "Vector.hpp"

namespace Math
{
    Vector::Vector(std::size_t p_size, bool isColumn)
        : Matrix(isColumn ? p_size : 1, isColumn ? 1 : p_size, 0) {};

    Vector::Vector(std::size_t p_size, double value, bool isColumn)
        : Matrix(isColumn ? p_size : 1, isColumn ? 1 : p_size, value) {};

    Vector::Vector(std::vector<double> p_values, bool isColumn)
        : Matrix(isColumn ? p_values.size() : 1, isColumn ? 1 : p_values.size(), p_values) {};

    Vector::Vector(Math::Matrix p_values)
        : Matrix(p_values)
    {
        if (rows != 1 && cols != 1)
            throw std::invalid_argument("only column or row matrices can be cast to vectors");
    };

    Matrix Vector::Transpose()
    {
        return Matrix(1, rows, values);
    };
    
    std::size_t Vector::size() const
    {
        return rows;
    };

    double Vector::operator[](std::size_t i) const
    {
        if ((i >= rows && i >= cols) || i < 0)
            throw std::invalid_argument("index out of range");

        return values[i];
    };
    
    double &Vector::operator[](std::size_t i)
    {
        if ((i >= rows && i >= cols) || i < 0)
            throw std::invalid_argument("index out of range");

        return values[i];
    };

    double Vector::at(std::size_t i) const
    {
        if ((i >= rows && i >= cols) || i < 0)
            throw std::invalid_argument("index out of range");

        return values[i];
    };

    double &Vector::at(std::size_t i)
    {
        if ((i >= rows && i >= cols) || i < 0)
            throw std::invalid_argument("index out of range");

        return values[i];
    };
}