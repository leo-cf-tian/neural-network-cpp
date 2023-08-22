#pragma once
#include <cmath>
#include <vector>
#include <bits/stdc++.h>

namespace Math
{
    struct Vector;

    struct Matrix
    {
        using matrix = std::vector<std::vector<double>>;
        std::size_t rows;
        std::size_t cols;

    public:
        Matrix(std::size_t p_rows, std::size_t p_cols);
        Matrix(std::size_t p_rows, std::size_t p_cols, double value);
        Matrix(std::size_t p_rows, std::size_t p_cols, matrix values);
        Matrix(matrix values);
        Matrix(std::size_t p_rows, std::size_t p_cols, std::vector<double> values);

        static Matrix RandomMatrix(std::size_t rows, std::size_t cols, double min = -1, double max = 1);

        Matrix operator+(Matrix const &matrix) const;
        /**
         * @brief Adds column vector to each column of matrix
         */
        Matrix operator+(Math::Vector const &vector) const;
        Matrix &operator+=(Matrix const &matrix);
        Matrix operator-(Matrix const &matrix) const;
        /**
         * @brief Subtracts vector from each column of matrix
         */
        Matrix operator-(Math::Vector const &vector) const;
        Matrix &operator-=(Matrix const &matrix);
        Matrix operator-();

        friend Matrix operator*(const double &num, Matrix const &matrix);
        Matrix operator*(const double &num) const;
        Matrix &operator*=(const double &num);
        Matrix operator/(const double &num) const;
        Matrix &operator/=(const double &num);

        /**
         * @brief Product of two matrices
         */
        Matrix operator*(Matrix const &matrix) const;
        
        /**
         * @brief Hadamard / element-wise product of two matrices
         */
        Matrix operator&(Matrix const &matrix) const;

        std::vector<double> operator[](std::size_t i) const;
        double at(std::size_t row, std::size_t col) const;
        double &at(std::size_t row, std::size_t col);

        operator std::vector<double>() const;
        operator double() const;

        Matrix Transpose();

        /**
         * @brief Applies a function to the matrix
         * @param fn fn(x): where x is the matrix member
         */
        Matrix Apply(std::function<double(double)> fn);

        /**
         * @brief Applies a function to each member of the matrix, with a different parameter for each member
         * @param fn fn(x, y): where x is the matrix member, and y is the argument matrix member
         * @param argMatrix Matrix of arguments corresponding to each member
         */
        Matrix ApplyForEach(std::function<double(double, double)> fn, Matrix argMatrix);

    protected:
        std::vector<double> values;
    };

    Matrix operator+(Matrix const &m1, Matrix const &m2);
    Matrix operator*(const double &num, Matrix const &matrix);
}



