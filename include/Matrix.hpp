#pragma once
#include <cmath>
#include <vector>

#include "ThreadPool.hpp"

namespace Math
{
    struct Vector;

    struct Matrix
    {
    protected:
        std::vector<double> values;
        static ThreadPool threadPool;

        /**
         * Uses threadpool to optimize matrix calculations
         * @param fn fn(start, end): where start and end are row numbers of the matrix
         * @param total number of row calculations required
         */
        static void UseThreadPool(std::function<void(unsigned int start, unsigned int end)> fn, int total);

    public:
        using matrix = std::vector<std::vector<double>>;
        std::size_t rows;
        std::size_t cols;

        Matrix(std::size_t p_rows, std::size_t p_cols);
        Matrix(std::size_t p_rows, std::size_t p_cols, double value);
        Matrix(std::size_t p_rows, std::size_t p_cols, matrix values);
        Matrix(matrix values);
        Matrix(std::size_t p_rows, std::size_t p_cols, std::vector<double> values);

        static Matrix RandomMatrix(std::size_t rows, std::size_t cols, double min = -1, double max = 1);

        Matrix operator+(Matrix const &matrix) const;
        /**
         * Adds column vector to each column of matrix
         */
        Matrix operator+(Math::Vector const &vector) const;
        Matrix &operator+=(Matrix const &matrix);
        Matrix operator-(Matrix const &matrix) const;
        /**
         * Subtracts column vector from each column of matrix
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
         * Product of two matrices
         */
        Matrix operator*(Matrix const &matrix) const;
        
        /**
         * Hadamard / element-wise product of two matrices
         */
        Matrix operator&(Matrix const &matrix) const;

        std::vector<double> operator[](std::size_t i) const;
        double at(std::size_t row, std::size_t col) const;
        double &at(std::size_t row, std::size_t col);

        operator std::vector<double>() const;
        operator double() const;

        Matrix Transpose();

        /**
         * Applies a function to the matrix
         * @param fn fn(x): where x is the matrix member
         */
        Matrix Apply(std::function<double(double)> fn);

        /**
         * Applies a function to each member of the matrix, with a different parameter for each member
         * @param fn fn(x, y): where x is the matrix member, and y is the argument matrix member
         * @param argMatrix Matrix of arguments corresponding to each member
         */
        Matrix ApplyForEach(std::function<double(double, double)> fn, Matrix argMatrix);

        /**
         * Prints matrix to console
        */
        void print();
    };

    Matrix operator*(const double &num, Matrix const &matrix);
}



