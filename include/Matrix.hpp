#pragma once
#include <cmath>
#include <vector>

namespace Math
{
    struct Matrix
    {
        using matrix = std::vector<std::vector<double>>;

    public:
        Matrix(std::size_t p_rows, std::size_t cols);
        Matrix(std::size_t p_rows, std::size_t cols, matrix values);
        Matrix(matrix values);
        Matrix(std::size_t p_rows, std::size_t p_cols, std::vector<double> values);

        static Matrix RandomMatrix(std::size_t rows, std::size_t cols);

        friend Matrix operator+(Matrix const &m1, Matrix const &m2);
        Matrix operator+(std::vector<double> const &vector) const;
        Matrix &operator+=(Matrix const &matrix);
        Matrix operator-(Matrix const &matrix) const;
        Matrix &operator-=(Matrix const &matrix);
        Matrix operator-();

        friend Matrix operator*(const double &num, Matrix const &matrix);
        Matrix operator*(const double &num) const;
        Matrix &operator*=(const double &num);
        Matrix operator/(const double &num) const;
        Matrix &operator/=(const double &num);

        Matrix operator*(std::vector<double> const &vector) const;
        Matrix operator*(Matrix const &matrix) const;

        std::vector<double> operator[](std::size_t i) const;

        operator std::vector<double>() const;

    protected:
        std::vector<double> values;
        std::size_t rows;
        std::size_t cols;
    };

    Matrix operator+(Matrix const &m1, Matrix const &m2);
    Matrix operator*(const double &num, Matrix const &matrix);
}



