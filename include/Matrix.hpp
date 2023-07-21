#pragma once
#include <cmath>
#include <vector>

namespace Math
{
    struct Matrix
    {
        using matrix = std::vector<std::vector<float>>;

    public:
        unsigned int rows;
        unsigned int cols;
        
        Matrix(unsigned int p_rows, unsigned int cols);
        Matrix(unsigned int p_rows, unsigned int cols, matrix values);
        Matrix(matrix values);
        Matrix(unsigned int p_rows, unsigned int p_cols, std::vector<float> values);

        static Matrix ColumnMatrix(std::vector<float> values);

        friend Matrix operator+(Matrix const &m1, Matrix const &m2);
        Matrix operator+(std::vector<float> const &vector) const;
        Matrix &operator+=(Matrix const &matrix);
        Matrix operator-(Matrix const &matrix) const;
        Matrix &operator-=(Matrix const &matrix);

        friend Matrix operator*(const float &num, Matrix const &matrix);
        Matrix &operator*=(const float &num);
        Matrix operator/(const float &num) const;
        Matrix &operator/=(const float &num);

        Matrix operator*(std::vector<float> const &vector) const;
        Matrix operator*(Matrix const &matrix) const;

        std::vector<float> operator[](unsigned int i) const;

        operator std::vector<float>() const;

    private:
        std::vector<float> values;
    };

    Matrix operator+(Matrix const &m1, Matrix const &m2);
    Matrix operator*(const float &num, Matrix const &matrix);
}



