#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "Matrix.hpp"

namespace Math
{
    Matrix::Matrix(std::size_t p_rows, std::size_t p_cols)
        : rows(p_rows), cols(p_cols)
    {
        if (rows < 1 || cols < 1) 
            throw std::invalid_argument("matrix dimensions must be positive");

        values = std::vector<double>(rows * cols, 0);
    };

    Matrix::Matrix(std::size_t p_rows, std::size_t p_cols, matrix p_values)
        : rows(p_rows), cols(p_cols)
    {
        if (rows < 1 || cols < 1) 
            throw std::invalid_argument("matrix dimensions must be positive");

        if (p_values.size() != rows)
            throw std::invalid_argument("matrix row count does not match specified size");

        for (auto row : p_values) {
            if (row.size() != cols) {
                throw std::invalid_argument("matrix column count does not match specified size");
            }
        }
        
        values = std::vector<double>(rows * cols, 0);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                values[i * cols + j] = p_values[i][j];
            }
        }
    };

    Matrix::Matrix(matrix p_values)
    {
        rows = p_values.size();
        cols = p_values[0].size();

        if (rows < 1 || cols < 1) 
            throw std::invalid_argument("matrix dimensions must be positive");

        for (auto row : p_values) {
            if (row.size() != cols) {
                throw std::invalid_argument("matrix column count does not match specified size");
            }
        }
            
        values = std::vector<double>(rows * cols, 0);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                values[i * cols + j] = p_values[i][j];
            }
        }
    };

    Matrix::Matrix(std::size_t p_rows, std::size_t p_cols, std::vector<double> p_values)
        : rows(p_rows), cols(p_cols)
    {
        if (rows < 1 || cols < 1) 
            throw std::invalid_argument("matrix dimensions must be positive");

        if (p_values.size() != rows * cols) 
            throw std::invalid_argument("length of values array does not match matrix size");
            
        values = p_values;
    };

    
    Matrix Matrix::ColumnMatrix(std::vector<double> p_values)
    {
        return Matrix(p_values.size(), 1, p_values);
    }

    Matrix Matrix::RandomMatrix(std::size_t rows, std::size_t cols)
    {
        srand(std::chrono::system_clock::now().time_since_epoch().count());

        std::vector<std::vector<double>> values = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                values[i][j] = rand() / static_cast<double>(RAND_MAX) * 200 - 100;
            }
        }

        Math::Matrix result(values);

        return result;
    }

    Matrix operator+(Matrix const &m1, Matrix const &m2)
    {
        if (m1.rows != m2.rows || m1.cols != m2.cols)
            throw std::invalid_argument("matrices are not of the same size");

        Matrix result(m1.rows, m2.cols);

        for (unsigned int i = 0; i < m1.rows * m1.cols; i++) {
                result.values[i] = m1.values[i] + m2.values[i];
        }

        return result;
    };
    
    Matrix Matrix::operator+(std::vector<double> const &vector) const
    {
        if (rows != 1 && cols != 1)
            throw std::invalid_argument("vectors can only be added to column or row matrices");

        if (vector.size() != rows * cols) 
            throw std::invalid_argument("length of values array does not match matrix size");

        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows * cols; i++) {
                result.values[i] = values[i] + vector[i];
        }

        return result;
    }

    Matrix &Matrix::operator+=(Matrix const &matrix)
    {
        if (rows != matrix.rows || cols != matrix.cols)
            throw std::invalid_argument("matrices are not of the same size");

        for (unsigned int i = 0; i < rows * cols; i++) {
                values[i] += matrix.values[i];
        }
        
        return *this;
    };

    Matrix Matrix::operator-(Matrix const &matrix) const
    {
        if (rows != matrix.rows || cols != matrix.cols)
            throw std::invalid_argument("matrices are not of the same size");

        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows * cols; i++) {
                result.values[i] = values[i] - matrix.values[i];
        }

        return result;
    };

    Matrix &Matrix::operator-=(Matrix const &matrix)
    {
        if (rows != matrix.rows || cols != matrix.cols)
            throw std::invalid_argument("matrices are not of the same size");

        for (unsigned int i = 0; i < rows * cols; i++) {
            values[i] -= matrix.values[i];
        }
        
        return *this;
    };

    Matrix Matrix::operator-()
    {
        return Matrix(rows, cols, values) * -1.0f;
    };

    Matrix operator*(const double &num, Matrix const &matrix)
    {
        Matrix result(matrix.rows, matrix.cols);

        for (unsigned int i = 0; i < matrix.rows; i++) {
            for (unsigned int j = 0; j < matrix.cols; j++) {
                result.values[i * matrix.cols + j] = matrix.values[i * matrix.cols + j] * num;
            }
        }

        return result;
    };

    Matrix Matrix::operator*(const double &num) const
    {
        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                result.values[i * cols + j] = values[i * cols + j] * num;
            }
        }

        return result;
    };

    Matrix &Matrix::operator*=(const double &num)
    {
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                values[i * cols + j] *= num;
            }
        }

        return *this;
    };

    Matrix Matrix::operator/(const double &num) const
    {
        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                result.values[i * cols + j] = values[i * cols + j] / num;
            }
        }

        return result;
    };

    Matrix &Matrix::operator/=(const double &num)
    {
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                values[i * cols + j] /= num;
            }
        }

        return *this;
    };

    Matrix Matrix::operator*(std::vector<double> const &vector) const
    {
        if (cols != vector.size())
            throw std::invalid_argument("matrix column count and vector length does not match");

       Matrix result(rows, 1);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                result.values[i] += values[i * cols + j] * vector[j];
            }
        }

        return result;
    };

    Matrix Matrix::operator*(Matrix const &matrix) const
    {
        if (cols != matrix.rows)
            throw std::invalid_argument("left matrix column count and right matrix row count does not match");

        Matrix result(rows, matrix.cols);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < matrix.cols; j++) {
                for (unsigned int k = 0; k < matrix.rows; k++) {
                    result.values[i * matrix.cols + j] += values[i * cols + k] * matrix.values[k * matrix.cols + j];
                }
            }
        }

        return result;
    };

    std::vector<double> Matrix::operator[](std::size_t i) const
    {
        if (i >= rows || i < 0)
            throw std::invalid_argument("index out of range");

        return std::vector<double>(values.begin() + i * cols, values.begin() + (i + 1) * cols);
    };

    Matrix::operator std::vector<double>() const
    {
        if (rows != 1 && cols != 1)
            throw std::invalid_argument("only column or row matrices can be cast to vectors");

        return values;
    }
}