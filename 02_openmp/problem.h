#ifndef PROBLEM_H
#define PROBLEM_H

#include <algorithm>
#include <cmath>

const double A1 = -1, B1 = 1, A2 = -1, B2 = 1;
const double CX = 1, CY = 1, R = 1;

inline double v0(double x, double y) { return std::sin(x + y); }

// Площадь круга (центр 0) в [0,a]x[0,b], точно через первообразную sqrt(R^2 - x^2).
inline double quadrant_area(double a, double b)
{
    a = std::min(a, R);
    b = std::min(b, R);
    if (a * a + b * b <= R * R)
        return a * b;
    double x0 = std::sqrt(R * R - b * b);
    double fa = a * std::sqrt(R * R - a * a) + R * R * std::asin(a / R);
    double fx0 = x0 * b + R * R * std::asin(x0 / R);
    return x0 * b + 0.5 * (fa - fx0);
}

// То же для a, b любого знака, площадь со знаком.
inline double signed_quadrant_area(double a, double b)
{
    double sign = (a < 0) == (b < 0) ? 1 : -1;
    return sign * quadrant_area(std::fabs(a), std::fabs(b));
}

// Площадь прямоугольник ∩ круг: включение-исключение по четырём углам.
inline double disk_area(double x1, double x2, double y1, double y2)
{
    x1 -= CX; x2 -= CX;
    y1 -= CY; y2 -= CY;
    return signed_quadrant_area(x2, y2) - signed_quadrant_area(x1, y2)
         - signed_quadrant_area(x2, y1) + signed_quadrant_area(x1, y1);
}

// S = mes([x1,x2]x[y1,y2] ∩ D) из раздела 4.
inline double area_D(double x1, double x2, double y1, double y2)
{
    double full = (x2 - x1) * (y2 - y1);
    double nx = std::max(x1 - CX, std::max(0.0, CX - x2));
    double ny = std::max(y1 - CY, std::max(0.0, CY - y2));
    if (nx * nx + ny * ny >= R * R)
        return full;
    double fx = std::max(std::fabs(x1 - CX), std::fabs(x2 - CX));
    double fy = std::max(std::fabs(y1 - CY), std::fabs(y2 - CY));
    if (fx * fx + fy * fy <= R * R)
        return 0;
    return std::min(full, std::max(0.0, full - disk_area(x1, x2, y1, y2)));
}

// Коэффициенты (11) в узле: доля клетки внутри D frac, F = frac (f = 1), Q = (1 - frac) / eps^2, eps = h^2.
inline void fictitious_coeffs(double x, double y, double h1, double h2, double &Q, double &F)
{
    double h = std::max(h1, h2), eps = h * h;
    double frac = area_D(x - 0.5 * h1, x + 0.5 * h1, y - 0.5 * h2, y + 0.5 * h2) / (h1 * h2);
    Q = (1 - frac) / (eps * eps);
    F = frac;
}

// Тестовая задача с точным решением u = sin(x+y) во всём квадрате: Q = 0, F = -Δu = 2 sin(x+y).
inline void exact_coeffs(double x, double y, double, double, double &Q, double &F)
{
    Q = 0;
    F = 2 * std::sin(x + y);
}

#endif
