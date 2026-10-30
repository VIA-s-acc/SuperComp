#ifndef SOLVER_H
#define SOLVER_H

#include <cmath>
#include <cstddef>
#include <vector>
#include "problem.h"

struct Grid {
    int M, N, s;
    double h1, h2;
    size_t n, stride;
    std::vector<double> mem;
    double *D, *w, *r, *p, *Ap;

    // Все массивы в одном блоке со сдвигом: иначе на POWER8 они попадают в одни наборы кэша.
    Grid(int M_, int N_)
        : M(M_), N(N_), s(N_ + 1), h1((B1 - A1) / M_), h2((B2 - A2) / N_),
          n(size_t(M_ + 1) * (N_ + 1)), stride((n + 8191) / 8192 * 8192 + 1664), mem(5 * stride, 0),
          D(&mem[0]), w(&mem[stride]), r(&mem[2 * stride]), p(&mem[3 * stride]), Ap(&mem[4 * stride]) {}

    double at(int i, int j) const { return w[i * s + j]; }
};

typedef void (*Coeffs)(double x, double y, double h1, double h2, double &Q, double &F);

// Граница: w = v0 (13). Внутри: D = 2/h1^2 + 2/h2^2 + Q, в r — правая часть B = F + Q*v0 из (12).
inline void setup(Grid &g, Coeffs coeffs)
{
    const double ih1 = 1 / (g.h1 * g.h1), ih2 = 1 / (g.h2 * g.h2);
    const int M = g.M, N = g.N, s = g.s;
    #pragma omp parallel for
    for (int i = 0; i <= M; i++)
        for (int j = 0; j <= N; j++) {
            double x = A1 + i * g.h1, y = A2 + j * g.h2;
            int k = i * s + j;
            if (i == 0 || i == M || j == 0 || j == N) {
                g.w[k] = v0(x, y);
                continue;
            }
            double Q, F;
            coeffs(x, y, g.h1, g.h2, Q, F);
            g.D[k] = 2 * ih1 + 2 * ih2 + Q;
            g.r[k] = F + Q * v0(x, y);
        }
}

// Одна строка истории: номер итерации, ||w(k) - w(k-1)||_E, ||r(k)||_E и, если задано эталонное
// решение, ошибка ||w(k) - w*|| в нормах C и E.
struct IterLog {
    int k;
    double dw, res, err_c, err_e;
};

// Метод сопряжённых градиентов (раздел 5). Возвращает число итераций.
inline int cg(Grid &g, double delta, std::vector<IterLog> *log = 0, const double *ref = 0)
{
    const int M = g.M, N = g.N, s = g.s;
    const double ih1 = 1 / (g.h1 * g.h1), ih2 = 1 / (g.h2 * g.h2), weight = g.h1 * g.h2;
    double *D = g.D, *w = g.w, *r = g.r, *p = g.p, *Ap = g.Ap;

    // Каждый цикл по узлам делится между нитями по строкам i.
    // Скалярные произведения — через reduction: у каждой нити своя частичная сумма.
    // Старт CG: r0 = B - A*w0, z0 = r0 / D, p1 = z0.
    double zr = 0;
    #pragma omp parallel for reduction(+:zr)
    for (int i = 1; i < M; i++)
        for (int j = 1; j < N; j++) {
            int k = i * s + j;
            double aw = D[k] * w[k] - ih1 * (w[k - s] + w[k + s]) - ih2 * (w[k - 1] + w[k + 1]);
            r[k] -= aw;
            p[k] = r[k] / D[k];
            zr += p[k] * r[k];
        }

    const int max_iter = (M - 1) * (N - 1);
    int it;
    for (it = 1; it <= max_iter; it++) {
        // Ap = A*p, alpha = (z, r) / (Ap, p).
        double pAp = 0, pp = 0;
        #pragma omp parallel for reduction(+:pAp, pp)
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                int k = i * s + j;
                Ap[k] = D[k] * p[k] - ih1 * (p[k - s] + p[k + s]) - ih2 * (p[k - 1] + p[k + 1]);
                pAp += Ap[k] * p[k];
                pp += p[k] * p[k];
            }
        double alpha = zr / pAp;

        // w += alpha*p (15), r -= alpha*Ap, новое (z, r) и (r, r).
        double zr_new = 0, rr = 0;
        #pragma omp parallel for reduction(+:zr_new, rr)
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                int k = i * s + j;
                w[k] += alpha * p[k];
                r[k] -= alpha * Ap[k];
                zr_new += r[k] * r[k] / D[k];
                rr += r[k] * r[k];
            }

        // Остановка (16): ||w(k+1) - w(k)|| = |alpha| * ||p||.
        double dw = std::fabs(alpha) * std::sqrt(weight * pp);
        if (log) {
            IterLog e = {it, dw, std::sqrt(weight * rr), 0, 0};
            if (ref) {
                double emax = 0, e2 = 0;
                #pragma omp parallel for reduction(max:emax) reduction(+:e2)
                for (int i = 1; i < M; i++)
                    for (int j = 1; j < N; j++) {
                        double d = std::fabs(w[i * s + j] - ref[i * s + j]);
                        emax = std::max(emax, d);
                        e2 += d * d;
                    }
                e.err_c = emax;
                e.err_e = std::sqrt(weight * e2);
            }
            log->push_back(e);
        }
        if (dw < delta)
            return it;

        // p = z + beta*p.
        double beta = zr_new / zr;
        zr = zr_new;
        #pragma omp parallel for
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                int k = i * s + j;
                p[k] = r[k] / D[k] + beta * p[k];
            }
    }
    return max_iter;
}

#endif
