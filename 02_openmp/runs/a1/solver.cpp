#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <vector>

const int ATTEMPT = 1;
const double A1 = -1, B1 = 1, A2 = -1, B2 = 1;

static double v0(double x, double y) { return std::sin(x + y); }

const double CX = 1, CY = 1, R = 1;

// Площадь круга (центр 0) в [0,a]x[0,b], точно через первообразную sqrt(R^2 - x^2).
static double quadrant_area(double a, double b)
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
static double signed_quadrant_area(double a, double b)
{
    double sign = (a < 0) == (b < 0) ? 1 : -1;
    return sign * quadrant_area(std::fabs(a), std::fabs(b));
}

// Площадь прямоугольник ∩ круг: включение-исключение по четырём углам.
static double disk_area(double x1, double x2, double y1, double y2)
{
    x1 -= CX; x2 -= CX;
    y1 -= CY; y2 -= CY;
    return signed_quadrant_area(x2, y2) - signed_quadrant_area(x1, y2)
         - signed_quadrant_area(x2, y1) + signed_quadrant_area(x1, y1);
}

// S_ij = mes(клетка ∩ D) из раздела 4.
static double area_D(double x1, double x2, double y1, double y2)
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

static double wtime()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s M N [delta]\n", argv[0]);
        return 2;
    }
    const int M = std::atoi(argv[1]), N = std::atoi(argv[2]);
    const double delta = argc > 3 ? std::atof(argv[3]) : 1e-6;

    double t0 = wtime();

    const double h1 = (B1 - A1) / M, h2 = (B2 - A2) / N;
    const double h = std::max(h1, h2), eps = h * h;
    const int s = N + 1;
    const size_t n = size_t(M + 1) * s;
    const double ih1 = 1 / (h1 * h1), ih2 = 1 / (h2 * h2), weight = h1 * h2;

    // Все массивы в одном блоке со сдвигом: иначе на POWER8 они попадают в одни наборы кэша.
    const size_t stride = (n + 8191) / 8192 * 8192 + 1664;
    std::vector<double> mem(5 * stride, 0);
    double *D = &mem[0], *w = &mem[stride], *r = &mem[2 * stride], *p = &mem[3 * stride], *Ap = &mem[4 * stride];

    // Граница: w = v0 (13). Внутри: F, Q по (11), D = 2/h1^2 + 2/h2^2 + Q, в r — B = F + Q*v0 из (12).
    #pragma omp parallel for
    for (int i = 0; i <= M; i++)
        for (int j = 0; j <= N; j++) {
            double x = A1 + i * h1, y = A2 + j * h2;
            int k = i * s + j;
            if (i == 0 || i == M || j == 0 || j == N) {
                w[k] = v0(x, y);
                continue;
            }
            double frac = area_D(x - 0.5 * h1, x + 0.5 * h1, y - 0.5 * h2, y + 0.5 * h2) / (h1 * h2);
            double Q = (1 - frac) / (eps * eps);
            D[k] = 2 * ih1 + 2 * ih2 + Q;
            r[k] = frac + Q * v0(x, y);
        }

    // Каждый цикл по узлам делится между нитями по строкам i.
    // Старт CG (раздел 5): r0 = B - A*w0, z0 = r0 / D, p1 = z0.
    double zr = 0;
    #pragma omp parallel for
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
        #pragma omp parallel for
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                int k = i * s + j;
                Ap[k] = D[k] * p[k] - ih1 * (p[k - s] + p[k + s]) - ih2 * (p[k - 1] + p[k + 1]);
                pAp += Ap[k] * p[k];
                pp += p[k] * p[k];
            }
        double alpha = zr / pAp;

        // w += alpha*p (15), r -= alpha*Ap, новое (z, r).
        double zr_new = 0;
        #pragma omp parallel for
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                int k = i * s + j;
                w[k] += alpha * p[k];
                r[k] -= alpha * Ap[k];
                zr_new += r[k] * r[k] / D[k];
            }

        // Остановка (16): ||w(k+1) - w(k)|| = |alpha| * ||p||.
        if (std::fabs(alpha) * std::sqrt(weight * pp) < delta)
            break;

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

    double time = wtime() - t0;
    if (it > max_iter)
        std::fprintf(stderr, "WARNING: iteration limit %d reached\n", max_iter);

    std::printf("RESULT impl=omp attempt=%d M=%d N=%d P=1 T=%d iters=%d time=%.6f w00=%.12f\n",
                ATTEMPT, M, N, omp_get_max_threads(), std::min(it, max_iter), time, w[(M / 2) * s + N / 2]);
    return 0;
}
