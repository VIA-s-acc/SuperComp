#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

const int ATTEMPT = 2;
const double A1 = -1, B1 = 1, A2 = -1, B2 = 1;

static double v0(double x, double y) { return std::sin(x + y); }

static bool in_D(double x, double y) { return (x - 1) * (x - 1) + (y - 1) * (y - 1) > 1; }

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
    std::vector<double> Q(n, 0), w(n, 0), r(n, 0), p(n, 0), Ap(n, 0);

    // Сетка (M+1)x(N+1) узлов, узел (i,j) хранится под индексом k = i*(N+1) + j.
    // Граничные узлы: значение Дирихле v0 (13), дальше не меняются.
    // Внутренние: Q и F по (11); в r сначала кладётся правая часть B = F + Q*v0 из (12).
    // Попытки 1-2: клетка целиком в D или целиком вне D — по положению её центрального узла.
    for (int i = 0; i <= M; i++)
        for (int j = 0; j <= N; j++) {
            double x = A1 + i * h1, y = A2 + j * h2;
            int k = i * s + j;
            if (i == 0 || i == M || j == 0 || j == N) {
                w[k] = v0(x, y);
                continue;
            }
            double F = in_D(x, y) ? 1 : 0;
            Q[k] = in_D(x, y) ? 0 : 1 / (eps * eps);
            r[k] = F + Q[k] * v0(x, y);
        }

    const double ih1 = 1 / (h1 * h1), ih2 = 1 / (h2 * h2), weight = h1 * h2;

    // Старт CG (раздел 5): внутри w0 = 0, поэтому r0 = B - A*w0 учитывает только граничные значения.
    // A*w — пятиточечный шаблон (12): (2/h1^2 + 2/h2^2 + Q)*w_ij минус четыре соседа.
    // z0 = r0 / D, диагональный предобуславливатель D = 2/h1^2 + 2/h2^2 + Q; первое направление p1 = z0.
    // zr = (z, r). Вес h1*h2 скалярного произведения (8) сокращается в alpha и beta.
    double zr = 0;
    for (int i = 1; i < M; i++)
        for (int j = 1; j < N; j++) {
            int k = i * s + j;
            double d = 2 * ih1 + 2 * ih2 + Q[k];
            double aw = d * w[k] - ih1 * (w[k - s] + w[k + s]) - ih2 * (w[k - 1] + w[k + 1]);
            r[k] -= aw;
            p[k] = r[k] / d;
            zr += p[k] * r[k];
        }

    const int max_iter = (M - 1) * (N - 1);
    int it;
    for (it = 1; it <= max_iter; it++) {
        // Ap = A*p, alpha = (z, r) / (Ap, p); pp = (p, p) нужно для условия остановки.
        double pAp = 0, pp = 0;
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                int k = i * s + j;
                Ap[k] = (2 * ih1 + 2 * ih2 + Q[k]) * p[k] - ih1 * (p[k - s] + p[k + s]) - ih2 * (p[k - 1] + p[k + 1]);
                pAp += Ap[k] * p[k];
                pp += p[k] * p[k];
            }
        double alpha = zr / pAp;

        // w += alpha*p (15), r -= alpha*Ap и новое (z, r), где z = r / D.
        double zr_new = 0;
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                int k = i * s + j;
                w[k] += alpha * p[k];
                r[k] -= alpha * Ap[k];
                zr_new += r[k] * r[k] / (2 * ih1 + 2 * ih2 + Q[k]);
            }

        // Остановка (16): w(k+1) - w(k) = alpha*p, его норма (8) = |alpha| * sqrt(h1*h2*(p, p)).
        if (std::fabs(alpha) * std::sqrt(weight * pp) < delta)
            break;

        // beta = (z_k, r_k) / (z_{k-1}, r_{k-1}), следующее направление p = z + beta*p.
        double beta = zr_new / zr;
        zr = zr_new;
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                int k = i * s + j;
                p[k] = r[k] / (2 * ih1 + 2 * ih2 + Q[k]) + beta * p[k];
            }
    }

    double time = wtime() - t0;
    if (it > max_iter)
        std::fprintf(stderr, "WARNING: iteration limit %d reached\n", max_iter);

    std::printf("RESULT impl=serial attempt=%d M=%d N=%d P=1 T=1 iters=%d time=%.6f w00=%.12f\n",
                ATTEMPT, M, N, std::min(it, max_iter), time, w[(M / 2) * s + N / 2]);
    return 0;
}
