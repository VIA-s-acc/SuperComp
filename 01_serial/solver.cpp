#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

const int ATTEMPT = 1;
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

    double rr = 0;
    for (int i = 1; i < M; i++)
        for (int j = 1; j < N; j++) {
            int k = i * s + j;
            double aw = (2 * ih1 + 2 * ih2 + Q[k]) * w[k] - ih1 * (w[k - s] + w[k + s]) - ih2 * (w[k - 1] + w[k + 1]);
            r[k] -= aw;
            p[k] = r[k];
            rr += r[k] * r[k];
        }

    const int max_iter = (M - 1) * (N - 1);
    int it;
    for (it = 1; it <= max_iter; it++) {
        double pAp = 0, pp = 0;
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                int k = i * s + j;
                Ap[k] = (2 * ih1 + 2 * ih2 + Q[k]) * p[k] - ih1 * (p[k - s] + p[k + s]) - ih2 * (p[k - 1] + p[k + 1]);
                pAp += Ap[k] * p[k];
                pp += p[k] * p[k];
            }
        double alpha = rr / pAp;

        double rr_new = 0;
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                int k = i * s + j;
                w[k] += alpha * p[k];
                r[k] -= alpha * Ap[k];
                rr_new += r[k] * r[k];
            }

        if (std::fabs(alpha) * std::sqrt(weight * pp) < delta)
            break;

        double beta = rr_new / rr;
        rr = rr_new;
        for (int i = 1; i < M; i++)
            for (int j = 1; j < N; j++) {
                int k = i * s + j;
                p[k] = r[k] + beta * p[k];
            }
    }

    double time = wtime() - t0;
    if (it > max_iter)
        std::fprintf(stderr, "WARNING: iteration limit %d reached\n", max_iter);

    std::printf("RESULT impl=serial attempt=%d M=%d N=%d P=1 T=1 iters=%d time=%.6f w00=%.12f\n",
                ATTEMPT, M, N, std::min(it, max_iter), time, w[(M / 2) * s + N / 2]);
    return 0;
}
