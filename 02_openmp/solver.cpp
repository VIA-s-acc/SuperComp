#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <stdint.h>
#include <vector>
#include "solver.h"

const int ATTEMPT = 4;

static double wtime()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Формат дампа: int32 M, int32 N, затем (M+1)(N+1) double, узел (i,j) под индексом i*(N+1)+j.
static bool write_dump(const char *path, const Grid &g)
{
    FILE *out = std::fopen(path, "wb");
    int32_t mn[2] = {g.M, g.N};
    bool ok = out && std::fwrite(mn, sizeof mn, 1, out) == 1 && std::fwrite(g.w, sizeof(double), g.n, out) == g.n;
    if (out)
        ok = std::fclose(out) == 0 && ok;
    return ok;
}

static bool read_dump(const char *path, const Grid &g, std::vector<double> &ref)
{
    FILE *in = std::fopen(path, "rb");
    int32_t mn[2] = {0, 0};
    ref.assign(g.n, 0);
    bool ok = in && std::fread(mn, sizeof mn, 1, in) == 1 && mn[0] == g.M && mn[1] == g.N
              && std::fread(&ref[0], sizeof(double), g.n, in) == g.n;
    if (in)
        std::fclose(in);
    return ok;
}

int main(int argc, char **argv)
{
    const char *dump = 0, *history = 0, *ref_path = 0;
    bool exact = false;
    std::vector<const char *> pos;
    for (int a = 1; a < argc; a++) {
        if (std::strcmp(argv[a], "--dump") == 0 && a + 1 < argc)
            dump = argv[++a];
        else if (std::strcmp(argv[a], "--history") == 0 && a + 1 < argc)
            history = argv[++a];
        else if (std::strcmp(argv[a], "--ref") == 0 && a + 1 < argc)
            ref_path = argv[++a];
        else if (std::strcmp(argv[a], "--exact") == 0)
            exact = true;
        else
            pos.push_back(argv[a]);
    }
    if (pos.size() < 2) {
        std::fprintf(stderr, "usage: %s M N [delta] [--dump file] [--history file] [--ref file] [--exact]\n", argv[0]);
        return 2;
    }
    const int M = std::atoi(pos[0]), N = std::atoi(pos[1]);
    const double delta = pos.size() > 2 ? std::atof(pos[2]) : 1e-6;

    double t0 = wtime();
    Grid g(M, N);
    setup(g, exact ? exact_coeffs : fictitious_coeffs);

    // Эталон для ошибки по итерациям: решение из файла или, для тестовой задачи (--exact), u = sin(x+y).
    std::vector<double> ref;
    if (ref_path && !read_dump(ref_path, g, ref)) {
        std::fprintf(stderr, "cannot read reference %s for grid %dx%d\n", ref_path, M, N);
        return 1;
    }
    if (exact && !ref_path) {
        ref.resize(g.n);
        for (int i = 0; i <= M; i++)
            for (int j = 0; j <= N; j++)
                ref[i * g.s + j] = v0(A1 + i * g.h1, A2 + j * g.h2);
    }
    std::vector<IterLog> log;
    int iters = cg(g, delta, history ? &log : 0, ref.empty() ? 0 : &ref[0]);
    double time = wtime() - t0;

    if (iters == (M - 1) * (N - 1))
        std::fprintf(stderr, "WARNING: iteration limit %d reached\n", iters);
    std::printf("RESULT impl=%s attempt=%d M=%d N=%d P=1 T=%d iters=%d time=%.6f w00=%.12f\n",
                exact ? "omp-exact" : "omp", ATTEMPT, M, N, omp_get_max_threads(), iters, time, g.at(M / 2, N / 2));

    if (dump && !write_dump(dump, g)) {
        std::fprintf(stderr, "cannot write %s\n", dump);
        return 1;
    }
    if (history) {
        FILE *out = std::fopen(history, "w");
        if (!out) {
            std::fprintf(stderr, "cannot write %s\n", history);
            return 1;
        }
        std::fprintf(out, "k dw res err_c err_e\n");
        for (size_t i = 0; i < log.size(); i++)
            std::fprintf(out, "%d %.6e %.6e %.6e %.6e\n", log[i].k, log[i].dw, log[i].res, log[i].err_c, log[i].err_e);
        std::fclose(out);
    }
    return 0;
}
