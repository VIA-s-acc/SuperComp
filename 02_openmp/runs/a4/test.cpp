#include <cmath>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <vector>
#include "solver.h"

static int failures = 0;

static void check(bool ok, const char *name, const char *details)
{
    std::printf("%s  %-56s %s\n", ok ? "PASS" : "FAIL", name, details);
    if (!ok)
        failures++;
}

static void test_area()
{
    const double pi = std::acos(-1.0);
    char buf[128];

    double whole = area_D(-1, 1, -1, 1);
    std::snprintf(buf, sizeof buf, "%.15f vs %.15f", whole, 4 - pi / 4);
    check(std::fabs(whole - (4 - pi / 4)) < 1e-12, "area: whole square = 4 - pi/4", buf);

    double corner = area_D(0, 1, 0, 1);
    std::snprintf(buf, sizeof buf, "%.15f vs %.15f", corner, 1 - pi / 4);
    check(std::fabs(corner - (1 - pi / 4)) < 1e-12, "area: [0,1]^2 = 1 - pi/4", buf);

    double left = area_D(-1, 0, -1, 1);
    std::snprintf(buf, sizeof buf, "%.15f", left);
    check(std::fabs(left - 2) < 1e-12, "area: left half has no disk part = 2", buf);

    double inside = area_D(0.9, 1.0, 0.9, 1.0);
    std::snprintf(buf, sizeof buf, "%.3e", inside);
    check(inside == 0, "area: cell inside the disk = 0", buf);

    const int n = 80;
    const double h = 2.0 / n;
    double sum = 0;
    bool bounded = true;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double a = area_D(-1 + i * h, -1 + (i + 1) * h, -1 + j * h, -1 + (j + 1) * h);
            bounded = bounded && a >= 0 && a <= h * h * (1 + 1e-12);
            sum += a;
        }
    std::snprintf(buf, sizeof buf, "error %.2e", std::fabs(sum - (4 - pi / 4)));
    check(std::fabs(sum - (4 - pi / 4)) < 1e-12, "area: sum over 80x80 cells = 4 - pi/4", buf);
    check(bounded, "area: 0 <= S_ij <= h1*h2 for every cell", "");
}

static void test_reference()
{
    const int grids[3] = {20, 40, 80}, iters_ref[3] = {44, 85, 167};
    const double w00_ref[3] = {0.240194487380, 0.239041584641, 0.241097385417};
    char buf[128];
    for (int t = 0; t < 3; t++) {
        Grid g(grids[t], grids[t]);
        setup(g, fictitious_coeffs);
        int iters = cg(g, 1e-6);
        double w00 = g.at(grids[t] / 2, grids[t] / 2);
        std::snprintf(buf, sizeof buf, "iters %d, w(0,0) = %.12f", iters, w00);
        char name[64];
        std::snprintf(name, sizeof name, "reference %dx%d: iters and w(0,0)", grids[t], grids[t]);
        check(iters == iters_ref[t] && std::fabs(w00 - w00_ref[t]) < 1e-9, name, buf);
    }
}

static double exact_error(int n)
{
    Grid g(n, n);
    setup(g, exact_coeffs);
    cg(g, 1e-10);
    double err = 0;
    for (int i = 1; i < n; i++)
        for (int j = 1; j < n; j++)
            err = std::max(err, std::fabs(g.at(i, j) - v0(A1 + i * g.h1, A2 + j * g.h2)));
    return err;
}

static void test_exact_solution()
{
    double e20 = exact_error(20), e40 = exact_error(40), e80 = exact_error(80);
    char buf[160];
    std::snprintf(buf, sizeof buf, "max error %.2e, %.2e, %.2e; ratios %.2f, %.2f",
                  e20, e40, e80, e20 / e40, e40 / e80);
    check(e80 < 1e-4 && e20 / e40 > 3.5 && e40 / e80 > 3.5,
          "exact u = sin(x+y): error drops ~4x per refinement", buf);
}

static void test_fictitious_region()
{
    const int grids[3] = {20, 40, 80};
    double prev = 1;
    bool ok = true;
    char buf[160] = "";
    for (int t = 0; t < 3; t++) {
        int n = grids[t];
        Grid g(n, n);
        setup(g, fictitious_coeffs);
        cg(g, 1e-6);
        double dev = 0;
        for (int i = 1; i < n; i++)
            for (int j = 1; j < n; j++) {
                double x = A1 + i * g.h1, y = A2 + j * g.h2;
                if (area_D(x - 0.5 * g.h1, x + 0.5 * g.h1, y - 0.5 * g.h2, y + 0.5 * g.h2) == 0)
                    dev = std::max(dev, std::fabs(g.at(i, j) - v0(x, y)));
            }
        ok = ok && dev < 1e-3 && dev < prev;
        prev = dev;
        std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf), "%s%.1e", t ? ", " : "", dev);
    }
    check(ok, "fictitious region: |w - v0| small and decreasing", buf);
}

static void test_threads()
{
    const int n = 80;
    std::vector<double> base;
    int base_iters = 0;
    bool ok = true;
    double worst = 0;
    const int counts[4] = {1, 2, 4, 8};
    for (int c = 0; c < 4; c++) {
        omp_set_num_threads(counts[c]);
        Grid g(n, n);
        setup(g, fictitious_coeffs);
        int iters = cg(g, 1e-6);
        if (c == 0) {
            base.assign(g.w, g.w + g.n);
            base_iters = iters;
            continue;
        }
        for (size_t k = 0; k < g.n; k++)
            worst = std::max(worst, std::fabs(g.w[k] - base[k]));
        ok = ok && iters == base_iters;
    }
    char buf[128];
    std::snprintf(buf, sizeof buf, "max |w_T - w_1| = %.1e over T = 2, 4, 8", worst);
    check(ok && worst < 1e-12, "threads: same solution on 1, 2, 4, 8 threads", buf);
}

static void test_history()
{
    Grid g(80, 80);
    setup(g, fictitious_coeffs);
    std::vector<IterLog> log;
    int iters = cg(g, 1e-6, &log);
    char buf[160];
    std::snprintf(buf, sizeof buf, "%zu entries, last dw %.2e, residual %.2e -> %.2e",
                  log.size(), log.back().dw, log.front().res, log.back().res);
    check(int(log.size()) == iters && log.back().dw < 1e-6 && log.back().res < 1e-3 * log.front().res,
          "history: one entry per iteration, stops at dw < delta", buf);
}

int main()
{
    test_area();
    test_reference();
    test_exact_solution();
    test_fictitious_region();
    test_threads();
    test_history();
    std::printf("%s: %d failed\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
