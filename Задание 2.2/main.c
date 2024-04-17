#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define PI 3.14159265358979323846

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}



double integrate(double (*func)(double), double a, double b, int nsteps) {
    double h = (b - a) / nsteps;
    double sum = 0.0;

    for (int i = 0; i < nsteps; i++) {
        sum += func(a + h * (i + 0.5));
    }

    return sum * h;
}

double integrate_omp(double (*func)(double), double a, double b, int nsteps) {
    double h = (b - a) / nsteps;
    double sum = 0.0;

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = nsteps / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (nsteps - 1) : (lb + items_per_thread - 1);

        double localsum = 0.0; // храним вычисленную часть суммы локально для каждого потока

        // параллельно-вычисляемый цикл FOR,
        // каждый поток вычисляет только 1/n-тую
        // часть всех значений (от lb до ub)
        for (int i = lb; i <= ub; i++) {
            localsum += func(a + h * (i + 0.5));
        }

        #pragma omp atomic // предотвращаем гонку потоков, делая операцию атомарной
            sum += localsum;
    }
    return sum * h;
}



double func(double x) {
    // просто некоторая функция, которую мы будем интегрировать
    return exp(-x * x);
}

double run_serial(double a, double b, int nsteps) {
    double t = cpuSecond();
    double res = integrate(func, a, b, nsteps);
    t = cpuSecond() - t;

    printf("\nResult: %.12f // Error: %.12f\n", res, fabs(res - sqrt(PI)));
    return t * 1000; // возвращаем значение в мс
}

double run_parallel(double a, double b, int nsteps) {
    double t = cpuSecond();
    double res = integrate_omp(func, a, b, nsteps);
    t = cpuSecond() - t;

    printf("\nResult: %.12f // Error: %.12f\n", res, fabs(res - sqrt(PI)));
    return t * 1000; // возвращаем значение в мс
}



int main() {
    int threads[] = { 1, 2, 4, 7, 8, 16, 20, 40 };

    // намеренно всё запускаем последовательно - это не ошибка!
    printf("\n=== SERIAL ===");
    double serial_results[2];
    serial_results[0] = run_serial(-4.0, 4.0, 40000000);
    printf("40M elapsed time: %.6f ms\n", serial_results[0]);
    serial_results[1] = run_serial(-4.0, 4.0, 80000000);
    printf("80M elapsed time: %.6f ms\n", serial_results[1]);

    printf("\n=== PARALLEL ===\n");
    for (int i = 0; i < 8; i++) {
        omp_set_num_threads(threads[i]);
        printf("Number of threads: %d\n", omp_get_max_threads());
        double parallel_results[2];
        parallel_results[0] = run_parallel(-4.0, 4.0, 40000000);
        printf("40M elapsed time: %.6f ms\n", parallel_results[0]);
        printf("40M accelerarion ratio: %.6f\n", serial_results[0] / parallel_results[0]);
        parallel_results[1] = run_parallel(-4.0, 4.0, 80000000);
        printf("80M elapsed time: %.6f ms\n", parallel_results[1]);
        printf("80M accelerarion ratio: %.6f\n", serial_results[1] / parallel_results[1]);
        printf("------\n");
    }

    return 0;
}
