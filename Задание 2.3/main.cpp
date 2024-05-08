#include <iostream>
#include <math.h>
#include <time.h>
#include <omp.h>

#define THAU 1.e-4
#define EPSILON 1.e-7

using namespace std;

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}



void prepare_values(double *a, double *b, double *x, int cols, int rows) {
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            a[i * rows + j] = (i == j) ? 2.0 : 1.0;
        }
        b[i] = cols + 1;
        x[i] = 0.0;
    }
}



double run_serial(int cols, int rows) {
    double *a = new double[cols * rows];
    double *b = new double[cols];
    double *x = new double[cols];
    prepare_values(a, b, x, cols, rows);

    double t = cpuSecond();

    double delta = 10.0 * EPSILON;
    while (delta > EPSILON) {
        double *prod = new double[cols];

        for (int i = 0; i < cols; i++) {
            prod[i] = 0.0;

            for (int j = 0; j < rows; j++) {
                prod[i] += a[i * rows + j] * x[j];
            }
        }

        delta = 0.0;
        for (int i = 0; i < cols; i++) {
            double diff = fabs(prod[i] - b[i]) / fabs(b[i]);
            delta += diff;
            x[i] = x[i] - THAU * (prod[i] - b[i]);
        }

        delete[] prod;
    }

    t = cpuSecond() - t;

    delete[] a;
    delete[] b;
    delete[] x;
    return t * 1000; // возвращаем значение в мс
}

double run_parallel_var1(int cols, int rows) {
    // для каждого распараллеливаемого цикла создается
    // отдельная параллельная секция #pragma omp parallel for
    double *a = new double[cols * rows];
    double *b = new double[cols];
    double *x = new double[cols];
    prepare_values(a, b, x, cols, rows);

    double t = cpuSecond();

    double delta = 10.0 * EPSILON;
    while (delta > EPSILON) {
        double *prod = new double[cols];

        #pragma omp parallel
        {
            #pragma omp for schedule(static) // методом научного тыка установлено,
                                             // что лучше всего для операций
                                             // подходит режим распред-я "static".
            for (int i = 0; i < cols; i++) {
                prod[i] = 0.0;

                for (int j = 0; j < rows; j++) {
                    prod[i] += a[i * rows + j] * x[j];
                }
            }
        }

        delta = 0.0;
        #pragma omp parallel
        {
            double localdelta = 0.0;
            #pragma omp for schedule(static)
            for (int i = 0; i < cols; i++) {
                double diff = fabs(prod[i] - b[i]) / fabs(b[i]);
                localdelta += diff;
                x[i] = x[i] - THAU * (prod[i] - b[i]);
            }
            #pragma omp atomic // предотвращаем одновременные обращения потоков
            delta += localdelta;
        }

        delete[] prod;
    }

    t = cpuSecond() - t;

    delete[] a;
    delete[] b;
    delete[] x;
    return t * 1000; // возвращаем значение в мс
}

double run_parallel_var2(int cols, int rows) {
    // создается одна параллельная секция #pragma omp
    // parallel, охватывающая весь итерационный алгоритм.
    double *a = new double[cols * rows];
    double *b = new double[cols];
    double *x = new double[cols];
    prepare_values(a, b, x, cols, rows);

    double t = cpuSecond();

    double delta = 10.0 * EPSILON;
    while (delta > EPSILON) {
        double *prod = new double[cols];

        delta = 0.0;
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < cols; i++) {
                prod[i] = 0.0;

                for (int j = 0; j < rows; j++) {
                    prod[i] += a[i * rows + j] * x[j];
                }
            }

            double localdelta = 0.0;
            #pragma omp for schedule(static)
            for (int i = 0; i < cols; i++) {
                double diff = fabs(prod[i] - b[i]) / fabs(b[i]);
                localdelta += diff;
                x[i] = x[i] - THAU * (prod[i] - b[i]);
            }
            #pragma omp atomic // предотвращаем одновременные обращения потоков
            delta += localdelta;

        }
        delete[] prod;
    }

    t = cpuSecond() - t;

    delete[] a;
    delete[] b;
    delete[] x;
    return t * 1000; // возвращаем значение в мс
}

int main() {
    int threads[] = { 1, 2, 4, 7, 8, 16, 20, 40, 60, 80 };
    int SIZE = 15000; // ~50 секунд на 1 ядре

    // намеренно всё запускаем последовательно - это не ошибка!
    printf("\n=== SERIAL ===\n");
    double serial_results;
    serial_results = run_serial(SIZE, SIZE);
    printf("\nElapsed time: %.6f ms\n", serial_results);

    printf("\n=== PARALLEL ===\n");
    for (int i = 0; i < 10; i++) {
        omp_set_num_threads(threads[i]);
        printf("Number of threads: %d\n", omp_get_max_threads());

        double parallel_var1_results;
        parallel_var1_results = run_parallel_var1(SIZE, SIZE);
        printf("\nVar1 elapsed time: %.6f ms\n", parallel_var1_results);
        printf("Var1 accelerarion ratio: %.6f\n", serial_results / parallel_var1_results);

        double parallel_var2_results;
        parallel_var2_results = run_parallel_var2(SIZE, SIZE);
        printf("\nVar2 elapsed time: %.6f ms\n", parallel_var2_results);
        printf("Var2 accelerarion ratio: %.6f\n", serial_results / parallel_var2_results);

        if (parallel_var1_results > parallel_var2_results) {
            printf("\nVar2 is faster on %.6f ms\n", parallel_var1_results - parallel_var2_results);
        }
        else {
            printf("\nVar1 is faster on %.6f ms\n", parallel_var2_results - parallel_var1_results);
        }
        printf("------\n");
    }

    return 0;
}
