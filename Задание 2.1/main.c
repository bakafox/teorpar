#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

//#define MATRIX_COLS 40000 // 20000 или 40000
//#define MATRIX_ROWS MATRIX_COLS
//#define MAX_THREADS 40 // 1, 2, 4, 7, 8, 16, 20, 40

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}



void matrix_vector_product(double *a, double *b, double *c, int cols, int rows) {
    for (int i = 0; i < cols; i++) {
        c[i] = 0.0;

        for (int j = 0; j < rows; j++) {
            c[i] += a[i * rows + j] * b[j];
        }
    }
}

void matrix_vector_product_omp(double *a, double *b, double *c, int cols, int rows) {
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = cols / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (cols - 1) : (lb + items_per_thread - 1);

        // параллельно-вычисляемый цикл FOR,
        // каждый поток вычисляет только 1/n-тую
        // часть всех значений (от lb до ub)
        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0; // Store – запись в память
            for (int j = 0; j < rows; j++) {
                // Load c[i], Load a[i][j], Load b[j], Store c[i]
                c[i] += a[i * rows + j] * b[j];
            }
        }
    }
}



double run_serial(int cols, int rows) {
    double *a, *b, *c;
    a = malloc(sizeof(*a) * cols * rows);
    b = malloc(sizeof(*b) * rows);
    c = malloc(sizeof(*c) * cols);

    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            a[i * rows + j] = i + j;
        }
    }
    for (int j = 0; j < rows; j++) {
        b[j] = j;
    }

    double t = cpuSecond();
    matrix_vector_product(a, b, c, cols, rows);
    t = cpuSecond() - t;

    free(a);
    free(b);
    free(c);
    return t * 1000; // возвращаем значение в мс
}

double run_parallel(int cols, int rows) {
    double *a, *b, *c;
    a = malloc(sizeof(*a) * cols * rows);
    b = malloc(sizeof(*b) * rows);
    c = malloc(sizeof(*c) * cols);

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = cols / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (cols - 1) : (lb + items_per_thread - 1);

        // параллельно-вычисляемый цикл FOR,
        // каждый поток вычисляет только 1/n-тую
        // часть всех значений (от lb до ub)
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < rows; j++) {
                a[i * rows + j] = i + j;
            }
            c[i] = 0.0;
        }
    }
    for (int j = 0; j < rows; j++) {
        b[j] = j;
    }

    double t = cpuSecond();
    matrix_vector_product_omp(a, b, c, cols, rows);
    t = cpuSecond() - t;

    free(a);
    free(b);
    free(c);
    return t * 1000; // возвращаем значение в мс
}



int main() {
    int threads[] = { 1, 2, 4, 7, 8, 16, 20, 40 };

    // намеренно всё запускаем последовательно - это не ошибка!
    printf("\n=== SERIAL ===\n");
    double serial_results[2];
    serial_results[0] = run_serial(20000, 20000);
    printf("20K elapsed time: %.6f ms\n", serial_results[0]);
    serial_results[1] = run_serial(40000, 40000);
    printf("40K elapsed time: %.6f ms\n", serial_results[1]);

    printf("\n=== PARALLEL ===\n");
    for (int i = 0; i < 8; i++) {
        omp_set_num_threads(threads[i]);
        printf("Number of threads: %d\n", omp_get_max_threads());
        double parallel_results[2];
        parallel_results[0] = run_parallel(20000, 20000);
        printf("20K elapsed time: %.6f ms\n", parallel_results[0]);
        printf("20K accelerarion ratio: %.6f\n", serial_results[0] / parallel_results[0]);
        parallel_results[1] = run_parallel(40000, 40000);
        printf("40K elapsed time: %.6f ms\n", parallel_results[1]);
        printf("40K accelerarion ratio: %.6f\n", serial_results[1] / parallel_results[1]);
        printf("------\n");
    }

    return 0;
}
