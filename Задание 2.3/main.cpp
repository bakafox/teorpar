#include <iostream>
#include <math.h>
#include <time.h>
#include <omp.h>

#define THAU 0.01
#define EPSILON 1.e-5

using namespace std;

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}



void prepare_values(double *a, double *b, double *c, int cols, int rows) {
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            a[i * rows + j] = (i == j) ? 2.0 : 1.0;
        }
        b[i] = cols + 1;
    }
}

void iterate(double *a, double *b, double *c, int cols, int rows) {
    for (int i = 0; i < cols; i++) {
        c[i] = 0.0;

        for (int j = 0; j < rows; j++) {
            c[i] += a[i * rows + j] * b[j];
        }
    }
}



double run_serial(int cols, int rows) {
    double *a = new double[cols * rows];
    double *b = new double[cols];
    double *c = new double[cols];
    prepare_values(a, b, c, cols, rows);

    double t = cpuSecond();

    double delta;
    do {
        double *temp = new double[cols];
        iterate(a, b, temp, cols, rows);
        delta = 0.0;
        for (int i = 0; i < cols; i++) {
            delta += fabs(temp[i] - b[i]);
            b[i] = temp[i];
        }
        delete[] temp;
    } while (delta > EPSILON);

    t = cpuSecond() - t;

    delete[] a;
    delete[] b;
    delete[] c;
    return t * 1000; // return value in ms
}

double run_parallel_var1(int cols, int rows) {
    double *a = new double[cols * rows];
    double *b = new double[cols];
    double *c = new double[cols];
    prepare_values(a, b, c, cols, rows);

    double t = cpuSecond();

    double delta;
    do {
        double *temp = new double[cols];
        #pragma omp parallel for
        for (int i = 0; i < cols; i++) {
            temp[i] = 0.0;

            for (int j = 0; j < rows; j++) {
                temp[i] += a[i * rows + j] * b[j];
            }
        }

        delta = 0.0;
        #pragma omp parallel for
        for (int i = 0; i < cols; i++) {
            delta += fabs(temp[i] - b[i]);
            b[i] = temp[i];
        }
        delete[] temp;
    } while (delta > EPSILON);

    t = cpuSecond() - t;

    delete[] a;
    delete[] b;
    delete[] c;
    return t * 1000; // return value in ms
}

double run_parallel_var2(int cols, int rows) {
    double *a = new double[cols * rows];
    double *b = new double[cols];
    double *c = new double[cols];
    prepare_values(a, b, c, cols, rows);

    double t = cpuSecond();

    double delta;
    do {
        double *temp = new double[cols];

        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < cols; i++) {
                temp[i] = 0.0;

                for (int j = 0; j < rows; j++) {
                    temp[i] += a[i * rows + j] * b[j];
                }
            }

            for (int i = 0; i < cols; i++) {
                delta += fabs(temp[i] - b[i]);
                b[i] = temp[i];
            }
        }
        delete[] temp;
    } while (delta > EPSILON);

    t = cpuSecond() - t;

    delete[] a;
    delete[] b;
    delete[] c;
    return t * 1000; // return value in ms
}

int main() {
    int threads[] = { 1, 2, 4, 7, 8, 16, 20, 40 };

    // run everything explicitly serial - this is not a mistake
    printf("\n=== SERIAL ===\n");
    double serial_results;
    serial_results = run_serial(2000, 2000);
    printf("Elapsed time: %.6f ms\n", serial_results);

    printf("\n=== PARALLEL ===\n");
    for (int i = 0; i < 8; i++) {
        omp_set_num_threads(threads[i]);
        printf("Number of threads: %d\n", omp_get_max_threads());

        printf("--- VAR 1 ---\n");
        double parallel_var1_results;
        parallel_var1_results = run_parallel_var1(2000, 2000);
        printf("Elapsed time: %.6f ms\n", parallel_var1_results);
        printf("Accelerarion ratio: %.6f\n", serial_results / parallel_var1_results);

        printf("\n--- VAR 2 ---\n");
        double parallel_var2_results;
        parallel_var2_results = run_parallel_var2(2000, 2000);
        printf("Elapsed time: %.6f ms\n", parallel_var2_results);
        printf("Accelerarion ratio: %.6f\n", serial_results / parallel_var2_results);

        if (parallel_var1_results <= parallel_var2_results) {
            printf("\nVAR 1 is faster on %.6f ms\n", parallel_var1_results - parallel_var2_results);
        }
        else {
            printf("\nVAR 2 is faster on %.6f ms\n", parallel_var2_results - parallel_var1_results);
        }
        printf("------\n");
    }

    return 0;
}