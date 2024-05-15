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
    // теоретически, можно дополнительно ускорить параллельные
    // версии программ, если внести эту функцию в код и распараллелить
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
    return t * 1000; // возвращаем значение в мс
}

double run_parallel_var1(int cols, int rows) {
    // для каждого распараллеливаемого цикла создается
    // отдельная параллельная секция #pragma omp parallel for
    double *a = new double[cols * rows];
    double *b = new double[cols];
    double *c = new double[cols];
    prepare_values(a, b, c, cols, rows);

    double t = cpuSecond();

    double delta = 0.0;
    do
    {
        double *temp = new double[cols];
        #pragma omp parallel
        {
            for (int i = 0; i < cols; i++) {
                temp[i] = 0.0;

                for (int j = 0; j < rows; j++) {
                    temp[i] += a[i * rows + j] * b[j];
                }
            }
        }

        #pragma omp parallel
        {
            double localdelta = 0.0;
            for (int i = 0; i < cols; i++)
            {
                localdelta += fabs(temp[i] - b[i]);
                b[i] = temp[i];
            }
            #pragma omp atomic // предотвращаем одновременные обращения потоков
                delta += localdelta;
        }
        delete[] temp;
    } while (delta > EPSILON);

    t = cpuSecond() - t;

    delete[] a;
    delete[] b;
    delete[] c;
    return t * 1000; // возвращаем значение в мс
}

double run_parallel_var2(int cols, int rows) {
    // создается одна параллельная секция #pragma omp
    // parallel, охватывающая весь итерационный алгоритм.
    double *a = new double[cols * rows];
    double *b = new double[cols];
    double *c = new double[cols];
    prepare_values(a, b, c, cols, rows);

    double t = cpuSecond();

    double delta = 0.0;
    do {
        #pragma omp parallel
        {
            double *temp = new double[cols];
            double localdelta = 0.0;

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
            
            #pragma omp atomic // предотвращаем одновременные обращения потоков
                delta += localdelta;
            delete[] temp;
        }
    } while (delta > EPSILON);

    t = cpuSecond() - t;

    delete[] a;
    delete[] b;
    delete[] c;
    return t * 1000; // возвращаем значение в мс
}

int main() {
    int threads[] = { 1, 2, 4, 7, 8, 16, 20, 40, 60, 80 };
    int SIZE = 5000;

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
            printf("\nVar1 is faster on %.6f ms\n", parallel_var1_results - parallel_var2_results);
        }
        else {
            printf("\nVar2 is faster on %.6f ms\n", parallel_var2_results - parallel_var1_results);
        }
        printf("------\n");
    }

    return 0;
}
