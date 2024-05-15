#include <iostream>
#include <math.h>
#include <time.h>
#include <thread> // РЕАЛИЗАЦИЯ ЧЕРЕЗ std::thread
#include <vector>

using namespace std;



double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}



void prepare_matrix(double *a, int start, int end, int cols, int rows) {
    for (int i = start; i < end; i++) {
        for (int j = 0; j < rows; j++) {
            a[i * rows + j] = i + j;
        }
    }
}

void prepare_vector(double *b, int start, int end, int cols, int rows) {
    for (int j = start; j < end; j++) {
        b[j] = j;
    }
}

void matrix_vector_product(double *a, double *b, double *c, int start, int end, int cols, int rows) {
    for (int i = start; i < end; i++) {
        c[i] = 0.0;

        for (int j = 0; j < rows; j++) {
            c[i] += a[i * rows + j] * b[j];
        }
    }
}



double run_serial(int cols, int rows) {
    double* a = new double[cols * rows];
    double* b = new double[rows];
    double* c = new double[cols];

    prepare_matrix(a, 0, cols, cols, rows);
    prepare_vector(b, 0, rows, cols, rows);

    double t = cpuSecond();
    matrix_vector_product(a, b, c, 0, cols, cols, rows);
    t = cpuSecond() - t;

    delete[] a;
    delete[] b;
    delete[] c;
    return t * 1000; // возвращаем значение в мс
}



double run_parallel(int cols, int rows, int num_threads) {
    double* a = new double[cols * rows];
    double* b = new double[rows];
    double* c = new double[cols];

    vector<thread> threads;
    int items_per_thread = cols / num_threads;

    for (int thread = 0; thread < num_threads; ++thread) {
        int lb = thread * items_per_thread;
        int ub = (thread == num_threads - 1) ? cols : (thread + 1) * items_per_thread;

        threads.emplace_back(prepare_matrix, a, lb, ub, cols, rows);
        threads.emplace_back(prepare_vector, b, lb, ub, cols, rows);
    }

    for (auto& thread : threads) {
        thread.join(); // объединяем все процессы, ждём завершения вычислений
    }

    // удаляем старые отработавшие подпроцессы (иначе будет ошибка)
    threads.clear();

    double t = cpuSecond();

    for (int thread = 0; thread < num_threads; ++thread) {
        int lb = thread * items_per_thread;
        int ub = (thread == num_threads - 1) ? rows : (thread + 1) * items_per_thread;

        threads.emplace_back(matrix_vector_product, a, b, c, lb, ub, cols, rows);
    }

    for (auto& thread : threads) {
        thread.join(); // объединяем все процессы, ждём завершения вычислений
    }

    t = cpuSecond() - t;

    delete[] a;
    delete[] b;
    delete[] c;
    return t * 1000; // возвращаем значение в мс
}



int main() {
    int threads[] = { 1, 2, 4, 7, 8, 16, 20, 40, 60, 80 };

    // намеренно всё запускаем последовательно - это не ошибка!
    printf("\n=== SERIAL ===\n");
    double serial_results[2];
    serial_results[0] = run_serial(20000, 20000);
    printf("20K elapsed time: %.6f ms\n", serial_results[0]);
    serial_results[1] = run_serial(40000, 40000);
    printf("40K elapsed time: %.6f ms\n", serial_results[1]);

    printf("\n=== PARALLEL ===\n");
    for (int i = 0; i < 10; i++) {
        printf("Number of threads: %d\n", threads[i]);
        double parallel_results[2];
        parallel_results[0] = run_parallel(20000, 20000, threads[i]);
        printf("20K elapsed time: %.6f ms\n", parallel_results[0]);
        printf("20K accelerarion ratio: %.6f\n", serial_results[0] / parallel_results[0]);
        parallel_results[1] = run_parallel(40000, 40000, threads[i]);
        printf("40K elapsed time: %.6f ms\n", parallel_results[1]);
        printf("40K accelerarion ratio: %.6f\n", serial_results[1] / parallel_results[1]);
        printf("------\n");
    }

    return 0;
}
