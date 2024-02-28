#include <stdio.h>
#include <omp.h>

int main(int argc, char **argv) {
    #pragma omp parallel num_threads(6) // число потоков
    { // FORK – порождение нового потока
        printf("[i] Hello, multithreaded world: thread %d of %d\n",
            omp_get_thread_num()+1, omp_get_num_threads());
    } // JOIN - ожидание завершения потока (объединение потоков управления)

    return 0;
}
