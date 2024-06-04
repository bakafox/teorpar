#include <iostream>
#include <cmath>
#include <cstring>
#include <memory>
#include <chrono>
#include <fstream>

#include <boost/program_options.hpp>

using namespace std;
namespace opt = boost::program_options;



void initialize(double *A, double *Anew, int m, int n) {
    memset(A, 0, n*m*sizeof(double));
    memset(Anew, 0, n*m*sizeof(double));

    A[0] = 10.0;             // 10 --- 20
    A[n-1] = 20.0;           //  |     |
    A[m*(n-1)] = 20.0;       //  |     |
    A[m*(n-1)+(n-1)] = 30.0; // 20 --- 30

    for(int i = 1; i < n-1; i++) {
        A[i] = 10.0 + (20.0 - 10.0)/(n-1)*i;
        A[m*(n-1)+i] = 20.0 + (30.0 - 20.0)/(n-1)*i;
    }

    for (int j = 1; j < m-1; j++) {
        A[j*n] = 10.0 + (20.0 - 10.0)/(m-1)*j;
        A[(j*n)+n-1] = 20.0 + (30.0 - 20.0)/(m-1)*j;
    }

    for (int k = 0; k < m*n; k++) {
        Anew[k] = A[k];
    }
}



auto iterate(double* A, double* Anew, int m, int n, int iterMax, double epsilonMin, double thau) {
    auto start = std::chrono::steady_clock::now();
    
    // Вычисляем матрицу, пока не дойдём до приемлимого epsilon
    double epsilon = 1.0;
    for (int iter = 0; iter < iterMax; iter++) {
        epsilon = 0.0;

        for (int i = 1; i < m-1; i++) {
            for (int j = 1; j < n - 1; j++) {
                Anew[i*n+j] = thau * (A[i*n + j] + A[i*n + (j-1)] + A[i*n + (j+1)] + A[(i-1)*n + j] + A[(i+1)*n + j]);

                epsilon = max(epsilon, fabs(Anew[i*n+j] - A[i*n+j]));
            }
        }

        double* temp = A;
        A = Anew;
        Anew = temp;

        if (epsilon < epsilonMin) {
            cout << "\nDone in " << iter << " iterations!\n";
            break;
        }

        if (iter == iterMax - 1) {
            cout << "\nIterations limit exceeded!\n";
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto timediff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Записываем матрицу в файл
    ofstream resultsFile("output_" + to_string(m) + "x" + to_string(n) + ".txt");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            resultsFile << A[i*n+j] << ' ';
        }
        resultsFile << '\n';
    }
    resultsFile.close();

    return timediff.count();
}



int main(int argc, char *argv[]) {
    // Получаем и парсим опции (по условиям)
    opt::options_description desc('\0');
    desc.add_options()
        ("epsilon", opt::value<double>())
        ("m", opt::value<int>())
        ("n", opt::value<int>())
        ("iter", opt::value<int>())
    ;

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);

    const double epsilonMin = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 1e-6;
    const int m = (vm.count("m")) ? vm["m"].as<int>() : -1;
    const int n = (vm.count("n")) ? vm["n"].as<int>() : -1;
    const int iterMax = (vm.count("iter")) ? vm["iter"].as<int>() : 1e6;

    // Инициализируем и выполняем вычисления
    if (m == -1 || n == -1) {
        const int presets[] = { 10, 13, 128, 256, 512, 1024 };

        cout << "\n=== RUNNING DEFAULT PRESETS... ===\n";
        for (int i = 0; i < 6; i++) {
            double *A = new double[presets[i]*presets[i]];
            double *Anew = new double[presets[i]*presets[i]];
            initialize(A, Anew, presets[i], presets[i]);

            cout << "Matrix size: " << presets[i] << " x " << presets[i] << "\n";
            cout << "Iterations: " << iterMax << ", Epsilon: " << epsilonMin << "\n";

            auto results = iterate(A, Anew, presets[i], presets[i], iterMax, epsilonMin, 0.25);
            cout << "Elapsed time: " << results << " ms\n";
            cout << "------\n\n";
        }
    }
    else {
        double *A = new double[n*m];
        double *Anew = new double[n*m];
        initialize(A, Anew, m, n);

        cout << "\n=== RUNNING USER PRESET... ===\n";
        cout << "Matrix size: " << m << " x " << n << "\n";
        cout << "Iterations: " << iterMax << ", Epsilon: " << epsilonMin << "\n";

        auto results = iterate(A, Anew, m, n, iterMax, epsilonMin, 0.25);
        cout << "Elapsed time: " << results << " ms\n";
        cout << "------\n\n";
    }

    cout << "All results saved in output.txt files.\n";
    return 0;
}
