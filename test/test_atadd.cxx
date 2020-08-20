#include <Kokkos_Core.hpp>

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>

using Scalar = float;
using Layout = Kokkos::LayoutLeft;
using matrix_type = typename Kokkos::View<Scalar**, Layout>;

matrix_type gen_2D_view(const int N0, const int N1, const Scalar val = 0)
{
    matrix_type ret("ret", N0, N1);
    Kokkos::parallel_for("setRet", Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
                         KOKKOS_LAMBDA(const int& i0, const int& i1) { ret(i0, i1) = val; });
    return ret;
}

int scatter_add(matrix_type grid, const matrix_type patch, const int x, const int y)
{
    // TODO size checks

    size_t N0 = patch.extent(0);
    size_t N1 = patch.extent(1);

    Kokkos::parallel_for(
        "scatter_add", Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>({0, 0}, {N0, N1}),
        KOKKOS_LAMBDA(const int& i, const int& j) { Kokkos::atomic_add(&grid(x + i, y + j), patch(i, j)); });

    // for(size_t i=0; i<N0; ++i) {
    //     for(size_t j=0; j<N1; ++j) {
    //         grid(x + i, y + j) += patch(i, j);
    //     }
    // }

    return 0;
}

std::string dump(const matrix_type& A, const size_t length_limit = 20)
{
    std::stringstream ss;

    size_t N0 = A.extent(0);
    size_t N1 = A.extent(1);
    bool print_dot0 = true;
    for (size_t i = 0; i < N0; ++i) {
        if (i > length_limit && i < N0 - length_limit) {
            if (print_dot0) {
                ss << "... \n";
                print_dot0 = false;
            }
            continue;
        }

        bool print_dot1 = true;
        for (size_t j = 0; j < N1; ++j) {
            if (j > length_limit && j < N1 - length_limit) {
                if (print_dot1) {
                    ss << "... ";
                    print_dot1 = false;
                }

                continue;
            }
            ss << A(i, j) << " ";
        }
        ss << std::endl;
    }

    return ss.str();
}

void quick_check(const int N0 = 20, const int N1 = 20, const int M0 = 2, const int M1 = 5, const int Npatch = 4,
                 const bool verbose = true)
{
    matrix_type grid("grid", N0, N1);

    auto A = gen_2D_view(M0, M1, 1);

    if (verbose) {
        std::cout << "input grid: \n" << dump(grid) << std::endl;
        std::cout << "A: \n" << dump(A) << std::endl;
    }

    std::vector<int> vec_x;
    std::vector<int> vec_y;
    // std::srand(std::time(nullptr)); // current time as seed
    std::srand(0);  // using 0 as seed, repeatable
    for (int i = 0; i < Npatch; ++i) {
        vec_x.push_back(1.0 * std::rand() / RAND_MAX * (N0 - M0));
        vec_y.push_back(1.0 * std::rand() / RAND_MAX * (N1 - M1));
    }

    Kokkos::parallel_for("ScAdd",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left>>({0, 0, 0}, {Npatch, M0, M1}),
                         KOKKOS_LAMBDA(const int& p, const int& i, const int& j) {
                             //  auto patch = gen_2D_view(M0, M1, 1);
                             auto x = vec_x[p];
                             auto y = vec_y[p];
                             //  auto x = p * M0 - p;
                             //  auto y = p * M1 - p;
                             Kokkos::atomic_add(&grid(x + i, y + j), A(i, j));
                         });

    if (verbose) {
        std::cout << "output grid: \n" << dump(grid) << std::endl;
        std::ofstream fout("grid.csv");
        fout << dump(grid, INT_MAX);
    }
}

int main(int argc, char* argv[])
{
    int npatch = 10000;
    int nrep = 10;

    if (argc > 1) {
        npatch = atoi(argv[1]);
    }

    if (argc > 2) {
        nrep = atoi(argv[2]);
    }

    std::cout << "npatch: " << npatch << " nrep: " << nrep << std::endl;
    std::vector<int> nthreads = {16};
    Kokkos::initialize(argc, argv);
    {
        Kokkos::Timer timer;
        for (int rep = 0; rep < nrep; ++rep) {
            quick_check(1000, 6000, 15, 30, npatch, nrep<2);
        }
        double time = timer.seconds();
        std::cout << "Kokkos::Timer: " << " " << time << std::endl;
    }
    Kokkos::finalize();
}
