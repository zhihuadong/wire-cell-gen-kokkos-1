#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>

#include <Kokkos_Random.hpp>
#include <Kokkos_DualView.hpp>
#include <impl/Kokkos_Timer.hpp>



// A Functor for generating uint64_t random numbers templated on the
// GeneratorPool type
template <class GeneratorPool>
struct generate_random {
  // Output View for the random numbers
  Kokkos::View<uint64_t*> vals;

  // The GeneratorPool
  GeneratorPool rand_pool;

  int samples;

  // Initialize all members
  generate_random(Kokkos::View<uint64_t*> vals_, GeneratorPool rand_pool_, int samples_)
      : vals(vals_), rand_pool(rand_pool_), samples(samples_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const {
    // Get a random number state from the pool for the active thread
    typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

    // Draw samples numbers from the pool as urand64 between 0 and
    // rand_pool.MAX_URAND64 Note there are function calls to get other type of
    // scalars, and also to specify Ranges or get a normal distributed float.
    for (int k = 0; k < samples; k++)
      vals(i * samples + k) = rand_gen.urand64();

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rand_gen);
  }
};





void checkSizes( int &N, int &M, int &S, int &nrepeat );
void randomNumberTest();

int main( int argc, char* argv[] )
{
  int N = -1;         // number of rows 2^12
  int M = -1;         // number of columns 2^10
  int S = -1;         // total size 2^22
  int nrepeat = 100;  // number of repeats of the test

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Rows" ) == 0 ) ) {
      N = pow( 2, atoi( argv[ ++i ] ) );
      printf( "  User N is %d\n", N );
    }
    else if ( ( strcmp( argv[ i ], "-M" ) == 0 ) || ( strcmp( argv[ i ], "-Columns" ) == 0 ) ) {
      M = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User M is %d\n", M );
    }
    else if ( ( strcmp( argv[ i ], "-S" ) == 0 ) || ( strcmp( argv[ i ], "-Size" ) == 0 ) ) {
      S = pow( 2, atof( argv[ ++i ] ) );
      printf( "  User S is %d\n", S );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  y^T*A*x Options:\n" );
      printf( "  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n" );
      printf( "  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n" );
      printf( "  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }

  // Check sizes.
  checkSizes( N, M, S, nrepeat );

  Kokkos::initialize( argc, argv );
  {

  // Allocate y, x vectors and Matrix A:
  double * const y = new double[ N ];
  double * const x = new double[ M ];
  double * const A = new double[ N * M ];

  // Initialize y vector.
  Kokkos::parallel_for( "y_init", N, KOKKOS_LAMBDA ( int i ) {
    y[ i ] = 1;
  });

  // Initialize x vector.
  Kokkos::parallel_for( "x_init", M, KOKKOS_LAMBDA ( int i ) {
    x[ i ] = 1;
  });

  // Initialize A matrix, note 2D indexing computation.
  Kokkos::parallel_for( "matrix_init", N, KOKKOS_LAMBDA ( int j ) {
    for ( int i = 0; i < M; ++i ) {
      A[ j * M + i ] = 1;
    }
  });

  // Timer products.
  Kokkos::Timer timer;

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    // Application: <y,Ax> = y^T*A*x
    double result = 0;

    Kokkos::parallel_reduce( "yAx", N, KOKKOS_LAMBDA ( int j, double &update ) {
      double temp2 = 0;

      for ( int i = 0; i < M; ++i ) {
        temp2 += A[ j * M + i ] * x[ i ];
      }

      update += y[ j ] * temp2;
    }, Kokkos::Sum<double>(result));

    // Output result.
    if ( repeat == ( nrepeat - 1 ) ) {
      printf( "  Computed result for %d x %d is %lf\n", N, M, result );
    }

    const double solution = (double) N * (double) M;

    if ( result != solution ) {
      printf( "  Error: result( %lf ) != solution( %lf )\n", result, solution );
    }
  }

  double time = timer.seconds();

  // Calculate bandwidth.
  // Each matrix A row (each of length M) is read once.
  // The x vector (of length M) is read N times.
  // The y vector (of length N) is read once.
  // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
  double Gbytes = 1.0e-9 * double( sizeof(double) * ( M + M * N + N ) );

  // Print results (problem size, time and bandwidth in GB/s).
  printf( "  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
          N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

  delete[] A;
  delete[] y;
  delete[] x;


  randomNumberTest();

  }
  Kokkos::finalize();

  return 0;
}

void checkSizes( int &N, int &M, int &S, int &nrepeat ) {
  // If S is undefined and N or M is undefined, set S to 2^22 or the bigger of N and M.
  if ( S == -1 && ( N == -1 || M == -1 ) ) {
    S = pow( 2, 22 );
    if ( S < N ) S = N;
    if ( S < M ) S = M;
  }

  // If S is undefined and both N and M are defined, set S = N * M.
  if ( S == -1 ) S = N * M;

  // If both N and M are undefined, fix row length to the smaller of S and 2^10 = 1024.
  if ( N == -1 && M == -1 ) {
    if ( S > 1024 ) {
      M = 1024;
    }
    else {
      M = S;
    }
  }

  // If only M is undefined, set it.
  if ( M == -1 ) M = S / N;

  // If N is undefined, set it.
  if ( N == -1 ) N = S / M;

  printf( "  Total size S = %d N = %d M = %d\n", S, N, M );

  // Check sizes.
  if ( ( S < 0 ) || ( N < 0 ) || ( M < 0 ) || ( nrepeat < 0 ) ) {
    printf( "  Sizes must be greater than 0.\n" );
    exit( 1 );
  }

  if ( ( N * M ) != S ) {
    printf( "  N * M != S\n" );
    exit( 1 );
  }
}


void randomNumberTest() {
    int size    = 1024;
    int samples = 1024;
    int seed = 5374857;

    // Create two random number generator pools one for 64bit states and one for 1024 bit states Both take an 64 bit unsigned integer seed to initialize a Random_XorShift64 generator which is used to fill the generators of the pool.
    Kokkos::Random_XorShift64_Pool<> rand_pool64(seed);
    Kokkos::Random_XorShift1024_Pool<> rand_pool1024(seed);
    Kokkos::DualView<uint64_t*> vals("Vals", size * samples);

    // Run some performance comparisons
    Kokkos::Timer timer;
    Kokkos::parallel_for(size,
                         generate_random<Kokkos::Random_XorShift64_Pool<> >(
                             vals.d_view, rand_pool64, samples));
    Kokkos::fence();

    timer.reset();
    Kokkos::parallel_for(size,
                         generate_random<Kokkos::Random_XorShift64_Pool<> >(
                             vals.d_view, rand_pool64, samples));
    Kokkos::fence();
    double time_64 = timer.seconds();

    Kokkos::parallel_for(size,
                         generate_random<Kokkos::Random_XorShift1024_Pool<> >(
                             vals.d_view, rand_pool1024, samples));
    Kokkos::fence();

    timer.reset();
    Kokkos::parallel_for(size,
                         generate_random<Kokkos::Random_XorShift1024_Pool<> >(
                             vals.d_view, rand_pool1024, samples));
    Kokkos::fence();
    double time_1024 = timer.seconds();

    printf("\n\n#Time XorShift64*:   %e %e\n", time_64, 1.0e-9 * samples * size / time_64);
    printf("#Time XorShift1024*: %e %e\n", time_1024, 1.0e-9 * samples * size / time_1024);

    Kokkos::deep_copy(vals.h_view, vals.d_view);







}









