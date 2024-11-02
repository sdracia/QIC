! matrix_mult.f90
! Program for Measuring the Performance of Matrix Multiplication Methods
! Author: Andrea Turci
! Date: 2/11/2024
!
! Overview:
!     This program measures and compares the performance of various matrix multiplication methods,
!     including explicit (row-by-column), explicit (column-by-row), and Fortran's intrinsic MATMUL function.
!     It allows users to specify a range of parameters to customize the test, making it possible to
!     experiment with different matrix sizes, random seeds, optimization flags, and multiplication types.
!
! Functional Description:
!     The program prompts the user to enter the following parameters:
!         1. Maximum matrix size (`max_size`): Sets the largest dimension for the test matrices.
!         2. Step size (`step`): Controls the increment size for the matrix dimension in each test.
!         3. Seed for random number generator (`seed`): Ensures reproducibility of random matrix entries.
!         4. Optimization flag (`opt_flag`): Specifies the compiler optimization level to be used.
!         5. Type of multiplication (`type_mult`): Chooses which multiplication methods are evaluated.
! 
!     Based on these inputs, the program generates matrices of complex numbers with random entries
!     and performs the selected matrix multiplication methods. The time taken for each multiplication
!     is recorded and saved to an output file named according to the parameters.
!
! Compilation:
!     Compile each module individually to enable separate testing and debugging:
!         gfortran -c debugger.f90
!         gfortran -c matrix_mult.f90
!     Then compile the final program by linking the object files:
!         gfortran -o mat1 matrix_mult.o debugger.o
!
! Execution:
!     Run the program executable:
!         ./mat1
!
! Inputs:
!     - Prompts for matrix parameters to guide the execution.
!     - Default values are available to allow convenient testing without all input fields.
!
! Outputs:
!     - A performance result file named based on parameters: <type_mult>_size_<max_size>_<opt_flag>_step_<step>.dat
!     - Contents of the output file:
!         - Each line corresponds to a matrix size, with timing values (in seconds) for each evaluated multiplication method.
!
! Notes:
!     - This program is intended for benchmarking and may consume significant memory and time,
!       especially with large matrices or high step values.
!
! Example Run:
!     The following run of the program:
!         ./mat1
!     may output a result file containing:
!         Explicit(i-j-k)    Column-major(i-k-j)    MATMUL
!         0.000002          0.000001               0.000002
!         0.000021          0.000012               0.000021
!
! Preconditions:
!     - Ensure valid input is provided for each parameter.
!     - Ensure that matrices are square, and dimensions match throughout calculations.
!
! Postconditions:
!     - Matrix multiplication times for each method are saved in the specified file.
!


program matrix_multiplication_performance
    use debugger
    use matmul_timing
    implicit none

    ! Parameters for matrix multiplication configuration
    integer :: max_size, step, seed, min_size
    character(len=20) :: type_mult
    character(len=10) :: opt_flag
    !integer :: io_status

    read(*, *) min_size
    read(*, *) max_size
    read(*, *) step
    read(*, *) seed
    read(*, '(A)') opt_flag
    read(*, '(A)') type_mult

    call perform_multiplications(min_size, max_size, step, seed, opt_flag, type_mult)

end program matrix_multiplication_performance