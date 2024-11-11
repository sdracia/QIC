! matrix_mult.f90
! Program for Measuring the Performance of Matrix Multiplication Methods
! Author: Andrea Turci
! Date: 2/11/2024
!
! Overview:
!     This program measures and compares the performance of various matrix multiplication methods,
!     including explicit (row-by-column), explicit (column-by-row), and Fortran's intrinsic MATMUL function.
!     Users specify matrix sizes, random seeds, optimization flags, and multiplication types to test.
!
! Functional Description:
!     Prompts for the following parameters:
!         1. Minimum matrix size (`min_size`): Sets the smallest dimension for the test matrices.
!         2. Maximum matrix size (`max_size`): Sets the largest dimension for the test matrices.
!         3. Step size (`step`): Controls the increment size for the matrix dimension in each test.
!         4. Seed (`seed`): Ensures reproducibility of random matrix entries.
!         5. Optimization flag (`opt_flag`): Specifies the compiler optimization level.
!         6. Multiplication type (`type_mult`): Chooses multiplication methods to evaluate.
!
!     Matrices are generated with random entries, and the selected multiplication methods are performed.
!     Timing data is saved to a file named according to the parameters.
!
!
! Inputs:
!     - Prompts for matrix parameters. Defaults are set for ease of testing.
!
! Outputs:
!     - A results file named as: `<type_mult>_size_<max_size>_<opt_flag>_step_<step>.dat`
!       - Contains timing results (in seconds) for each multiplication method per matrix size.
!
! Notes:
!     - This benchmark program may use substantial memory/time, especially with large matrices.
!
! Preconditions:
!     - Ensure valid input for each parameter.
!     - Matrices must be square, with matching dimensions across calculations.
!
! Postconditions:
!     - Matrix multiplication times are saved in the results file.
!
! Example of file that can be produced:
!         Explicit(i-j-k)    Column-major(i-k-j)    MATMUL
!         0.000002          0.000001               0.000002


program matrix_multiplication_performance
    use debugger          ! Import debugging module for checkpoints
    use matrix_mult       ! Import matrix multiplication module
    implicit none

    ! Parameters for matrix multiplication configuration
    integer :: max_size, step, seed, min_size
    character(len=20) :: type_mult
    character(len=10) :: opt_flag

    ! Read user-defined parameters for matrix multiplication
    ! Minimum size for matrix testing
    read(*, *) min_size
    ! Maximum size for matrix testing
    read(*, *) max_size
    ! Step size for incrementing matrix dimensions
    read(*, *) step
    ! Seed for random number generator for reproducibility
    read(*, *) seed
    ! Compiler optimization flag (O1, O2, O3)
    read(*, '(A)') opt_flag
    ! Type of multiplication to be evaluated (e.g., matmul, row-col, col-row, ALL)
    read(*, '(A)') type_mult

    ! Main subroutine to execute matrix multiplications based on input parameters
    call perform_multiplications(min_size, max_size, step, seed, opt_flag, type_mult)

end program matrix_multiplication_performance
