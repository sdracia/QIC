! main.f90
! Program for Testing Matrix Operations on Complex Matrices
! Author: Andrea Turci
! Date: 3/11/2024
!
! Overview:
!     This program demonstrates the functionality of the `complex8_matrix` derived type
!     from the `mod_matrix_c8` module. It initializes a complex matrix with random entries,
!     calculates its trace, computes its adjoint (conjugate transpose), and verifies specific
!     mathematical properties of the matrix and its adjoint. Additionally, it saves the matrix
!     data to an output file.
!
! Functional Description:
!     The program prompts the user to specify:
!         1. Matrix size (`size`): Sets the dimension of the square matrix.
!         2. Seed for random number generator (`seed`): Ensures reproducibility of random entries.
!
!     With these inputs, the program performs the following operations:
!         1. Initializes a square matrix `A` of complex(8) values with randomly generated real and imaginary parts.
!         2. Computes the trace of the matrix `A`.
!         3. Calculates the adjoint (conjugate transpose) of `A`, verifies that the adjoint of the adjoint
!            matrix matches `A`, and computes the trace of the adjoint.
!         4. Verifies that the trace of `A`'s adjoint matches the conjugate of `A`'s trace.
!         5. Writes matrix data and results to an output text file for analysis.
!
! Compilation:
!     Compile each module and main program separately to support modular development and debugging:
!         gfortran -c debugger.f90
!         gfortran -c mod_matrix_c8.f90
!     Then compile the main program by linking all object files:
!         gfortran -o main debugger.o mod_matrix_c8.o main.f90
!
! Execution:
!     Run the program executable:
!         ./main
!
! Inputs:
!     - Matrix size: Integer input defining the matrix dimensions.
!     - Random seed: Integer input for random number generation to ensure reproducibility.
!
! Outputs:
!     - Text file named based on matrix size and seed, e.g., `matrix_result_<size>x<size>_seed_<seed>.dat`.
!     - Contents of the output file:
!         - Matrix dimensions, trace, elements of `A`, elements of `A`'s adjoint, and their respective traces.
!
! Notes:
!     - Random values for matrix entries provide a means of testing generalized matrix operations.
!     - The program verifies mathematical identities for trace and adjoint, ensuring accuracy.
!     - The output file facilitates data storage for external review and analysis.
!
! Example Run:
!     An example execution may prompt:
!         Enter size of the matrix (default 3):
!         Enter seed (default 12345):
!     Upon completion, the output file might include:
!         Matrices Size: 3 x 3
!         ORIGINAL MATRIX:
!         Trace: (1.2345, 0.6789)
!         Elements:
!         (1.0000,0.5000) (0.1000,0.2000) (0.3000,0.4000)
!         ...
!
! Preconditions:
!     - Ensure valid, positive integer input for matrix size and seed.
!     - The dimensions provided must be suitable for the system's memory limits.
!
! Postconditions:
!     - Matrix operations and identities are verified and saved in the output file.
!     - The generated file provides comprehensive data on matrix operations for further analysis.

program main
    use mod_matrix_c8
    use debugger
    implicit none
    ! This program tests the functionality of the complex8_matrix derived type
    ! defined in the mod_matrix_c8 module. It initializes matrices, computes
    ! their trace and adjoint, and writes the adjoint matrix to a file.

    type(complex8_matrix) :: A      ! Declare matrices A, B, and C
    complex(8) :: trace_A         ! Variable to store the trace
    integer :: rows, cols                  ! Matrix dimensions
    logical :: debug
    type(complex8_matrix) :: A_adjoint
    type(complex8_matrix) :: A_adjoint_adjoint
    complex(8) :: trace_A_conjugate
    complex(8) :: trace_A_adjoint
    logical :: areEqual
    integer :: size
    integer :: i,j
    complex(8) :: out_tr                            ! Output trace
    integer :: ii                                ! Loop variable
    integer :: io_status, seed, m
    integer, allocatable :: seed_array(:)
    character(len=50) :: filename

    debug = .true.

    do
        print*, "Enter size of the matrix (default 3):"
        read(*, *, IOSTAT=io_status) size
        if (io_status == 0 .and. size > 0) exit
        print*, "Invalid input. Please enter a positive integer for size."
        size = 3  ! Default value
    end do

    do
        print*, "Enter seed (default 12345):"
        read(*, *, IOSTAT=io_status) seed
        if (io_status == 0 .and. seed > 0) exit
        print*, "Invalid input. Please enter a positive integer for seed."
        seed = 12345  ! Default value
    end do

    ! SEED MANAGEMENT:

    ! Get the size of the seed array required by random_seed
    call random_seed(size=m)
    allocate(seed_array(m))

    ! Fill the seed array with your seed value
    seed_array = seed

    ! Set the seed for the random number generator
    call random_seed(put=seed_array)

    rows = size
    cols = size

    ! Initialize matrix A
    call initMatrix(A, rows, cols)

    do i = 1, rows
        do j = 1, cols
            ! Generate random real and imaginary parts, cast them to double precision, and assign to A%elem
            call random_number(A%elem(i, j)%re)
            call random_number(A%elem(i, j)%im)
            A%elem(i, j) = dcmplx(A%elem(i, j)%re, A%elem(i, j)%im)  ! Ensure complex(8) type
        end do
    end do

    ! Calculate the trace of matrix A
    trace_A = .Tr. A
    print *, "Trace of matrix A:", trace_A

    ! Test manual calculation of the Trace
    ! Initialize trace to zero
    out_tr = (0.0d0, 0.0d0)

    ! Sum the diagonal elements
    do ii = 1, A%size(1)
        out_tr = out_tr + A%elem(ii, ii)
    end do

    ! Expected trace value for matrix A (manual calculation)
    if (abs(trace_A - out_tr) < 1.0d-5) then
        call checkpoint_character(debug = .true., verbosity = 1, msg = "Trace of A is correct")
    else
        call checkpoint_character(debug = .true., verbosity = 1, msg = "Trace of A is incorrect")
    end if

    ! Verification of (A^H)^H = A
    ! Calculate the conjugate transpose (adjoint) of matrix A
    A_adjoint = .Adj. A
    trace_A_adjoint = .Tr. A_adjoint

    A_adjoint_adjoint = .Adj. A_adjoint

    ! Verify that (A^H)^H = A
    call matrices_are_equal(A, A_adjoint_adjoint, areEqual)

    if (areEqual) then
        call checkpoint_character(debug = .true., verbosity = 1, msg = "Verification: (A^H)^H = A is correct")
    else
        call checkpoint_character(debug = .true., verbosity = 1, msg = "Verification: (A^H)^H = A is incorrect")
    end if

    ! Verification of tr(A^H) = conjugate of tr(A)
    ! Calculate the trace of the adjoint of A
    trace_A_conjugate = .Tr. A_adjoint

    ! Verify that tr(A^H) = \bar{tr(A)}
    if (abs(trace_A_conjugate - conjg(trace_A)) < 1.0d-5) then
        call checkpoint_character(debug = .true., verbosity = 1, msg = "Verification: tr(A^H) = conjugate of tr(A) is correct")
    else
        call checkpoint_character(debug = .true., verbosity = 1, msg = "Verification: tr(A^H) = conjugate of tr(A) is incorrect")
    end if

    ! Writing everything to the text file
    call CMatDumpTXT(A, A_adjoint, trace_A, trace_A_adjoint, seed, filename)
    print *, "The original matrix has been written to file: ", filename

end program main
