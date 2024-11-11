module matrix_mult
    use debugger

    ! This module contains subroutines for matrix multiplication using different methods
    ! and also for performance measurement. It includes:
    ! 1. perform_multiplications: Measures the performance of different matrix multiplication methods.
    ! 2. prepare_output_file: Prepares the output filename for saving results.
    ! 3. matrix_multiply_explicit: Explicit row-by-column matrix multiplication.
    ! 4. matrix_multiply_column: Column-by-row matrix multiplication.
    implicit none
    
contains 

!=============================================================
! SUBROUTINE: perform_multiplications
! 
! This subroutine performs matrix multiplications based on the specified parameters and measures 
! the time taken for each method (explicit row-by-column, column-by-row, and Fortran's intrinsic MATMUL).
!
! INPUTS:
!     max_size    (integer)  Maximum size of the matrices to test.
!     step        (integer)  Step size for matrix size increments.
!     seed        (integer)  Seed for random number generation for reproducibility.
!     opt_flag    (character) Optimization flag for compiler settings.
!     type_mult   (character) Type of multiplication method to evaluate ("row-col", "col-row", "matmul", "ALL").
!
! OUTPUTS:
!     Writes the computation times for each multiplication method to an output file based on the specified type.
!
! POSTCONDITIONS:
!     - The computation times for each multiplication method are recorded in the output file.
!     - Matrices A, B, and result matrices (C_explicit, C_intrinsic) are deallocated after use.
!=============================================================
subroutine perform_multiplications(min_size, max_size, step, seed, opt_flag, type_mult)
    use debugger
    implicit none

    ! Inputs
    integer, intent(in) :: max_size, step, seed, min_size
    character(len=10), intent(in) :: opt_flag
    character(len=20), intent(in) :: type_mult

    ! Local variables
    real(8), allocatable :: A(:,:), B(:,:), C_explicit(:,:), C_intrinsic(:,:)
    real(8) :: start_time, end_time, time_explicit, time_column, time_matmul
    character(len=50) :: filename  ! Output filename for performance results
    logical :: flag
    integer :: i  ! Loop variable for matrix sizes
    integer, allocatable :: seed_array(:)
    integer :: m

    ! Preconditions: Validate inputs
    call checkpoint_real(debug=.TRUE., msg='Beginning matrix multiplication process.')
    if (max_size <= 0 .or. step <= 0 .or. step >= max_size) then
        print*, "Error: Invalid matrix size or step configuration."
        return
    end if

    ! Prepare the output file for storing results
    call prepare_output_file(filename, type_mult, min_size, max_size, opt_flag, step, seed)

    ! Seed management for random number generation
    call random_seed(size=m)
    allocate(seed_array(m))
    seed_array = seed
    call random_seed(put=seed_array)

    ! Loop over matrix sizes from min_size to max_size with the specified step
    do i = min_size, max_size, step
        print*, "------------------------"
        print*, "Matrix size:", i
        allocate(A(i,i), B(i,i), C_explicit(i,i), C_intrinsic(i,i))

        ! Initialize matrices A and B with random values
        call random_number(A)
        call random_number(B)

        ! Perform the selected multiplication method(s)
        if (type_mult == "ALL" .or. type_mult == "row-col") then
            ! Row-by-column method
            call cpu_time(start_time)
            C_explicit = 0.0_8
            call matrix_multiply_explicit(A, B, C_explicit, i)
            call cpu_time(end_time)
            time_explicit = end_time - start_time
            call checkpoint_real(debug = .TRUE., verbosity= 2, msg = 'Time taken for row-col method', var1 = time_explicit)
        end if 

        if (type_mult == "ALL" .or. type_mult == "col-row") then
            ! Column-by-row method
            call cpu_time(start_time)
            C_explicit = 0.0_8
            call matrix_multiply_column(A, B, C_explicit, i)
            call cpu_time(end_time)
            time_column = end_time - start_time
            call checkpoint_real(debug = .TRUE., verbosity = 2, msg = 'Time taken for col-row method', var1 = time_column)
        end if 

        if (type_mult == "ALL" .or. type_mult == "matmul") then
            ! MATMUL intrinsic method
            call cpu_time(start_time)
            C_intrinsic = matmul(A, B)
            call cpu_time(end_time)
            time_matmul = end_time - start_time
            call checkpoint_real(debug = .TRUE., verbosity = 2, msg = 'Time taken for intrinsic MATMUL', var1 = time_matmul)
        end if

        ! Write the times to the output file based on selected method
        if (type_mult == "ALL") then
            write(20, '(I6, 3X, F12.6, 3X, F12.6, 3X, F12.6)') i, time_explicit, time_column, time_matmul
        else if (type_mult == "row-col") then
            write(20, '(I6, 3X, F12.6)') i, time_explicit
        else if (type_mult == "col-row") then
            write(20, '(I6, 3X, F12.6)') i, time_column
        else if (type_mult == "matmul") then
            write(20, '(I6, 3X, F12.6)') i, time_matmul
        end if

        ! Deallocate matrices A, B, and results C after use
        deallocate(A, B, C_explicit, C_intrinsic)
    end do

    ! Close the output file
    close(20) 
end subroutine perform_multiplications


!=============================================================
! SUBROUTINE: prepare_output_file
!
! Prepares the output filename based on user parameters, and checks if the file exists. 
! If the file does not exist, it creates a new file and writes a header.
!
! INPUTS:
!     filename   (character) Output filename for storing results.
!     type_mult  (character) Type of multiplication ("row-col", "col-row", "matmul", "ALL").
!     max_size   (integer)   Maximum size of matrices for the test.
!     opt_flag   (character) Compiler optimization flag.
!     step       (integer)   Step size for matrix size increments.
!     seed       (integer)   Random seed used for reproducibility.
!
! POSTCONDITIONS:
!     - The filename string is updated with the appropriate filename.
!     - If the file doesn't exist, it is created with a header for the data.
!=============================================================
subroutine prepare_output_file(filename, type_mult, min_size, max_size, opt_flag, step, seed)
    implicit none
    character(len=50), intent(out) :: filename
    character(len=20), intent(in) :: type_mult
    integer, intent(in) :: max_size, step, min_size, seed
    character(len=6) :: min_size_str, max_size_str, step_str, seed_str
    character(len=6), intent(in) :: opt_flag
    logical :: flag

    ! Convert integers to strings
    write(min_size_str, '(I0)') min_size
    write(max_size_str, '(I0)') max_size
    write(step_str, '(I0)') step
    write(seed_str, '(I0)') seed

    ! Create the filename
    write(filename, '(A, A, A, A, A, A, A, A)') "data/" // trim(type_mult) // "_size_", &
        trim(min_size_str), "-" // trim(max_size_str), "_step_", &
        trim(step_str), "_flag_" // trim(opt_flag) // ".dat"

    ! Check if the file exists, create it if not and write the header
    inquire(file=filename, exist=flag)
    if (.not. flag) then
        open(unit=20, file=filename, status="replace", action="write")
        if (type_mult == "ALL") then
            write(20, '(A)') '    Size    Explicit(i-j-k)    Column-major(i-k-j)    MATMUL'
        else if (type_mult == "row-col") then
            write(20, '(A)') '    Size    Explicit(i-j-k)'
        else if (type_mult == "col-row") then
            write(20, '(A)') '    Size    Column-major(i-k-j)'
        else if (type_mult == "matmul") then
            write(20, '(A)') '    Size    MATMUL'
        end if
    end if
end subroutine prepare_output_file


!=============================================================
! SUBROUTINE: matrix_multiply_explicit
! 
! Performs matrix multiplication using an explicit row-by-column (i-j-k) approach.
!
! INPUTS:
!     A       - First input matrix.
!     B       - Second input matrix.
!     C       - Result matrix to store the multiplication.
!     n       - Size of the matrices (assumed square).
!
! PRECONDITIONS:
!     - Matrices A, B, and C are allocated, and n is a valid matrix size.
!
! POSTCONDITIONS:
!     - Matrix C holds the result of A * B.
!=============================================================
subroutine matrix_multiply_explicit(A, B, C, n)
    use debugger
    implicit none

    ! Inputs
    integer, intent(in) :: n  ! Dimension of the matrices
    real(8), intent(in) :: A(n,n), B(n,n)
    real(8), intent(out) :: C(n,n)
    integer :: i, j, k

    ! Preconditions check
    if (size(A,1) /= n .or. size(A,2) /= n .or. size(B,1) /= n .or. size(B,2) /= n .or. size(C,1) /= n .or. size(C,2) /= n) then
        print*, "Error: Invalid matrix dimensions for explicit multiplication."
        return
    end if

    ! Begin multiplication
    call checkpoint_integer(debug = .TRUE., verbosity = 3, msg = 'Starting explicit multiplication', var1 = n)

    ! Perform the multiplication in row-by-column order
    do i = 1, n
        do j = 1, n
            C(i,j) = 0.0_8
            do k = 1, n
                C(i,j) = C(i,j) + A(i,k) * B(k,j)
            end do
        end do
    end do

    ! End multiplication
    call checkpoint_integer(debug = .TRUE., verbosity = 3, msg = 'Finished explicit multiplication, with ', var1 = n)
end subroutine matrix_multiply_explicit



!=============================================================
! SUBROUTINE: matrix_multiply_column
!
! Performs matrix multiplication using a column-by-row (i-k-j) approach.
!
! INPUTS:
!     A       - First input matrix.
!     B       - Second input matrix.
!     C       - Result matrix to store the multiplication.
!     n       - Size of the matrices (assumed square).
!
! PRECONDITIONS:
!     - Matrices A, B, and C are allocated, and n is a valid matrix size.
!
! POSTCONDITIONS:
!     - Matrix C holds the result of A * B.
!=============================================================
subroutine matrix_multiply_column(A, B, C, n)
    use debugger
    implicit none

    ! Inputs
    integer, intent(in) :: n  ! Dimension of the matrices
    real(8), intent(in) :: A(n,n), B(n,n)
    real(8), intent(out) :: C(n,n)
    integer :: i, j, k

    ! Preconditions check
    if (size(A,1) /= n .or. size(A,2) /= n .or. size(B,1) /= n .or. size(B,2) /= n .or. size(C,1) /= n .or. size(C,2) /= n) then
        print*, "Error: Invalid matrix dimensions for column multiplication."
        return
    end if

    ! Begin multiplication
    call checkpoint_integer(debug = .TRUE., verbosity = 3, msg = 'Starting column multiplication', var1 = n)

    ! Perform the multiplication in column-by-row order
    do i = 1, n
        do k = 1, n
            do j = 1, n
                C(i,j) = C(i,j) + A(i,k) * B(k,j)
            end do
        end do
    end do

    ! End multiplication
    call checkpoint_integer(debug = .TRUE., verbosity = 3, msg = 'Finished column multiplication, with', var1 = n)
end subroutine matrix_multiply_column


end module matrix_mult