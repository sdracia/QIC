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
    implicit none

    ! Parameters for matrix multiplication configuration
    integer :: max_size, step, seed
    character(len=20) :: type_mult
    character(len=10) :: opt_flag
    integer :: io_status

    ! Prompt user for maximum matrix size
    do
        print*, "Enter max_size (default 900):"
        read(*, *, IOSTAT=io_status) max_size
        if (io_status == 0 .and. max_size > 0) exit
        print*, "Invalid input. Please enter a positive integer for max_size."
        max_size = 900  ! Default value
    end do

    ! Prompt user for step size
    do
        print*, "Enter step (default 100, must be less than max_size):"
        read(*, *, IOSTAT=io_status) step
        if (io_status == 0 .and. step > 0 .and. step < max_size) exit
        print*, "Invalid input. Please enter a positive integer less than max_size."
    end do

    ! Prompt user for random seed
    do
        print*, "Enter seed (default 12345):"
        read(*, *, IOSTAT=io_status) seed
        if (io_status == 0 .and. seed > 0) exit
        print*, "Invalid input. Please enter a positive integer for seed."
        seed = 12345  ! Default value
    end do

    ! Prompt user for optimization flag
    do
        print*, "Enter optimization flag (O1, O2, O3; default O2):"
        read(*, '(A)', IOSTAT=io_status) opt_flag
        opt_flag = trim(adjustl(opt_flag))
        if (io_status == 0 .and. (opt_flag == "O1" .or. opt_flag == "O2" .or. opt_flag == "O3")) exit
        print*, "Invalid input. Please enter one of O1, O2, O3."
        opt_flag = "O2"  ! Default value
    end do

    ! Prompt user for type of multiplication
    do
        print*, "Enter type of multiplication (matmul, row-col, col-row, ALL; default ALL):"
        read(*, '(A)', IOSTAT=io_status) type_mult
        type_mult = trim(adjustl(type_mult))
        if (io_status == 0 .and. (type_mult == "matmul" .or. type_mult == "row-col" &
            .or. type_mult == "col-row" .or. type_mult == "ALL")) exit
        print*, "Invalid input. Please enter one of matmul, row-col, col-row, ALL."
        type_mult = "ALL"  ! Default value
    end do

    ! Call the main subroutine to perform matrix multiplications
    call perform_multiplications(max_size, step, seed, opt_flag, type_mult)

end program matrix_multiplication_performance


!=============================================================
!  SUBROUTINE: perform_multiplications
!  
!  This subroutine performs the matrix multiplications based on
!  the specified parameters and measures the time taken for each method.
!  
!  INPUTS:
!     max_size    (integer)  Maximum size of the matrices.
!     step        (integer)  Step size for matrix size increments.
!     seed        (integer)  Seed for random number generation.
!     opt_flag    (character) Optimization flag for compiler settings.
!     type_mult   (character) Type of multiplication method to evaluate.
!  
!  OUTPUTS:
!     Writes the computation times to a file based on the specified type of multiplication.
!  
!  POSTCONDITIONS:
!     The resultant times for each multiplication method are recorded in the output file.
!     The matrices A, B, and the result matrix C are deallocated after use.
!
subroutine perform_multiplications(max_size, step, seed, opt_flag, type_mult)
    use debugger
    implicit none
    integer, intent(in) :: max_size, step, seed
    character(len=10), intent(in) :: opt_flag
    character(len=20), intent(in) :: type_mult
    real(8), allocatable :: A(:,:), B(:,:), C_explicit(:,:), C_intrinsic(:,:)
    real(8) :: start_time, end_time, time_explicit, time_column, time_matmul
    character(len=50) :: filename  ! Output filename for performance results
    logical :: flag
    integer :: i, file_unit  ! File unit number for output
    integer, allocatable :: seed_array(:)
    integer :: m

    ! Preconditions
    call checkpoint_real(debug=.TRUE., msg='Beginning matrix multiplication process.')
    if (max_size <= 0 .or. step <= 0 .or. step >= max_size) then
        print*, "Error: Invalid matrix size or step configuration."
        return
    end if

    ! Prepare output file based on user inputs
    call prepare_output_file(filename, type_mult, max_size, opt_flag, step)

    ! SEED MANAGEMENT:

    ! Get the size of the seed array required by random_seed
    call random_seed(size=m)
    allocate(seed_array(m))

    ! Fill the seed array with your seed value
    seed_array = seed

    ! Set the seed for the random number generator
    call random_seed(put=seed_array)

    ! Loop over matrix sizes from step to max_size
    do i = step, max_size, step
        print*, "------------------------"
        print*, "Matrix size:", i
        allocate(A(i,i), B(i,i), C_explicit(i,i), C_intrinsic(i,i))

        ! Initialize matrices A and B with random values
        call random_number(A)
        call random_number(B)

        ! Measure time for the explicit row-by-column method (i-j-k order)
        call cpu_time(start_time)
        C_explicit = 0.0_8
        call matrix_multiply_explicit(A, B, C_explicit, i)
        call cpu_time(end_time)
        time_explicit = end_time - start_time

        call checkpoint_real(debug = .TRUE., verbosity= 2, msg = 'Time taken for explicit method', var1 = time_explicit)

        ! Measure time for the column-by-row approach (i-k-j order)
        call cpu_time(start_time)
        C_explicit = 0.0_8
        call matrix_multiply_column(A, B, C_explicit, i)
        call cpu_time(end_time)
        time_column = end_time - start_time

        call checkpoint_real(debug = .TRUE., verbosity = 2, msg = 'Time taken for column method', var1 = time_column)

        ! Measure time for Fortran's MATMUL intrinsic function
        call cpu_time(start_time)
        C_intrinsic = matmul(A, B)
        call cpu_time(end_time)
        time_matmul = end_time - start_time

        call checkpoint_real(debug = .TRUE., verbosity = 2, msg = 'Time taken for intrinsic MATMUL', var1 = time_matmul)

        ! Record the computation times for each method in the output file
        if (type_mult == "ALL") then
            write(20, '(F12.6, 3X, F12.6, 3X, F12.6)') time_explicit, time_column, time_matmul
        else if (type_mult == "row-col") then
            write(20, '(F12.6)') time_explicit
        else if (type_mult == "col-row") then
            write(20, '(F12.6)') time_column
        else if (type_mult == "matmul") then
            write(20, '(F12.6)') time_matmul
        end if

        ! Deallocate memory for matrices A, B, and results C
        deallocate(A, B, C_explicit, C_intrinsic)
    end do

    close(20)  ! Close the output file
end subroutine perform_multiplications

!=============================================================
! SUBROUTINE: prepare_output_file
! 
! Prepares the output filename based on user parameters.
!
! INPUTS:
!    filename - Output filename for storing results
!    type_mult - Type of multiplication specified by the user
!    max_size  - Maximum matrix size
!    opt_flag  - Optimization flag
!    step      - Step size
!
! POSTCONDITIONS:
!    - The filename string is updated with appropriate information.
!
subroutine prepare_output_file(filename, type_mult, max_size, opt_flag, step)
    implicit none
    character(len=50), intent(out) :: filename
    character(len=20), intent(in) :: type_mult
    integer, intent(in) :: max_size, step
    character(len=6) :: max_size_str, step_str
    character(len=10), intent(in) :: opt_flag
    logical :: flag

    ! Convert integers to strings without padding
    write(max_size_str, '(I0)') max_size
    write(step_str, '(I0)') step

    ! Create the filename without spaces
    write(filename, '(A, A, A, A, A, A)') "data_mult/" // trim(type_mult) // "_size_", &
        trim(max_size_str), "_" // trim(opt_flag) // "_step_", trim(step_str) // ".dat"

    ! Check if file exists, and if not, create it with the header
        inquire(file=filename, exist=flag)
        if (.not. flag) then
            open(unit=20, file=filename, status="replace", action="write")
            if (type_mult == "ALL") then
                write(20, '(A)') 'Explicit(i-j-k)    Column-major(i-k-j)    MATMUL'
            else if (type_mult == "row-col") then
                write(20, '(A)') 'Explicit(i-j-k)'
            else if (type_mult == "col-row") then
                write(20, '(A)') 'Column-major(i-k-j)'
            else if (type_mult == "matmul") then
                write(20, '(A)') 'MATMUL'
            end if
        end if
end subroutine prepare_output_file


!=============================================================
! SUBROUTINE: matrix_multiply_explicit
! 
! Performs matrix multiplication using an explicit row-by-column (i-j-k) approach.
!
! INPUTS:
!    A       - First input matrix
!    B       - Second input matrix
!    C       - Result matrix to store the multiplication
!    n       - Size of the matrices (assumed square)
!
! PRECONDITIONS:
!    - Matrices A, B, and C are allocated and n is a valid matrix size.
!
subroutine matrix_multiply_explicit(A, B, C, n)
    use debugger
    implicit none
    integer, intent(in) :: n  ! Dimension of the matrices
    real(8), intent(in) :: A(n,n), B(n,n)
    real(8), intent(out) :: C(n,n)
    integer :: i, j, k

    ! Preconditions check
    if (size(A,1) /= n .or. size(A,2) /= n .or. size(B,1) /= n .or. size(B,2) /= n .or. size(C,1) /= n .or. size(C,2) /= n) then
        print*, "Error: Invalid matrix dimensions for explicit multiplication."
        return
    end if

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

    call checkpoint_integer(debug = .TRUE., verbosity = 3, msg = 'Finished explicit multiplication, with ', var1 = n)
end subroutine matrix_multiply_explicit


!=============================================================
! SUBROUTINE: matrix_multiply_column
! 
! Performs matrix multiplication using a column-by-row (i-k-j) approach.
!
! INPUTS:
!    A       - First input matrix
!    B       - Second input matrix
!    C       - Result matrix to store the multiplication
!    n       - Size of the matrices (assumed square)
!
! PRECONDITIONS:
!    - Matrices A, B, and C are allocated and n is a valid matrix size.
!
subroutine matrix_multiply_column(A, B, C, n)
    use debugger
    implicit none
    integer, intent(in) :: n  ! Dimension of the matrices
    real(8), intent(in) :: A(n,n), B(n,n)
    real(8), intent(out) :: C(n,n)
    integer :: i, j, k

    ! Preconditions check
    if (size(A,1) /= n .or. size(A,2) /= n .or. size(B,1) /= n .or. size(B,2) /= n .or. size(C,1) /= n .or. size(C,2) /= n) then
        print*, "Error: Invalid matrix dimensions for column multiplication."
        return
    end if

    call checkpoint_integer(debug = .TRUE., verbosity = 3, msg = 'Starting column multiplication', var1 = n)

    ! Perform the multiplication in column-by-row order
    do i = 1, n
        do k = 1, n
            do j = 1, n
                C(i,j) = C(i,j) + A(i,k) * B(k,j)
            end do
        end do
    end do

    call checkpoint_integer(debug = .TRUE., verbosity = 3, msg = 'Finished column multiplication, with', var1 = n)
end subroutine matrix_multiply_column