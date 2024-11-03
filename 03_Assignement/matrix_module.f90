module matmul_timing
    use debugger

    ! No implicitly declared variable
    implicit none
    
contains 

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
    subroutine perform_multiplications(min_size, max_size, step, seed, opt_flag, type_mult)
        use debugger
        implicit none
        integer, intent(in) :: max_size, step, seed, min_size
        character(len=10), intent(in) :: opt_flag
        character(len=20), intent(in) :: type_mult
        real(8), allocatable :: A(:,:), B(:,:), C_explicit(:,:), C_intrinsic(:,:)
        real(8) :: start_time, end_time, time_explicit, time_column, time_matmul
        character(len=50) :: filename  ! Output filename for performance results
        logical :: flag
        integer :: i ! File unit number for output
        integer, allocatable :: seed_array(:)
        integer :: m

        ! Preconditions
        call checkpoint_real(debug=.TRUE., msg='Beginning matrix multiplication process.')
        if (max_size <= 0 .or. step <= 0 .or. step >= max_size) then
            print*, "Error: Invalid matrix size or step configuration."
            return
        end if

        ! Prepare output file based on user inputs
        call prepare_output_file(filename, type_mult, min_size, max_size, opt_flag, step, seed)

        !open(unit=20, file=filename, status="old", action="write")

        ! SEED MANAGEMENT:

        ! Get the size of the seed array required by random_seed
        call random_seed(size=m)
        allocate(seed_array(m))

        ! Fill the seed array with your seed value
        seed_array = seed

        ! Set the seed for the random number generator
        call random_seed(put=seed_array)

        ! Loop over matrix sizes from step to max_size
        do i = min_size, max_size, step
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
                write(20, '(I6, 3X, F12.6, 3X, F12.6, 3X, F12.6)') i, time_explicit, time_column, time_matmul
            else if (type_mult == "row-col") then
                write(20, '(I6, 3X, F12.6)') i, time_explicit
            else if (type_mult == "col-row") then
                write(20, '(I6, 3X, F12.6)') i, time_column
            else if (type_mult == "matmul") then
                write(20, '(I6, 3X, F12.6)') i, time_matmul
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
    subroutine prepare_output_file(filename, type_mult, min_size, max_size, opt_flag, step, seed)
        implicit none
        character(len=50), intent(out) :: filename
        character(len=20), intent(in) :: type_mult
        integer, intent(in) :: max_size, step, min_size, seed
        character(len=6) :: min_size_str, max_size_str, step_str, seed_str
        character(len=6), intent(in) :: opt_flag
        logical :: flag

        ! Convert integers to strings without padding
        write(min_size_str, '(I0)') min_size
        write(max_size_str, '(I0)') max_size
        write(step_str, '(I0)') step
        write(seed_str, '(I0)') seed

        ! Create the filename without spaces
        write(filename, '(A, A, A, A, A, A, A, A)') "data/" // trim(type_mult) // "_size_", &
            trim(min_size_str), "-" // trim(max_size_str), "_step_", &
            trim(step_str), "_flag_" // trim(opt_flag) // ".dat"
        

        ! Check if file exists, and if not, create it with the header
        inquire(file=filename, exist=flag)
        if (.not. flag) then
            open(unit=20, file=filename, status="replace", action="write")
            ! write(20, '(A)')
            if (type_mult == "ALL") then
                write(20, '(A)') '    Size    Explicit(i-j-k)    Column-major(i-k-j)    MATMUL'
            else if (type_mult == "row-col") then
                write(20, '(A)') '    Size    Explicit(i-j-k)'
            else if (type_mult == "col-row") then
                write(20, '(A)') '    Size    Column-major(i-k-j)'
            else if (type_mult == "matmul") then
                write(20, '(A)') '    Size    MATMUL'
            end if
            !close(20)
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

end module matmul_timing