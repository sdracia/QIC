program matrix_multiplication_performance
    implicit none

    ! ---------------------------------------------------------------
    ! This program is designed to evaluate and compare the speed of 
    ! three different matrix multiplication techniques: an explicit 
    ! row-by-column approach (i-j-k order), a column-by-row method 
    ! (i-k-j order), and Fortran’s built-in MATMUL function.
    !
    ! I increase the matrix size, performing 
    ! each multiplication method and measuring the time taken. Results 
    ! are saved to a file for easy review and analysis, allowing us to 
    ! understand which method scales best with larger matrices.
    ! ---------------------------------------------------------------

    ! the parameters are: max_sixe; n; step; seed; filename; flag; type of matrix multipl; 

    integer, parameter :: max_size = 2000  ! Largest matrix size to test
    integer :: n  ! Current matrix size in the loop
    real(8), allocatable :: A(:,:), B(:,:), C(:,:), C_explicit(:,:), C_intrinsic(:,:)
    real(8) :: start_time, end_time, time_explicit, time_column, time_matmul
    character(len=50) :: filename  ! Name for the output file
    integer :: file_unit  ! Unit number for file operations

    ! Creates a filename for the output file and opens it for writing
    write(filename, '("matrix_performance", I0, ".dat")') max_size
    open(unit=20, file=filename, status="replace", action="write")
    write(20, '(A)') 'Explicit(i-j-k)    Column-major(i-k-j)    MATMUL'

    do n = 100, max_size, 100
        print*, "Matrix size:", n
        allocate(A(n,n), B(n,n), C_explicit(n,n), C_intrinsic(n,n))

        ! Initializes matrices A and B with random values to simulate
        ! real-world data for each multiplication test.
        call random_number(A)
        call random_number(B)

        ! Measures time for the explicit row-by-column method (i-j-k order)
        call cpu_time(start_time)
        C_explicit = 0.0d0
        call matrix_multiply_explicit(A, B, C_explicit, n)
        call cpu_time(end_time)
        time_explicit = end_time - start_time

        ! Measures time for the column-by-row approach (i-k-j order)
        call cpu_time(start_time)
        C_explicit = 0.0d0
        call matrix_multiply_column(A, B, C_explicit, n)
        call cpu_time(end_time)
        time_column = end_time - start_time

        ! Measures time for Fortran's MATMUL intrinsic function
        call cpu_time(start_time)
        C_intrinsic = matmul(A, B)
        call cpu_time(end_time)
        time_matmul = end_time - start_time

        ! Records the computation times for each method in the output file
        write(20, '(F12.6, 3X, F12.6, 3X, F12.6)') time_explicit, time_column, time_matmul

        deallocate(A, B, C_explicit, C_intrinsic)
    end do

    close(20)
end program matrix_multiplication_performance

! ---------------------------------------------------------------
! This subroutine performs matrix multiplication using a nested 
! loop order of i-j-k, iterating first by rows, then columns, 
! and finally the innermost loop by depth. It’s designed to provide 
! a baseline comparison for performance.
! ---------------------------------------------------------------
subroutine matrix_multiply_explicit(A, B, C, n)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: A(n,n), B(n,n)
    real(8), intent(inout) :: C(n,n)
    integer :: i, j, k

    do i = 1, n
        do j = 1, n
            do k = 1, n
                C(i,j) = C(i,j) + A(i,k) * B(k,j)
            end do
        end do
    end do
end subroutine matrix_multiply_explicit

! ---------------------------------------------------------------
! This subroutine applies an alternative nested loop order of 
! i-k-j, iterating first by rows, then depth, and finally by columns.
! This structure provides insight into the performance impact 
! of changing loop order in Fortran’s column-major layout.
! ---------------------------------------------------------------
subroutine matrix_multiply_column(A, B, C, n)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: A(n,n), B(n,n)
    real(8), intent(inout) :: C(n,n)
    integer :: i, j, k

    do i = 1, n
        do k = 1, n
            do j = 1, n
                C(i,j) = C(i,j) + A(i,k) * B(k,j)
            end do
        end do
    end do
end subroutine matrix_multiply_column
