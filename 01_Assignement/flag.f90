program limit_size
    implicit none

    ! ---------------------------------------------------------------
    ! This program benchmarks the performance of an explicit matrix 
    ! multiplication algorithm for varying matrix sizes up to a maximum 
    ! of 2500x2500. The execution time for the multiplication is recorded 
    ! and written to a file for analysis.
    ! 
    ! The multiplication is performed using a nested loop following the 
    ! (i-j-k) order, which is typical in Fortran.
    ! ---------------------------------------------------------------

    integer, parameter :: max_size = 2500  ! Maximum size for matrices
    integer :: n  ! Current size of the matrices
    real(8), allocatable :: A(:,:), B(:,:), C(:,:), C_explicit(:,:), C_intrinsic(:,:)
    real(8) :: start_time, end_time, time_explicit
    character(len=50) :: filename  ! Filename for output
    integer :: file_unit  ! File unit for output operations

    ! Prepare output filename and open file
    write(filename, '("flag_03_explicit_", I0, ".dat")') max_size
    open(unit=20, file=filename, status="replace", action="write")
    write(20, '(A)') 'Explicit'  ! Write header to file

    ! Loop over different matrix sizes
    do n = 100, max_size, 100
        print*, "Matrix size:", n  ! Display current matrix size
        allocate(A(n,n), B(n,n), C_explicit(n,n), C_intrinsic(n,n))

        ! Initialize matrices A and B with random values
        call random_number(A)
        call random_number(B)

        ! Measure time for explicit matrix multiplication (i-j-k)
        call cpu_time(start_time)
        C_explicit = 0.0d0  ! Initialize result matrix
        call matrix_multiply_explicit(A, B, C_explicit, n)  ! Call multiplication subroutine
        call cpu_time(end_time)
        time_explicit = end_time - start_time  ! Calculate elapsed time
        print*, "Explicit multiplication (i-j-k) time:", time_explicit

        ! Write elapsed time to the output file
        write(20, '(F12.6)') time_explicit

        deallocate(A, B, C_explicit, C_intrinsic)  ! Deallocate matrices
    end do

    close(20)  ! Close the output file
end program limit_size

! Subroutine for explicit matrix multiplication (i-j-k)
subroutine matrix_multiply_explicit(A, B, C, n)
    implicit none
    integer, intent(in) :: n  ! Size of the matrices
    real(8), intent(in) :: A(n,n), B(n,n)  ! Input matrices
    real(8), intent(inout) :: C(n,n)  ! Output result matrix
    integer :: i, j, k

    ! Perform matrix multiplication using explicit (i-j-k) order
    do i = 1, n
        do j = 1, n
            do k = 1, n
                C(i,j) = C(i,j) + A(i,k) * B(k,j)  ! Update result
            end do
        end do
    end do
end subroutine matrix_multiply_explicit
