program matrix_multiplication_performance
    implicit none
    integer, parameter :: max_size = 1700
    integer :: n
    real(8), allocatable :: A(:,:), B(:,:), C(:,:), C_explicit(:,:), C_intrinsic(:,:)
    real(8) :: start_time, end_time
    real(8) :: time_explicit, time_column, time_matmul
    character(len=50) :: filename
    integer :: file_unit

    ! Generate the filename with the maximum matrix size in it
    write(filename, '("matrix_performance_max_size_", I0, ".dat")') max_size
    open(unit=20, file=filename, status="replace", action="write")

    ! Write header to the file
    write(20, '(A)') 'Explicit(i-j-k)    Column-major(i-k-j)    MATMUL'

    ! Loop over different matrix sizes
    do n = 100, max_size, 100
        print*, "Matrix size:", n
        allocate(A(n,n), B(n,n), C_explicit(n,n), C_intrinsic(n,n))

        ! Initialize matrices A and B with random values
        call random_number(A)
        call random_number(B)

        ! ---------------------------------------------------
        ! --------------- EXPLICIT MULTIPLICATION (i-j-k)  ---
        ! ---------------------------------------------------
        call cpu_time(start_time)
        C_explicit = 0.0d0
        call matrix_multiply_explicit(A, B, C_explicit, n)
        call cpu_time(end_time)
        time_explicit = end_time - start_time
        print*, "Explicit multiplication (i-j-k) time:", time_explicit

        ! ---------------------------------------------------
        ! --------------- COLUMN-MAJOR ORDER (i-k-j) ---------
        ! ---------------------------------------------------
        call cpu_time(start_time)
        C_explicit = 0.0d0
        call matrix_multiply_column(A, B, C_explicit, n)
        call cpu_time(end_time)
        time_column = end_time - start_time
        print*, "Column-major order (i-k-j) time:", time_column

        ! ---------------------------------------------------
        ! --------------- INTRINSIC MATMUL FUNCTION ----------
        ! ---------------------------------------------------
        call cpu_time(start_time)
        C_intrinsic = matmul(A, B)
        call cpu_time(end_time)
        time_matmul = end_time - start_time
        print*, "MATMUL function time:", time_matmul

        ! Write the timing data to the file
        write(20, '(F12.6, 3X, F12.6, 3X, F12.6)') time_explicit, time_column, time_matmul

        ! Deallocate matrices
        deallocate(A, B, C_explicit, C_intrinsic)
    end do

    close(20)
end program matrix_multiplication_performance

! ---------------------------------------------------
! --------------- EXPLICIT MATRIX MULTIPLICATION ----
! ---------------------------------------------------
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

! ---------------------------------------------------
! --------------- COLUMN-MAJOR ORDER (i-k-j) --------
! ---------------------------------------------------
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
