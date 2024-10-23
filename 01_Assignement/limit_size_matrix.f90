program limit_size
    implicit none
    integer, parameter :: max_size = 10000
    integer :: n
    real(8), allocatable :: A(:,:), B(:,:), C(:,:), C_explicit(:,:), C_intrinsic(:,:)
    real(8) :: start_time, end_time
    real(8) :: time_explicit, time_column, time_matmul
    character(len=50) :: filename
    integer :: file_unit

    ! Generate the filename with the maximum matrix size in it
    write(filename, '("limit_max_size_", I0, ".dat")') max_size
    open(unit=20, file=filename, status="replace", action="write")

    ! Write header to the file
    write(20, '(A)') 'MATMUL'

    ! Loop over different matrix sizes
    do n = 100, max_size, 100
        print*, "Matrix size:", n
        allocate(A(n,n), B(n,n), C_explicit(n,n), C_intrinsic(n,n))

        ! Initialize matrices A and B with random values
        call random_number(A)
        call random_number(B)


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
end program limit_size

