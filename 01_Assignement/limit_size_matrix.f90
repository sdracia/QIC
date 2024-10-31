program limit_size
    implicit none

    ! ---------------------------------------------------------------
    ! This program benchmarks the Fortran MATMUL function alone
    ! for increasing matrix sizes, testing up to a defined maximum.
    ! Output: Writes computation times to a file, allowing for analysis of 
    ! MATMUL's performance scaling as matrix size grows.
    ! ---------------------------------------------------------------

    integer, parameter :: max_size = 10000  ! Largest matrix size to test
    integer :: n  ! Current matrix size in the loop
    real(8), allocatable :: A(:,:), B(:,:), C_explicit(:,:), C_intrinsic(:,:)
    real(8) :: start_time, end_time, time_matmul
    character(len=50) :: filename  ! Output filename
    integer :: file_unit  ! File I/O unit number

    ! Generate filename and open file to record MATMUL times
    write(filename, '("limit_matmul_max_size_", I0, ".dat")') max_size
    open(unit=20, file=filename, status="replace", action="write")
    write(20, '(A)') 'MATMUL'

    do n = 100, max_size, 100
        print*, "Matrix size:", n
        allocate(A(n,n), B(n,n), C_explicit(n,n), C_intrinsic(n,n))

        ! Initialize matrices A and B with random values
        call random_number(A)
        call random_number(B)

        ! Measure time for MATMUL function
        call cpu_time(start_time)
        C_intrinsic = matmul(A, B)
        call cpu_time(end_time)
        time_matmul = end_time - start_time
        print*, "MATMUL function time:", time_matmul

        ! Write MATMUL time to file
        write(20, '(F12.6)') time_matmul

        deallocate(A, B, C_explicit, C_intrinsic)
    end do

    close(20)
end program limit_size
