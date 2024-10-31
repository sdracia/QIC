program matrix_multiplication_performance
    implicit none

    ! Parameters
    integer :: max_size, step, seed
    character(len=20) :: type_mult
    character(len=10) :: opt_flag
    integer :: io_status

    ! Prompt user for parameters
    do
        print*, "Enter max_size (default 900):"
        read(*, *, IOSTAT=io_status) max_size
        if (io_status == 0 .and. max_size > 0) exit
        print*, "Invalid input. Please enter a positive integer for max_size."
        max_size = 900  ! Default value
    end do

    !do
    !    print*, "Enter n (default 100):"
    !    read(*, *, IOSTAT=io_status) n
    !    if (io_status == 0 .and. n > 0) exit
    !    print*, "Invalid input. Please enter a positive integer for n."
    !    n = 100  ! Default value
    !end do

    do
        print*, "Enter step (default 100, must be less than max_size):"
        read(*, *, IOSTAT=io_status) step
        if (io_status == 0 .and. step > 0 .and. step < max_size) exit
        print*, "Invalid input. Please enter a positive integer less than max_size."
    end do

    do
        print*, "Enter seed (default 12345):"
        read(*, *, IOSTAT=io_status) seed
        if (io_status == 0 .and. seed > 0) exit
        print*, "Invalid input. Please enter a positive integer for seed."
        seed = 12345  ! Default value
    end do

    do
        print*, "Enter optimization flag (O1, O2, O3; default O2):"
        read(*, '(A)', IOSTAT=io_status) opt_flag
        opt_flag = trim(adjustl(opt_flag))
        if (io_status == 0 .and. (opt_flag == "O1" .or. opt_flag == "O2" .or. opt_flag == "O3")) exit
        print*, "Invalid input. Please enter one of O1, O2, O3."
        opt_flag = "O2"  ! Default value
    end do

    do
        print*, "Enter type of multiplication (matmul, row-col, col-row, ALL; default ALL):"
        read(*, '(A)', IOSTAT=io_status) type_mult
        type_mult = trim(adjustl(type_mult))
        if (io_status == 0 .and. (type_mult == "matmul" .or. type_mult == "row-col" &
            .or. type_mult == "col-row" .or. type_mult == "ALL")) exit
        print*, "Invalid input. Please enter one of matmul, row-col, col-row, ALL."
        type_mult = "ALL"  ! Default value
    end do

    ! Call to the main matrix multiplication process
    call perform_multiplications(max_size, step, seed, opt_flag, type_mult)

end program matrix_multiplication_performance


subroutine perform_multiplications(max_size, step, seed, opt_flag, type_mult)
    implicit none
    integer, intent(in) :: max_size, step, seed
    character(len=10), intent(in) :: opt_flag
    character(len=20), intent(in) :: type_mult
    real(8), allocatable :: A(:,:), B(:,:), C(:,:), C_explicit(:,:), C_intrinsic(:,:)
    real(8) :: start_time, end_time, time_explicit, time_column, time_matmul
    character(len=50) :: filename  ! Output filename
    logical :: flag
    integer :: i, file_unit  ! File unit number

    ! Prepare output file
    call prepare_output_file(filename, type_mult, max_size, opt_flag, step)

    open(unit=20, file=filename, status="replace", action="write")

    ! Set the random seed for reproducibility
    call random_seed()

    ! Loop over matrix sizes
    do i = step, max_size, step
        print*, "Matrix size:", i
        allocate(A(i,i), B(i,i), C_explicit(i,i), C_intrinsic(i,i))

        ! Initialize matrices A and B with random values
        call random_number(A)
        call random_number(B)

        ! Measures time for the explicit row-by-column method (i-j-k order)
        call cpu_time(start_time)
        C_explicit = 0.0_8
        call matrix_multiply_explicit(A, B, C_explicit, i)
        call cpu_time(end_time)
        time_explicit = end_time - start_time

        ! Measures time for the column-by-row approach (i-k-j order)
        call cpu_time(start_time)
        C_explicit = 0.0_8
        call matrix_multiply_column(A, B, C_explicit, i)
        call cpu_time(end_time)
        time_column = end_time - start_time

        ! Measures time for Fortran's MATMUL intrinsic function
        call cpu_time(start_time)
        C_intrinsic = matmul(A, B)
        call cpu_time(end_time)
        time_matmul = end_time - start_time

        ! Records the computation times for each method in the output file
        if (type_mult == "ALL") then
            write(20, '(F12.6, 3X, F12.6, 3X, F12.6)') time_explicit, time_column, time_matmul
        else if (type_mult == "row-col") then
            write(20, '(F12.6)') time_explicit
        else if (type_mult == "col-row") then
            write(20, '(F12.6)') time_column
        else if (type_mult == "matmul") then
            write(20, '(F12.6)') time_matmul
        end if

        deallocate(A, B, C_explicit, C_intrinsic)
    end do

    close(20)
end subroutine perform_multiplications

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
    write(filename, '(A, A, A, A, A, A)') "" // trim(type_mult) // "_size_", &
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
            close(20)
        end if
end subroutine prepare_output_file



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