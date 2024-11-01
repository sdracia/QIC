program main
    use mod_matrix_c8
    use debugger
    implicit none
    ! This program tests the functionality of the complex8_matrix derived type
    ! defined in the mod_matrix_c8 module. It initializes matrices, computes
    ! their trace and adjoint, and writes the adjoint matrix to a file.

    type(complex8_matrix) :: A, B, C      ! Declare matrices A, B, and C
    complex(8) :: trace_B, trace_A         ! Variable to store the trace
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

    debug = .true.

    ! Define dimensions for the matrices
    ! size = 3

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

    ! Calculate the trace of the adjoint of A
    trace_A_conjugate = .Tr. A_adjoint

    ! Verify that tr(A^H) = \bar{tr(A)}
    if (abs(trace_A_conjugate - conjg(trace_A)) < 1.0d-5) then
        call checkpoint_character(debug = .true., verbosity = 1, msg = "Verification: tr(A^H) = conjugate of tr(A) is correct")
    else
        call checkpoint_character(debug = .true., verbosity = 1, msg = "Verification: tr(A^H) = conjugate of tr(A) is incorrect")
    end if

    call CMatDumpTXT(A, A_adjoint, trace_A, trace_A_adjoint, 'matrix_output.txt')
    print *, "The original matrix has been written to file: matrix_output.txt"

end program main
