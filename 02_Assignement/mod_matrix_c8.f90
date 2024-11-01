module mod_matrix_c8
    use debugger
    implicit none
    ! This module defines a derived type for a double complex matrix
    ! and provides various operations such as initialization, trace calculation,
    ! adjoint calculation, and output functionality.

    type :: complex8_matrix
        ! Store the dimensions of the matrix
        integer, dimension(2) :: size
        ! Store the elements of the matrix (make it allocatable)
        complex(8), allocatable :: elem(:,:)
    end type complex8_matrix

    ! Define the interface for the adjoint operator
    interface operator(.Adj.)
        module procedure CMatAdjoint
    end interface operator(.Adj.)

    ! Define the interface for the trace operator
    interface operator(.Tr.)
        module procedure CMatTrace
    end interface operator(.Tr.)

contains

    ! Subroutine to initialize the matrix
    subroutine initMatrix(cmx, rows, cols)
        ! Initializes a complex8_matrix with specified dimensions.
        type(complex8_matrix), intent(out) :: cmx  ! Output matrix to initialize
        integer, intent(in) :: rows, cols           ! Input dimensions

        call checkpoint_integer(debug = .true., verbosity = 3, msg = "Initializing matrix", var2 = rows, var3 = cols)


        ! Set the size of the matrix
        cmx%size(1) = rows
        cmx%size(2) = cols

        ! Allocate the matrix elements
        allocate(cmx%elem(rows, cols))

        ! Initialize elements to zero
        cmx%elem = (0.0d0, 0.0d0)
    end subroutine initMatrix

    ! Function to calculate the adjoint of the matrix
    function CMatAdjoint(cmx) result(cmxadj)
        ! Computes the adjoint (conjugate transpose) of a complex matrix.
        type(complex8_matrix), intent(in) :: cmx           ! Input matrix
        type(complex8_matrix) :: cmxadj                     ! Output adjoint matrix

        ! Set dimensions for the adjoint matrix
        cmxadj%size(1) = cmx%size(2)
        cmxadj%size(2) = cmx%size(1)

        ! Allocate the adjoint matrix
        allocate(cmxadj%elem(cmxadj%size(1), cmxadj%size(2)))

        ! Compute the adjoint: conjugate transpose
        cmxadj%elem = conjg(transpose(cmx%elem))
    end function CMatAdjoint

    ! Function to calculate the trace of the matrix
    function CMatTrace(cmx) result(tr)
        ! Computes the trace of a square complex matrix.
        type(complex8_matrix), intent(in) :: cmx   ! Input matrix
        complex(8) :: tr                            ! Output trace
        integer :: ii                                ! Loop variable

        ! Initialize trace to zero
        tr = (0.0d0, 0.0d0)

        ! Sum the diagonal elements
        do ii = 1, cmx%size(1)
            tr = tr + cmx%elem(ii, ii)
        end do
    end function CMatTrace

    ! Subroutine to write the matrix to a file in a readable format
    subroutine CMatDumpTXT(cmx, cmx_adjoint, trace_cmx, trace_cmx_adjoint, filename)
        ! Writes the matrix to a specified text file.
        type(complex8_matrix), intent(in) :: cmx                ! Input matrix to write
        type(complex8_matrix), intent(in) :: cmx_adjoint                ! Input matrix to write
        character(len=*), intent(in) :: filename                 ! Output filename
        integer :: i, j                                          ! Loop variables
        complex(8) :: trace_cmx
        complex(8) :: trace_cmx_adjoint

        ! Open file for writing and check for errors
        open(unit=10, file=filename, status='replace', iostat=i)
        if (i /= 0) then
            print *, "Error opening file: ", filename
            return
        end if

        ! Write the dimensions
        write(10, *) "Matrices Size: ", cmx%size(1), " x ", cmx%size(2)
        write(10, *) "ORIGINAL MATRIX:"
        write(10, *) "Trace: ", trace_cmx
        write(10, *) "Elements:"

        ! Write the elements of the matrix
        do i = 1, cmx%size(1)
            do j = 1, cmx%size(2)
                write(10, '(A)', advance='no') '('  ! Start with '(' for each complex number
                write(10, '(F7.4, ",", F7.4)', advance='no') real(cmx%elem(i, j)), aimag(cmx%elem(i, j))  ! Real and imaginary with comma separator
                write(10, '(A)', advance='no') ')'  ! End with ')'
                if (j < cmx%size(2)) write(10, '(A)', advance='no') ' '  ! Space between elements in the row
            end do
            write(10, *)  ! New line for each row
        end do

        write(10, *)

        write(10, *) "ADJOINT MATRIX:"
        write(10, *) "Trace: ", trace_cmx_adjoint
        write(10, *) "Elements:"

        ! Write the elements of the matrix
        do i = 1, cmx_adjoint%size(1)
            do j = 1, cmx_adjoint%size(2)
                write(10, '(A)', advance='no') '('  ! Start with '(' for each complex number
                write(10, '(F7.4, ",", F7.4)', advance='no') real(cmx_adjoint%elem(i, j)), aimag(cmx_adjoint%elem(i, j))  ! Real and imaginary with comma separator
                write(10, '(A)', advance='no') ')'  ! End with ')'
                if (j < cmx_adjoint%size(2)) write(10, '(A)', advance='no') ' '  ! Space between elements in the row
            end do
            write(10, *)  ! New line for each row
        end do


        close(10)

        call checkpoint_character(debug = .true., verbosity = 1, msg = "Matrix written to file", var1 = filename)

    end subroutine CMatDumpTXT

    subroutine matrices_are_equal(A, B, isEqual)
        type(complex8_matrix), intent(in) :: A, B
        logical, intent(out) :: isEqual
        integer :: i, j, rows, cols
    
        ! Initialize the equality flag to true
        isEqual = .true.
    
        ! Get the dimensions of the matrices
        rows = A%size(1)
        cols = A%size(2)
    
        ! Check if the dimensions match
        if (rows /= B%size(1) .or. cols /= B%size(2)) then
            isEqual = .false.
            return
        end if
    
        ! Compare each element
        do i = 1, rows
            do j = 1, cols
                if (A%elem(i, j) /= B%elem(i, j)) then
                    isEqual = .false.
                    return
                end if
            end do
        end do
    end subroutine matrices_are_equal
    

    

end module mod_matrix_c8
