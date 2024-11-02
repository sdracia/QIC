module mod_matrix_c8
    use debugger
    implicit none
    ! Module: mod_matrix_c8
    ! This module defines a derived type 'complex8_matrix' for handling double complex matrices.
    ! It includes subroutines and functions for initializing matrices, calculating the trace,
    ! computing the adjoint (conjugate transpose), checking matrix equality, and outputting
    ! matrix data to a text file. The module also supports the `.Adj.` and `.Tr.` operators
    ! for adjoint and trace operations, respectively.

    type :: complex8_matrix
        ! A derived type to represent a double complex matrix.
        ! Components:
        ! - size: Array of two integers specifying the dimensions (rows and columns).
        ! - elem: Allocatable 2D array of complex(8) numbers representing matrix elements.
        integer, dimension(2) :: size
        complex(8), allocatable :: elem(:,:)
    end type complex8_matrix

    interface operator(.Adj.)
        ! Interface to overload the `.Adj.` operator for computing the adjoint of a matrix.
        module procedure CMatAdjoint
    end interface operator(.Adj.)

    interface operator(.Tr.)
        ! Interface to overload the `.Tr.` operator for computing the trace of a matrix.
        module procedure CMatTrace
    end interface operator(.Tr.)

contains

    subroutine initMatrix(cmx, rows, cols)
        ! Subroutine: initMatrix
        ! Initializes a complex8_matrix instance to the specified dimensions and
        ! allocates memory for matrix elements.
        ! Parameters:
        ! - cmx: Output complex8_matrix to initialize.
        ! - rows: Number of rows in the matrix.
        ! - cols: Number of columns in the matrix.
        type(complex8_matrix), intent(out) :: cmx
        integer, intent(in) :: rows, cols

        call checkpoint_integer(debug = .true., verbosity = 3, msg = "Initializing matrix", var2 = rows, var3 = cols)

        cmx%size(1) = rows
        cmx%size(2) = cols
        allocate(cmx%elem(rows, cols))
        cmx%elem = (0.0d0, 0.0d0)
    end subroutine initMatrix

    function CMatAdjoint(cmx) result(cmxadj)
        ! Function: CMatAdjoint
        ! Computes the adjoint (conjugate transpose) of the input complex8_matrix.
        ! Parameters:
        ! - cmx: Input complex8_matrix to find the adjoint of.
        ! Returns:
        ! - cmxadj: Adjoint matrix of the input.
        type(complex8_matrix), intent(in) :: cmx
        type(complex8_matrix) :: cmxadj

        cmxadj%size(1) = cmx%size(2)
        cmxadj%size(2) = cmx%size(1)
        allocate(cmxadj%elem(cmxadj%size(1), cmxadj%size(2)))
        cmxadj%elem = conjg(transpose(cmx%elem))
    end function CMatAdjoint

    function CMatTrace(cmx) result(tr)
        ! Function: CMatTrace
        ! Calculates the trace of a square complex8_matrix, which is the sum of the diagonal elements.
        ! Parameters:
        ! - cmx: Input complex8_matrix whose trace is to be calculated.
        ! Returns:
        ! - tr: Complex(8) scalar representing the trace of the matrix.
        type(complex8_matrix), intent(in) :: cmx
        complex(8) :: tr
        integer :: ii

        tr = (0.0d0, 0.0d0)
        do ii = 1, cmx%size(1)
            tr = tr + cmx%elem(ii, ii)
        end do
    end function CMatTrace

    subroutine CMatDumpTXT(cmx, cmx_adjoint, trace_cmx, trace_cmx_adjoint, seed, filename)
        ! Subroutine: CMatDumpTXT
        ! Writes the original matrix, its adjoint, and their traces to a text file in a readable format.
        ! Parameters:
        ! - cmx: Input complex8_matrix to output.
        ! - cmx_adjoint: Adjoint of the input matrix.
        ! - trace_cmx: Trace of the original matrix.
        ! - trace_cmx_adjoint: Trace of the adjoint matrix.
        ! - seed: Integer seed used for generating matrix values.
        ! - filename: Character string for the output filename (auto-generated).
        type(complex8_matrix), intent(in) :: cmx
        type(complex8_matrix), intent(in) :: cmx_adjoint
        character(len=50), intent(out) :: filename
        integer :: i, j
        complex(8) :: trace_cmx, trace_cmx_adjoint
        character(len=6) :: dim_1, dim_2, char_seed
        integer :: seed

        write(dim_1, '(I0)') cmx%size(1)
        write(dim_2, '(I0)') cmx%size(2)
        write(char_seed, '(I0)') seed
        write(filename, '(A, A, A, A, A, A)') "matrix_result_" // trim(dim_1) // "x" // trim(dim_2), "_seed_" // trim(char_seed) // ".dat"
        
        open(unit=10, file=filename, status='replace', iostat=i)
        if (i /= 0) then
            print *, "Error opening file: ", filename
            return
        end if

        write(10, *) "Matrices Size: ", cmx%size(1), " x ", cmx%size(2)
        write(10, *) "ORIGINAL MATRIX:"
        write(10, *) "Trace: ", trace_cmx
        write(10, *) "Elements:"

        do i = 1, cmx%size(1)
            do j = 1, cmx%size(2)
                write(10, '(A)', advance='no') '('
                write(10, '(F7.4, ",", F7.4)', advance='no') real(cmx%elem(i, j)), aimag(cmx%elem(i, j))
                write(10, '(A)', advance='no') ')'
                if (j < cmx%size(2)) write(10, '(A)', advance='no') ' '
            end do
            write(10, *)
        end do

        write(10, *)
        write(10, *) "ADJOINT MATRIX:"
        write(10, *) "Trace: ", trace_cmx_adjoint
        write(10, *) "Elements:"

        do i = 1, cmx_adjoint%size(1)
            do j = 1, cmx_adjoint%size(2)
                write(10, '(A)', advance='no') '('
                write(10, '(F7.4, ",", F7.4)', advance='no') real(cmx_adjoint%elem(i, j)), aimag(cmx_adjoint%elem(i, j))
                write(10, '(A)', advance='no') ')'
                if (j < cmx_adjoint%size(2)) write(10, '(A)', advance='no') ' '
            end do
            write(10, *)
        end do

        close(10)
        call checkpoint_character(debug = .true., verbosity = 1, msg = "Matrix written to file", var1 = filename)
    end subroutine CMatDumpTXT

    subroutine matrices_are_equal(A, B, isEqual)
        ! Subroutine: matrices_are_equal
        ! Compares two complex8_matrix instances to determine if they are equal in size and element values.
        ! Parameters:
        ! - A: First complex8_matrix to compare.
        ! - B: Second complex8_matrix to compare.
        ! - isEqual: Logical output indicating whether matrices are equal (true) or not (false).
        type(complex8_matrix), intent(in) :: A, B
        logical, intent(out) :: isEqual
        integer :: i, j, rows, cols

        isEqual = .true.
        rows = A%size(1)
        cols = A%size(2)

        if (rows /= B%size(1) .or. cols /= B%size(2)) then
            isEqual = .false.
            return
        end if

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
