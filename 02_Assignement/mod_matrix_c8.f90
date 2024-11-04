!=============================================================
! MODULE: mod_matrix_c8
!
! This module defines the derived type 'complex8_matrix' to handle double
! complex matrices and provides subroutines and functions for initialization,
! adjoint (conjugate transpose), trace calculation, equality checking, and 
! file output.
!
module mod_matrix_c8
    use debugger
    implicit none

    type :: complex8_matrix
        integer, dimension(2) :: size      ! Matrix dimensions (rows, columns)
        complex(8), allocatable :: elem(:,:)   ! Matrix elements
    end type complex8_matrix

    interface operator(.Adj.)
        module procedure CMatAdjoint   ! Operator overload for adjoint (.Adj.)
    end interface operator(.Adj.)

    interface operator(.Tr.)
        module procedure CMatTrace     ! Operator overload for trace (.Tr.)
    end interface operator(.Tr.)

contains

    !=============================================================
    ! SUBROUTINE: initMatrix
    !
    ! Initializes a complex8_matrix instance to specified dimensions
    ! and allocates memory for matrix elements.
    !
    ! INPUTS:
    !    cmx  - Output complex8_matrix to initialize
    !    rows - Number of rows for the matrix
    !    cols - Number of columns for the matrix
    !
    subroutine initMatrix(cmx, rows, cols)
        type(complex8_matrix), intent(out) :: cmx
        integer, intent(in) :: rows, cols

        ! Debugging checkpoint to track matrix initialization dimensions
        call checkpoint_integer(debug = .true., verbosity = 3, msg = "Initializing matrix", var2 = rows, var3 = cols)

        ! Set matrix dimensions
        cmx%size(1) = rows
        cmx%size(2) = cols

        ! Allocate memory for the elements array with specified dimensions
        allocate(cmx%elem(rows, cols))

        ! Initialize all elements to (0.0, 0.0)
        cmx%elem = (0.0d0, 0.0d0)
    end subroutine initMatrix

    !=============================================================
    ! FUNCTION: CMatAdjoint
    !
    ! Computes the adjoint (conjugate transpose) of a complex8_matrix.
    !
    ! INPUT:
    !    cmx    - Input complex8_matrix
    !
    ! RETURNS:
    !    cmxadj - Adjoint (conjugate transpose) of the input matrix
    !
    function CMatAdjoint(cmx) result(cmxadj)
        type(complex8_matrix), intent(in) :: cmx
        type(complex8_matrix) :: cmxadj

        ! Set the size of the adjoint matrix by transposing the dimensions
        cmxadj%size(1) = cmx%size(2)
        cmxadj%size(2) = cmx%size(1)

        ! Allocate memory for the adjoint matrix with transposed dimensions
        allocate(cmxadj%elem(cmxadj%size(1), cmxadj%size(2)))

        ! Compute the adjoint by taking the conjugate transpose of the elements
        cmxadj%elem = conjg(transpose(cmx%elem))
    end function CMatAdjoint

    !=============================================================
    ! FUNCTION: CMatTrace
    !
    ! Calculates the trace of a square complex8_matrix (sum of diagonal elements).
    !
    ! INPUT:
    !    cmx - Input complex8_matrix
    !
    ! RETURNS:
    !    tr  - Trace of the matrix
    !
    function CMatTrace(cmx) result(tr)
        type(complex8_matrix), intent(in) :: cmx
        complex(8) :: tr
        integer :: ii

        ! Initialize trace to zero
        tr = (0.0d0, 0.0d0)

        ! Sum the diagonal elements to compute the trace
        do ii = 1, cmx%size(1)
            tr = tr + cmx%elem(ii, ii)
        end do
    end function CMatTrace

    !=============================================================
    ! SUBROUTINE: CMatDumpTXT
    !
    ! Writes the original matrix, its adjoint, and their traces to a text file.
    !
    ! INPUTS:
    !    cmx              - Input complex8_matrix
    !    cmx_adjoint      - Adjoint of the input matrix
    !    trace_cmx        - Trace of the original matrix
    !    trace_cmx_adjoint - Trace of the adjoint matrix
    !    seed             - Integer seed used for generating matrix values
    !    filename         - Character string for the output filename
    !
    subroutine CMatDumpTXT(cmx, cmx_adjoint, trace_cmx, trace_cmx_adjoint, seed, filename)
        type(complex8_matrix), intent(in) :: cmx, cmx_adjoint
        character(len=50), intent(out) :: filename
        integer :: i, j
        complex(8) :: trace_cmx, trace_cmx_adjoint
        character(len=6) :: dim_1, dim_2, char_seed
        integer :: seed

        ! Convert matrix dimensions and seed to strings for filename
        write(dim_1, '(I0)') cmx%size(1)
        write(dim_2, '(I0)') cmx%size(2)
        write(char_seed, '(I0)') seed
        write(filename, '(A, A, A, A, A, A)') "complex_matrix/matrix_result_" // trim(dim_1) // "x", &
            trim(dim_2), "_seed_" // trim(char_seed) // ".dat"
        
        ! Open file for writing with error check
        open(unit=10, file=filename, status='replace', iostat=i)
        if (i /= 0) then
            print *, "Error opening file: ", filename
            return
        end if

        ! Write matrix dimensions and trace to file
        write(10, *) "Matrices Size: ", cmx%size(1), " x ", cmx%size(2)
        write(10, *) "ORIGINAL MATRIX:"
        write(10, *) "Trace: ", trace_cmx
        write(10, *) "Elements:"

        ! Write elements of the original matrix
        do i = 1, cmx%size(1)
            do j = 1, cmx%size(2)
                write(10, '(A)', advance='no') '('
                write(10, '(F7.4, ",", F7.4)', advance='no') real(cmx%elem(i, j)), aimag(cmx%elem(i, j))
                write(10, '(A)', advance='no') ')'
                if (j < cmx%size(2)) write(10, '(A)', advance='no') ' '  ! Separate elements with space
            end do
            write(10, *)  ! New line after each row
        end do

        write(10, *)
        write(10, *) "ADJOINT MATRIX:"
        write(10, *) "Trace: ", trace_cmx_adjoint
        write(10, *) "Elements:"

        ! Write elements of the adjoint matrix
        do i = 1, cmx_adjoint%size(1)
            do j = 1, cmx_adjoint%size(2)
                write(10, '(A)', advance='no') '('
                write(10, '(F7.4, ",", F7.4)', advance='no') real(cmx_adjoint%elem(i, j)), aimag(cmx_adjoint%elem(i, j))
                write(10, '(A)', advance='no') ')'
                if (j < cmx_adjoint%size(2)) write(10, '(A)', advance='no') ' '  ! Separate elements with space
            end do
            write(10, *)  ! New line after each row
        end do

        close(10)

        ! Debugging checkpoint to confirm file writing
        call checkpoint_character(debug = .true., verbosity = 1, msg = "Matrix written to file", var1 = filename)
    end subroutine CMatDumpTXT

    !=============================================================
    ! SUBROUTINE: matrices_are_equal
    !
    ! Checks if two complex8_matrix instances are equal in size and element values.
    !
    ! INPUTS:
    !    A       - First complex8_matrix for comparison
    !    B       - Second complex8_matrix for comparison
    !
    ! OUTPUT:
    !    isEqual - Logical result indicating if matrices are equal (true) or not (false)
    !
    subroutine matrices_are_equal(A, B, isEqual)
        type(complex8_matrix), intent(in) :: A, B
        logical, intent(out) :: isEqual
        integer :: i, j, rows, cols

        ! Assume matrices are equal initially
        isEqual = .true.

        ! Retrieve matrix dimensions
        rows = A%size(1)
        cols = A%size(2)

        ! Check if dimensions match
        if (rows /= B%size(1) .or. cols /= B%size(2)) then
            isEqual = .false.
            return
        end if

        ! Compare individual elements for equality
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
