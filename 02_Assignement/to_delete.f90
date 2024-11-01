subroutine matrices_are_equal(A, B, isEqual)
    use mod_matrix_c8
    implicit none
    type(complex8_matrix), intent(in) :: A, B
    logical, intent(out) :: isEqual
    integer :: i, j, rows, cols

    ! Initialize the equality flag to true
    isEqual = .true.

    ! Get the dimensions of the matrices
    rows = A%nrows
    cols = A%ncols

    ! Check if the dimensions match
    if (rows /= B%nrows .or. cols /= B%ncols) then
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
