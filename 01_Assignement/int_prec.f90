program int_precision
    implicit none

    ! ---------------------------------------------------------------
    ! This program demonstrates integer overflow by performing a simple
    ! addition using two different integer precisions: INTEGER*2 and INTEGER*4.
    ! 
    ! With INTEGER*2, the result will overflow since 2000000 exceeds its range,
    ! producing an incorrect result. Using INTEGER*4, however, handles the 
    ! calculation properly due to its larger range.
    ! ---------------------------------------------------------------

    integer*2 :: int2_result  ! Stores result for INTEGER*2 addition
    integer*4 :: int4_result  ! Stores result for INTEGER*4 addition

    ! INTEGER*2 addition: causes overflow due to limited range
    int2_result = 2000000 + 1
    print*, "Sum with INTEGER*2:", int2_result

    ! INTEGER*4 addition: no overflow, handles large values correctly
    int4_result = 2000000 + 1
    print*, "Sum with INTEGER*4:", int4_result

end program int_precision
