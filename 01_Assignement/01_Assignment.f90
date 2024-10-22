program integer_precision
    implicit none
    integer*2 :: int2_result
    integer*4 :: int4_result

    ! Try sum with INTEGER*2 (this will likely overflow)
    int2_result = 2000000 + 1
    print*, "Sum with INTEGER*2:", int2_result

    ! Sum with INTEGER*4 (this should work fine)
    int4_result = 2000000 + 1
    print*, "Sum with INTEGER*4:", int4_result
end program integer_precision
