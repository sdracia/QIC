program real_precision
    implicit none
    real :: pi, sqrt2, result_single
    real*8 :: pi_d, sqrt2_d, result_double

    ! Single precision calculation
    pi = 3.1415927 * 1.0e32
    sqrt2 = sqrt(2.0) * 1.0e21
    result_single = pi + sqrt2
    print*, "Sum with single precision:", result_single

    ! Double precision calculation
    pi_d = 3.141592653589793238 * 1.0d32
    sqrt2_d = sqrt(2.0d0) * 1.0d21
    result_double = pi_d + sqrt2_d
    print*, "Sum with double precision:", result_double
end program real_precision
