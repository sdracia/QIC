program real_precision
    implicit none

    ! ---------------------------------------------------------------
    ! This program illustrates the differences in precision between single
    ! and double-precision floating-point calculations.
    ! 
    ! It calculates the sum of π multiplied by a large factor and the 
    ! square root of 2 also multiplied by a large factor. The single 
    ! precision result may lose some accuracy due to rounding errors, 
    ! while the double precision calculation maintains greater accuracy.
    ! ---------------------------------------------------------------

    real :: pi, sqrt2, result_single       ! Variables for single precision
    real*8 :: pi_d, sqrt2_d, result_double  ! Variables for double precision

    ! Single precision calculations
    pi = 3.1415927 * 1.0e32               ! Approximation of pi in single precision
    sqrt2 = sqrt(2.0) * 1.0e21            ! Approximation of sqrt(2) in single precision
    result_single = pi + sqrt2            ! Sum of single precision values
    print*, "Sum with single precision:", result_single

    ! Double precision calculations
    pi_d = 3.141592653589793238 * 1.0d32  ! More accurate pi in double precision
    sqrt2_d = sqrt(2.0d0) * 1.0d21        ! More accurate sqrt(2) in double precision
    result_double = pi_d + sqrt2_d        ! Sum of double precision values
    print*, "Sum with double precision:", result_double

end program real_precision
