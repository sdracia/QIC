program test
    implicit none

    integer :: ii, jj
    character(len=1000) :: message


    open(1, file = 'error.txt', status = 'old')
    do ii=1, 32
        jj = ii
        read(1, '(A)') message
        print*, trim(message)
    end do

end program test