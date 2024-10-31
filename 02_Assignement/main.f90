program main
    use debugger
    implicit none
    logical :: debug_mode
    real(8) :: a, b, c

    ! Enable debug mode
    debug_mode = .TRUE.
    a = 1.234
    b = 5.678
    c = 9.101

    ! Call checkpoint with various verbosity levels
    call checkpoint(debug_mode, verbosity = 1, msg = 'Checkpoint at level 1')
    call checkpoint(debug_mode, verbosity = 2, msg = 'Checkpoint at level 2 with variables', var1 = a, var2 = b)
    call checkpoint(debug_mode, verbosity = 3, msg = 'Checkpoint at level 3 with full details', var1 = a, var2 = b, var3 = c)

    ! Disable debug mode and test (no output should occur)
    debug_mode = .FALSE.
    call checkpoint(debug_mode, verbosity = 3, msg = 'This should not print')
end program main
