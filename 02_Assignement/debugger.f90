module debugger
    implicit none
contains
    subroutine checkpoint(debug, verbosity, msg, var1, var2, var3)
        ! Arguments
        logical, intent(in) :: debug                              ! Debug flag
        integer, intent(in), optional :: verbosity                ! Verbosity level
        character(len=*), intent(in), optional :: msg             ! Optional message
        real(8), intent(in), optional :: var1, var2, var3         ! Optional variables

        integer :: vlevel
        vlevel = 1               ! Default verbosity level

        if (present(verbosity)) vlevel = verbosity

        ! Only execute if debugging is enabled
        if (debug) then
            ! Verbosity Level 1: Basic checkpoint message
            if (vlevel == 1) then
                if (present(msg)) then
                    print*, 'Checkpoint:', trim(msg)
                else
                    print*, 'Checkpoint: Debugging checkpoint reached.'
                end if
            end if

            ! Verbosity Level 2: Include additional variables if provided
            if (vlevel == 2) then
                if (present(msg)) then
                    print*, 'Detailed Checkpoint:', trim(msg)
                else
                    print*, 'Detailed Checkpoint: Debugging checkpoint reached.'
                end if
                if (present(var1)) print*, 'Variable 1:', var1
            end if

            ! Verbosity Level 3: Provide additional details for debugging
            if (vlevel == 3) then
                if (present(msg)) then
                    print*, 'Checkpoint reached with verbosity level 3, Full details: ', trim(msg)
                else
                    print*, 'Completly detailed Checkpoint: Debugging checkpoint reached.'
                end if
                if (present(var1)) print*, 'Detailed Variable 1:', var1
                if (present(var2)) print*, 'Detailed Variable 2:', var2
                if (present(var3)) print*, 'Detailed Variable 3:', var3
            end if

            if (vlevel > 3) then
                print*, 'Unvalid value for verbosity. Chose between 1, 2 and 3.'
            end if

            print*, '-------------------------------'
        end if
    end subroutine checkpoint
end module debugger
