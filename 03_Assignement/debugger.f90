module debugger
    implicit none

contains

    ! Subroutine for checkpoint with real variables
    subroutine checkpoint_real(debug, verbosity, msg, var1, var2, var3)
        logical, intent(in) :: debug
        integer, intent(in), optional :: verbosity
        character(len=*), intent(in), optional :: msg
        real(8), intent(in), optional :: var1, var2, var3

        call checkpoint_core(debug, verbosity, msg, var1, var2, var3)
    end subroutine checkpoint_real

    ! Subroutine for checkpoint with integer variables
    subroutine checkpoint_integer(debug, verbosity, msg, var1, var2, var3)
        logical, intent(in) :: debug
        integer, intent(in), optional :: verbosity
        character(len=*), intent(in), optional :: msg
        integer, intent(in), optional :: var1, var2, var3

        call checkpoint_core(debug, verbosity, msg, var1, var2, var3)
    end subroutine checkpoint_integer

    ! Subroutine for checkpoint with character variables
    subroutine checkpoint_character(debug, verbosity, msg, var1, var2, var3)
        logical, intent(in) :: debug
        integer, intent(in), optional :: verbosity
        character(len=*), intent(in), optional :: msg
        character(len=*), intent(in), optional :: var1, var2, var3

        call checkpoint_core(debug, verbosity, msg, var1, var2, var3)
    end subroutine checkpoint_character

    ! Core checkpoint subroutine to handle verbosity and debugging output
    subroutine checkpoint_core(debug, verbosity, msg, var1, var2, var3)
        logical, intent(in) :: debug
        integer, intent(in), optional :: verbosity
        character(len=*), intent(in), optional :: msg
        class(*), intent(in), optional :: var1, var2, var3

        integer :: vlevel
        vlevel = 1   ! Default verbosity level
        if (present(verbosity)) vlevel = verbosity

        if (debug) then
            if (vlevel == 1) then
                if (present(msg)) then
                    print*, 'Checkpoint:', trim(msg)
                else
                    print*, 'Checkpoint: Debugging checkpoint reached.'
                end if
            end if

            if (vlevel == 2) then
                if (present(msg)) then
                    print*, 'Detailed Checkpoint:', trim(msg)
                else
                    print*, 'Detailed Checkpoint: Debugging checkpoint reached.'
                end if
                if (present(var1)) call print_variable(var1, 'time = ')
            end if

            if (vlevel == 3) then
                if (present(msg)) then
                    print*, 'Full details:', trim(msg)
                else
                    print*, 'Fully detailed Checkpoint: Debugging checkpoint reached.'
                end if
                if (present(var1)) call print_variable(var1, 'n_size = :')
                if (present(var2)) call print_variable(var2, 'rows = ')
                if (present(var3)) call print_variable(var3, 'cols = ')
            end if

            if (vlevel > 3) then
                print*, 'Invalid verbosity value. Choose between 1, 2, and 3.'
            end if

            ! print*, '-------------------------------'
        end if
    end subroutine checkpoint_core

    ! Helper subroutine to print variables of any type
    subroutine print_variable(var, label)
        class(*), intent(in) :: var
        character(len=*), intent(in) :: label

        select type(var)
            type is (real(8))
                print*, label, var
            type is (integer)
                print*, label, var
            type is (character(len=*))
                print*, label, trim(var)
            class default
                print*, label, 'Unknown data type'
        end select
    end subroutine print_variable

end module debugger