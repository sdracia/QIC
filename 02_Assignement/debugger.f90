!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! QUANTUM INFORMATION AND COMPUTING 2024/25 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Assignment 2.2 - CHECKPOINT MODULE WITH CORE SUBROUTINE
! This module extends the checkpoint functionality by providing
! subroutines specialized for real, integer, and character variables.
! 
! - Features:
!   (a) Includes control via a logical variable (Debug=.TRUE. or .FALSE.)
!   (b) Offers verbosity levels (controlled by optional integer 'verbosity' parameter)
!       - Level 1: Basic checkpoint message.
!       - Level 2: Detailed checkpoint with optional variable printout.
!       - Level 3: Full verbosity message with additional variable printout.
!   (c) Allows printing of optional, user-defined message and variables.
!   (d) Supports mixed-type variables through a core subroutine, `checkpoint_core`.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Module Overview:
! - This module provides three subroutines: checkpoint_real, checkpoint_integer, 
!   and checkpoint_character, which handle checkpoints for different data types.
! - All three subroutines call a core subroutine, `checkpoint_core`, which processes 
!   verbosity levels and debugging output.
!
! - The modularized structure allows reusability and flexibility in the debugging 
!   process, supporting type-specific checkpoints for real, integer, and character data.
!
! - A helper subroutine, `print_variable`, manages the type detection and printing 
!   of any variable passed to the core checkpoint.

module debugger
    implicit none

contains

    ! Subroutine for checkpoint with real variables
    ! - Parameters:
    !   * debug: logical flag for enabling/disabling debug output
    !   * verbosity: optional integer verbosity level (1, 2, or 3)
    !   * msg: optional message string for custom checkpoint message
    !   * var1, var2, var3: optional real variables for additional information
    subroutine checkpoint_real(debug, verbosity, msg, var1, var2, var3)
        logical, intent(in) :: debug
        integer, intent(in), optional :: verbosity
        character(len=*), intent(in), optional :: msg
        real(8), intent(in), optional :: var1, var2, var3

        call checkpoint_core(debug, verbosity, msg, var1, var2, var3)
    end subroutine checkpoint_real

    ! Subroutine for checkpoint with integer variables
    ! - Parameters are similar to checkpoint_real but for integer data type
    subroutine checkpoint_integer(debug, verbosity, msg, var1, var2, var3)
        logical, intent(in) :: debug
        integer, intent(in), optional :: verbosity
        character(len=*), intent(in), optional :: msg
        integer, intent(in), optional :: var1, var2, var3

        call checkpoint_core(debug, verbosity, msg, var1, var2, var3)
    end subroutine checkpoint_integer

    ! Subroutine for checkpoint with character variables
    ! - Parameters are similar to checkpoint_real but for character data type
    subroutine checkpoint_character(debug, verbosity, msg, var1, var2, var3)
        logical, intent(in) :: debug
        integer, intent(in), optional :: verbosity
        character(len=*), intent(in), optional :: msg
        character(len=*), intent(in), optional :: var1, var2, var3

        call checkpoint_core(debug, verbosity, msg, var1, var2, var3)
    end subroutine checkpoint_character

    ! Core checkpoint subroutine to handle verbosity and debugging output
    ! - Manages verbosity levels, optional message, and variable printing.
    ! - Parameters:
    !   * debug: controls whether debug output is active
    !   * verbosity: integer (optional) to set level of detail (1, 2, or 3)
    !   * msg: optional message to customize the checkpoint output
    !   * var1, var2, var3: optional variables to print based on verbosity
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
        end if
    end subroutine checkpoint_core

    ! Helper subroutine to print variables of any type
    ! - Uses type detection to print real, integer, and character variables
    ! - Parameters:
    !   * var: variable to print
    !   * label: descriptive label for the printed variable
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
