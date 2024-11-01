module converter_module
    implicit none
contains

    subroutine print_variable(var, label)
        implicit none
        class(*), intent(in) :: var  ! Utilizza un argomento polimorfico
        character(len=*), intent(in) :: label

        ! Stampa la variabile con l'etichetta fornita
        select type(var)
        type is (integer)
            print *, label, var
        type is (real)
            print *, label, var
        type is (character)
            print *, label, trim(var)
        class default
            print *, 'Tipo non supportato.'
        end select
    end subroutine print_variable

end module converter_module

subroutine my_function(var1)
    use converter_module
    implicit none
    class(*), optional :: var1  ! Variabile polimorfa
    character(len=20) :: var1_str  ! Variabile stringa per la conversione

    ! Verifica se var1 è presente
    if (present(var1)) then
        ! Utilizza SELECT TYPE per determinare il tipo di var1
        select type(var1)
        type is (integer)
            write(var1_str, '(I20)') var1  ! Scrivi var1 in var1_str
        type is (real)
            write(var1_str, '(F20.10)') var1  ! Scrivi var1 in var1_str
        type is (character)
            var1_str = trim(var1)  ! Assegna direttamente a var1_str
        class default
            print *, 'Tipo non supportato.'
            return
        end select

        call print_variable(var1, trim(var1_str))  ! Passa la stringa convertita
    else
        print *, 'var1 non è presente.'
    end if
end subroutine my_function

program main
    use converter_module
    implicit none

    integer :: int_var
    real :: real_var
    character(len=20) :: char_var

    int_var = 42
    real_var = 3.14
    char_var = "Hello"

    ! Chiama la funzione passando variabili di tipi diversi
    call my_function(int_var)
    call my_function(real_var)
    call my_function(char_var)
end program main
