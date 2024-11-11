def checkpoint(debug, verbosity=1, msg=None, var1=None, var2=None, var3=None):
    """Checkpoint function for debugging with verbosity control.

    Args:
        debug (bool): Flag to enable debugging output.
        verbosity (int, optional): Level of detail in the output (1, 2, or 3).
        msg (str, optional): Message to display with the checkpoint.
        var1 (optional): First variable to display, typically used for 'time' or 'n_size'.
        var2 (optional): Second variable to display, typically used for 'rows'.
        var3 (optional): Third variable to display, typically used for 'cols'.
    """
    if not debug:
        return  # Exit if debugging is not enabled

    if verbosity == 1:
        if msg:
            print(f'Checkpoint: {msg}')
        else:
            print('Checkpoint: Debugging checkpoint reached.')
            
    elif verbosity == 2:
        if msg:
            print(f'Detailed Checkpoint: {msg}')
        else:
            print('Detailed Checkpoint: Debugging checkpoint reached.')
            
        if var1 is not None:
            print_variable(var1, 'time = ')
            
    elif verbosity == 3:
        if msg:
            print(f'Full details: {msg}')
        else:
            print('Fully detailed Checkpoint: Debugging checkpoint reached.')
        
        if var1 is not None:
            print_variable(var1, 'n_size = ')
        if var2 is not None:
            print_variable(var2, 'rows = ')
        if var3 is not None:
            print_variable(var3, 'cols = ')
    else:
        print('Invalid verbosity value. Choose between 1, 2, and 3.')


def print_variable(var, label):
    """Helper function to print a variable with its label.

    Args:
        var: Variable to display.
        label (str): Label to display with the variable.
    """
    print(f'{label}{var}')
