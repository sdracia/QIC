def checkpoint(debug, verbosity=1, msg=None, var1=None, var2=None, var3=None):
    """
    checkpoint:
        Debugging function with verbosity control to display messages and variable values.

    Parameters
    ----------
    debug : bool
        Flag to enable or disable debugging output.
    verbosity : int, optional
        Level of detail in the output. Valid values are:
            - 1: Basic message.
            - 2: Detailed message with one variable (e.g., 'time').
            - 3: Full details with up to three variables (e.g., 'n_size', 'rows', 'cols').
        Default is 1.
    msg : str, optional
        Custom message to display with the checkpoint. Default is None.
    var1 : optional
        First variable to display, typically used for 'time' or 'n_size'. Default is None.
    var2 : optional
        Second variable to display, typically used for 'rows'. Default is None.
    var3 : optional
        Third variable to display, typically used for 'cols'. Default is None.

    Returns
    -------
    None
        Outputs debug messages to the console when enabled.

    Notes
    -----
    - The function exits early if `debug` is set to `False`.
    - For invalid `verbosity` values, a warning message is printed.
    """
    if not debug:
        return  # Exit immediately if debugging is not enabled

    if verbosity == 1:  # Basic debugging message
        if msg:
            print(f'Checkpoint: {msg}')
        else:
            print('Checkpoint: Debugging checkpoint reached.')

    elif verbosity == 2:  # Detailed debugging message with one variable
        if msg:
            print(f'Detailed Checkpoint: {msg}')
        else:
            print('Detailed Checkpoint: Debugging checkpoint reached.')
        
        if var1 is not None:  # Print the first variable if provided
            print_variable(var1, 'time = ')

    elif verbosity == 3:  # Fully detailed debugging message with up to three variables
        if msg:
            print(f'Full details: {msg}')
        else:
            print('Fully detailed Checkpoint: Debugging checkpoint reached.')
        
        if var1 is not None:  # Print the first variable if provided
            print_variable(var1, 'n_size = ')
        if var2 is not None:  # Print the second variable if provided
            print_variable(var2, 'rows = ')
        if var3 is not None:  # Print the third variable if provided
            print_variable(var3, 'cols = ')

    else:
        print('Invalid verbosity value. Choose between 1, 2, and 3.')  # Warning for invalid verbosity


def print_variable(var, label):
    """
    print_variable:
        Helper function to print a variable with its label.

    Parameters
    ----------
    var : any
        Variable to display. Can be of any type.
    label : str
        Label to display alongside the variable.

    Returns
    -------
    None
        Outputs the labeled variable to the console.
    """
    print(f'{label}{var}')  # Print the variable with its associated label
