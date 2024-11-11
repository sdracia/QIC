"""
  FUNCTIONS:
"""

# IMPORT ZONE
import subprocess
import sys

# ============================================================================================

def read_parameters(filename):
    """Reads parameters from a specified file and returns them as a dictionary.

    Inputs:
        filename (str): The name of the input file containing parameters.

    Purpose:
        Reads a file containing parameters formatted with values and optional comments, 
        parses them, and converts values into appropriate types.

    Outputs:
        dict: Returns a dictionary with parsed parameters:
              - Nmin (int): Minimum matrix size.
              - Nmax (int): Maximum matrix size.
              - m (int): Number of steps between Nmin and Nmax.
              - seed (int): Random seed.
              - opt_flag (str): Optimization flag for compilation.
              - type_mult (str): Matrix multiplication method.
    Raises:
        ValueError: If parameters cannot be converted to the required types.
        FileNotFoundError: If the specified input file is not found.
    """
    params = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Ignore empty lines and comments
                if line.strip() and not line.strip().startswith('!'):
                    parts = line.split('!', 1)
                    value = parts[0].strip()  # Get the parameter value
                    name = parts[1].strip() if len(parts) > 1 else ''  # Get the name (comment)
                    params[name] = value  # Store in dictionary using comment as key (optional)
        
        # Convert values to appropriate types
        return {
            'Nmin': int(params.get('N_min', 0)),
            'Nmax': int(params.get('N_max', 0)),
            'm': int(params.get('m', 0)),
            'seed': int(params.get('seed', 0)),
            'opt_flag': params.get('opt_flag', ''),
            'type_mult': params.get('type_mult', '')
        }
    
    except ValueError:
        print('Please enter valid integers for Nmin, Nmax, and m.')
        sys.exit(1)
    except FileNotFoundError:
        print('Input parameters file not found.')
        sys.exit(1)


# Compile the Fortran code
def compile_module(module_file):
    """Compiles a Fortran module using gfortran.

    Inputs:
        module_file (str): The name of the Fortran module file to compile.

    Purpose:
        Compiles a Fortran module, producing an object file if successful.

    Outputs:
        bool: Returns True if the compilation is successful, False otherwise.
    """
    try:
        print(f'Compiling {module_file}...')
        subprocess.run(['gfortran', module_file, '-c'], check=True)
        print('Compilation successful!')
    except subprocess.CalledProcessError as e:
        print(f'Error during compilation: {e}')
        return False
    return True

# Compile a Fortran program with object files
def compile_fortran(source_file, exec_name, *object_files):
    """Compiles a Fortran program along with any required modules.

    Inputs:
        source_file (str): The main Fortran source file to compile.
        exec_name (str): The name of the executable to generate.
        object_files (tuple): List of object files for additional modules.

    Purpose:
        Compiles the main Fortran source file with the provided object files to create an executable.

    Outputs:
        bool: Returns True if the compilation is successful, False otherwise.
    """
    try:
        print(f'Compiling {source_file} with additional modules: {object_files}...')
        
        # Create the command with source file and object files
        command = ['gfortran', '-o', exec_name, source_file] + list(object_files)
        subprocess.run(command, check=True)
        print('Compilation successful!')
    except subprocess.CalledProcessError as e:
        print(f'Error during compilation: {e}')
        return False
    return True

# Run the compiled Fortran executable
def run_executable(exec_name, input_data):
    """Runs the Fortran executable with the given input data.

    Inputs:
        exec_name (str): The name of the compiled executable to run.
        input_data (str): The input data to pass to the executable.

    Purpose:
        Runs the executable with specified input data and captures the program's output.
        Prints an error message if execution fails.

    Outputs:
        str: Returns the output captured from the program if successful, or None if an error occurs.
    """
    try:
        # Split the input data into individual lines
        input_lines = input_data.strip().splitlines()
        # Create a dictionary with the relevant named parameters
        params = {
            "Nmin": input_lines[0],
            "Nmax": input_lines[1],
            "n_points": input_lines[2],
            "seed": input_lines[3],
            "flag": input_lines[4],
            "type_mult": input_lines[5],
        }

        # Format the output message
        formatted_input = ', '.join(f"{key}={value}" for key, value in params.items())
        print(f'Running {exec_name} with input: {formatted_input}')

        # Run the executable with the input data
        result = subprocess.run([f'./{exec_name}'], input=input_data, text=True, capture_output=True, shell=True, check=True)
        print('Execution successful.\n')
        
        # Return the output from the program
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f'Error during execution: {e}')
        return None

# Main function to run the program for different values of N
def main(Nmin, Nmax, m, seed, opt_flag, type_mult):
    """Iterates over different matrix sizes and runs the executable for each.

    Inputs:
        Nmin (int): Minimum matrix size.
        Nmax (int): Maximum matrix size.
        m (int): Number of values to test between Nmin and Nmax.
        seed (int): Random seed for reproducibility.
        opt_flag (str): Optimization flag for compilation.
        type_mult (str): Type of matrix multiplication to perform.

    Purpose:
        Generates input data for matrix sizes between Nmin and Nmax, runs the Fortran executable
        with each input, and captures and processes the output for further analysis.

    Outputs:
        None
    """
    min_size = Nmin
    max_size = Nmax
    step = m
    seed = seed
    opt_flag = opt_flag
    type_mult = type_mult

    # Prepare the input data as a single string
    input_data = f"{min_size}\n{max_size}\n{step}\n{seed}\n{opt_flag}\n{type_mult}\n"
    
    # Run the executable with the input data
    output = run_executable(executable_name, input_data)
    
    # Further processing and saving output can be done here if required
    # For example: parse the output, store results, etc.

# ============================================================================================

# MAIN EXECUTION BLOCK
if __name__ == '__main__':
    # Read parameters from an input file (input_params.txt)
    parameters = read_parameters('input_params.txt')
  
    # Extract parameters from the dictionary
    Nmin = parameters['Nmin']
    Nmax = parameters['Nmax']
    m = parameters['m']
    seed = parameters['seed']
    opt_flag = parameters['opt_flag']
    type_mult = parameters['type_mult']

    # Validate inputs
    if Nmin <= 0 or Nmax <= 0 or m <= 0:
        print('All inputs must be greater than 0.')
        sys.exit(1)
    if Nmin >= Nmax:
        print('Nmin must be less than Nmax.')
        sys.exit(1)

    # Validate optimization flag
    valid_opt_flags = {'O1', 'O2', 'O3'}
    if opt_flag not in valid_opt_flags:
        print(f'opt_flag must be one of {", ".join(valid_opt_flags)}.')
        sys.exit(1)

    # Validate multiplication type
    valid_type_mults = {'matmul', 'ALL', 'row-col', 'col-row'}
    if type_mult not in valid_type_mults:
        print(f'type_mult must be one of {", ".join(valid_type_mults)}.')
        sys.exit(1)

    # Define Fortran files and executable name
    fortran_file = 'matrix_progr.f90'
    executable_name = 'matrix_progr.x'
    modules = ['debugger.f90', 'matrix_module.f90']
    object_files = ['debugger.o', 'matrix_module.o']

    # Compile the Fortran modules and program
    if compile_module(modules[0]) and compile_module(modules[1]):
        if compile_fortran(fortran_file, executable_name, *object_files):
            # Run the main function with the given parameters
            main(Nmin, Nmax, m, seed, opt_flag, type_mult)

    print(f"Finished saving on file, named {type_mult}_size_{Nmin}-{Nmax}_step_{m}_flag_{opt_flag}_{seed}.dat")
