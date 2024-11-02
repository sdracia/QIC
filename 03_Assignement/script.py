###############################################
## QUANTUM INFORMATION AND COMPUTING 2024/25 ##
###############################################

# Assignment 3.1 - SCALING OF THE MATRIX-MATRIX MULTIPLICATION
# Consider the code developed in the Exercise 3 from Assignment 1 (matrix-matrix multiplication):
# (a) Write a python script that changes N between two values 'N_min' and 'N_max', and launches the program.
# (b) Store the results of the execution time in different files depending on the multiplication method used.
# (c) Fit the scaling of the execution time for different methods as a function of the input size. Consider
# the largest possible difference between 'N_min' and 'N_max'.
# (d) Plot results for different multiplication methods

"""
  FUNCTIONS:

  compile_module(module_file)
          Inputs   | module_file (str): The name of the Fortran module file to compile.
          Purpose  | Compiles a Fortran module using gfortran.
          Outputs  | Returns True if the compilation is successful, False otherwise.

  compile_fortran(source_file, exec_name, *object_files)
          Inputs   | source_file (str): The main Fortran source file to compile.
                   | exec_name (str): The name of the executable to generate.
                   | object_files (list of str): List of object files for additional modules.
          Purpose  | Compiles the main Fortran source file along with any provided object 
                   | files, using gfortran to create an executable.
          Outputs  | Returns True if the compilation is successful, False otherwise.

  run_executable(exec_name, input_data)
          Inputs   | exec_name (str): The name of the compiled executable to run.
                   | input_data (str): The input data to pass to the executable.
          Purpose  | Runs the Fortran executable with the specified input data, capturing 
                   | the output. If the execution fails, an error is displayed.
          Outputs  | Returns the captured output from the program if successful, or None 
                   | if an error occurs.

  main(Nmin, Nmax, m)
          Inputs   | Nmin (int): The minimum value of matrix size N.
                   | Nmax (int): The maximum value of matrix size N.
                   | m (int): The number of different N values to test between Nmin and Nmax.
          Purpose  | Loops through m evenly spaced values between Nmin and Nmax, launching 
                   | the Fortran program with each value of N, capturing the output, and 
                   | extracting the elapsed times for different multiplication methods. The 
                   | results are saved to separate files (one for each method).
          Outputs  | None

  PRECONDITIONS AND VALIDATIONS:

  - The script validates that all inputs (Nmin, Nmax, and m) are positive integers and that 
    Nmin is less than Nmax, otherwise throws an error.

  USAGE:
    python script.py Nmin Nmax m

    - Nmin: The minimum matrix size.
    - Nmax: The maximum matrix size.
    - m: The number of matrix sizes to test between Nmin and Nmax.

  EXAMPLES:
    To run the script with Nmin=100, Nmax=1000, and m=10:
    python script.py 100 1000 10
    
"""

# IMPORT ZONE
import subprocess
import sys

# ============================================================================================

# FUNCTIONS
def read_parameters(filename):
    """Reads parameters from a given file and returns them as a dictionary."""
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
  try:
    print(f'Compiling {module_file}...')
    subprocess.run(['gfortran', module_file, '-c'], check=True)
    print('Compilation successful!')
  except subprocess.CalledProcessError as e:
    print(f'Error during compilation: {e}')
    return False
  return True

# Compile a fortran program with object files,
def compile_fortran(source_file, exec_name, *object_files):
  try:
    print(f'Compiling {source_file} with additional modules: {object_files}...')
    
    # Create the command with source file and object files
    command = ['gfortran', '-o', exec_name, source_file] + list(object_files)
    subprocess.run(command, check=True)
    print('Compilation successful!')
    print('\n')
  except subprocess.CalledProcessError as e:
    print(f'Error during compilation: {e}')
    return False
  return True

  
def run_executable(exec_name, input_data):
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

        result = subprocess.run([f'./{exec_name}'], input=input_data, text=True, capture_output=True, shell=True, check=True)
        print('Execution successful.')
        print('\n')
        
        # Return the output from the program
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f'Error during execution: {e}')
        return None


# Main function to run the program for different values of n
def main(Nmin, Nmax, m, seed, opt_flag, type_mult):
  min_size = Nmin
  max_size = Nmax       # Example max size
  step = m           # Example step value
  seed = seed            # Example seed value
  opt_flag = opt_flag      # Example optimization flag (string)
  type_mult = type_mult    # Example multiplication type (string)

  # Prepare the input data as a single string
  input = f"{min_size}\n{max_size}\n{step}\n{seed}\n{opt_flag}\n{type_mult}\n"
  output = run_executable(executable_name, input)


# ============================================================================================
# MAIN
if __name__ == '__main__':
  # Leggi i parametri da un file di testo
  # Read parameters from the input file
  parameters = read_parameters('input_params.txt')
  
  # Extract parameters
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

  valid_opt_flags = {'O1', 'O2', 'O3'}
  if opt_flag not in valid_opt_flags:
      print(f'opt_flag must be one of {", ".join(valid_opt_flags)}.')
      sys.exit(1)

  # Validate type_mult
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
      # Run the main function with given Nmin, Nmax, and m
      main(Nmin, Nmax, m, seed, opt_flag, type_mult)
  
  print(f"Finished saving on file, named {type_mult}_size_{Nmin}-{Nmax}_{opt_flag}_step_{m}.dat")
