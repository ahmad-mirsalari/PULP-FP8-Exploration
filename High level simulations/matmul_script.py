import subprocess
import argparse
import os

base_path = os.path.dirname(os.path.abspath(__file__))
matrix_script = os.path.join(base_path, "matmul.py")

# Define the command template with placeholders for M, N, P, and float_input
command_template1 = f"/bin/python3 {matrix_script} --M={{}} --N={{}} --P={{}} --std={{}} --float={{}}"
command_template2 = f"/bin/python3 {matrix_script} --M={{}} --N={{}} --P={{}} --std={{}} --float={{}} --MAC_flag=true"

# Define the range of float inputs you want to use
float_inputs = ["FP8_CUSTOM", "FP16", "FP16ALT"]  # Add more options as needed
matrix_inputs = ["16", "32", "64"]
std_inputs = ["0.01", "0.5", "1"]

parser = argparse.ArgumentParser()
parser.add_argument('--M', type=int, default=16, help='Value for M')
parser.add_argument('--N', type=int, default=16, help='Value for N')
parser.add_argument('--P', type=int, default=16, help='Value for P')
parser.add_argument('--std', type=float, default=1, help='Value for std')

# Parse command-line arguments
args = parser.parse_args()

# Iterate over the list of float inputs
for std_input in std_inputs:
    for matrix_input in matrix_inputs:
        for float_input in float_inputs:
            # Format the command with the current float input
            if float_input == "FP8_CUSTOM" :
                command = command_template1.format(matrix_input, matrix_input,matrix_input, std_input, float_input)
            else:
                command= command_template2.format(matrix_input, matrix_input,matrix_input, std_input, float_input)

            # Execute the command using subprocess
            try:
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with error: {e}")
