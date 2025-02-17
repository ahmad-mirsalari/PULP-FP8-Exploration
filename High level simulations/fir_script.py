import subprocess
import argparse
import os

base_path = os.path.dirname(os.path.abspath(__file__))
fir_script = os.path.join(base_path, "fir.py")

# Define the command template with placeholders for M, N, P, and float_input
command_template1 = f"/bin/python3 {fir_script} --LENGTH={{}} --ORDER={{}} --std={{}} --float={{}}"
command_template2 = f"/bin/python3 {fir_script} --LENGTH={{}} --ORDER={{}} --std={{}} --float={{}} --MAC_flag=true --vec_flag=false"

# Define the range of float inputs you want to use
float_inputs = ["FP8_CUSTOM", "FP16", "FP16ALT"]   # Add more options as needed
length_inputs = ["1024"]
order_inputs = ["100"]
std_inputs = ["0.01", "0.5", "1"]

parser = argparse.ArgumentParser()
parser.add_argument('--LENGTH', type=int, default=512, help='Value for length')
parser.add_argument('--ORDER', type=int, default=100, help='Value for order')
parser.add_argument('--std', type=float, default=1, help='Value for std')

# Parse command-line arguments
args = parser.parse_args()

# Iterate over the list of float inputs
for std_input in std_inputs:
    for length_input in length_inputs:
        for order_input in order_inputs:
            for float_input in float_inputs:
                # Format the command with the current float input
                if float_input == "FP8_CUSTOM" :
                    command = command_template1.format(length_input, order_input, std_input, float_input)
                else:
                    command= command_template2.format(length_input, order_input, std_input, float_input)

                # Execute the command using subprocess
                try:
                    subprocess.run(command, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Command failed with error: {e}")
