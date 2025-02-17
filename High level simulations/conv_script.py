import subprocess
import argparse
import os

base_path = os.path.dirname(os.path.abspath(__file__))
conv_script = os.path.join(base_path, "conv.py")

# Define the command template with placeholders for M, N, P, and float_input
command_template1 = f"/bin/python3 {conv_script} --IMG_WIDTH={{}} --FILT_WIN={{}} --std={{}} --float={{}} --MAC_flag=false"
command_template2 = f"/bin/python3 {conv_script} --IMG_WIDTH={{}} --FILT_WIN={{}} --std={{}} --float={{}} --MAC_flag=true"

# Define the range of float inputs you want to use
float_inputs = ["FP8_CUSTOM", "FP16", "FP16ALT"]  # Add more options as needed
IMG_WIDTH_inputs = ["16"]
FILT_WIN_inputs = ["3"]
std_inputs = ["0.01", "0.5", "1"]

parser = argparse.ArgumentParser()
parser.add_argument('--IMG_WIDTH', type=int, default=16, help='Value for IMG_WIDTH')
parser.add_argument('--FILT_WIN', type=int, default=16, help='Value for FILT_WIN')
parser.add_argument('--std', type=float, default=1, help='Value for std')

# Parse command-line arguments
args = parser.parse_args()

# Iterate over the list of float inputs
for std_input in std_inputs:
    for IMG_WIDTH_input in IMG_WIDTH_inputs :
        for FILT_WIN_input in FILT_WIN_inputs:
            for float_input in float_inputs:
                # Format the command with the current float input
                if float_input == "FP8_CUSTOM" :
                    command = command_template1.format(IMG_WIDTH_input, FILT_WIN_input, std_input, float_input)
                else:
                    command= command_template2.format(IMG_WIDTH_input, FILT_WIN_input, std_input, float_input)

                # Execute the command using subprocess
                try:
                    subprocess.run(command, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Command failed with error: {e}")
