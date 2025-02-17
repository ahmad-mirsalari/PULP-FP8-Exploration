import os
import argparse
import sys

import torch
import matplotlib.pyplot as plt
from fp_quantization import fp8_quantizer

def relative_absolute_error(true, pred):
    true_mean = torch.mean(true)
    squared_error_num = torch.sum(torch.abs(true - pred))
    squared_error_den = torch.sum(torch.abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss


def mean_squared_error(true, pred):
    squared_error = torch.square(true - pred)
    sum_squared_error = torch.sum(squared_error)
    size = true.size(dim=0) * true.size(dim=1)
    mse_loss = sum_squared_error / size
    return mse_loss


def matrix_init(IN, dt):
    temp = torch.zeros((IN.shape[0], IN.shape[1]), dtype=dt)
    # iterate through rows of IN
    for i in range(IN.shape[0]):
        # iterate through columns of IN
        for j in range(IN.shape[1]):
            temp[i][j] = IN[i][j]
    return temp

def matrix_init_custom(IN, MANTISA_BITS):
    temp = torch.zeros((IN.shape[0], IN.shape[1]), dtype=torch.float32)
    # iterate through rows of IN
    for i in range(IN.shape[0]):
        # iterate through columns of IN
        for j in range(IN.shape[1]):
            temp[i][j] = fp8_quantizer(IN[i][j], torch.tensor(MANTISA_BITS))
    return temp



def error_metric(ref, res, output_file=None):

    # Calculate metrics manually because metrics don't support bfloat16
    d = ref - res
    mse_f = torch.mean(d**2)
    mae_f = torch.mean(abs(d))
    rmse_f = torch.sqrt(mse_f)
    r2_f = 1-(torch.sum(d**2)/torch.sum((ref-torch.mean(ref))**2))
    results = []

    # Append the results to a list
    results.append("Results of metrics:")
    results.append(f"MAE: {mae_f.item()}")
    results.append(f"MSE: {mse_f.item()}")
    results.append(f"RMSE: {rmse_f.item()}")
    results.append(f"R-Squared: {r2_f.item()}")
    
    rae = relative_absolute_error(ref, res)
    results.append(f"RAE is {rae.item()}")

    # If an output file is provided, write the results to it
    if output_file:
        with open(output_file, "w") as file:
            for result in results:
                file.write(result + "\n")
    
    # Otherwise, print the results to the console
    else:
        for result in results:
            print(result)

# Example usage:
# error_metric(ref_data, res_data, "output.txt")  # Write results to a file
# error_metric(ref_data, res_data)  # Print results to the console

def matrix_mult(Xs, Ys, dt, mac_flag, cast_flag, cast_to, MANTISA_BITS):
    Rs = torch.zeros((Xs.shape[0], Ys.shape[1]), dtype=dt[2])
    # iterate through rows of X
    for i in range(Xs.shape[0]):
        # iterate through columns of Y
        for j in range(Ys.shape[1]):
            temp = torch.tensor([0], dtype=dt[2])
            # iterate through rows of Y
            for k in range(Ys.shape[0]):
                a = Xs[i][k]
                b = Ys[k][j]
                if cast_flag == "true":
                    if cast_to == "FP16":
                        a = a.to(torch.float16)
                        b = b.to(torch.float16)
                    elif cast_to == "FP16ALT":
                        a = a.to(torch.bfloat16)
                        b = b.to(torch.bfloat16)
                if mac_flag == "true":
                    a = a.to(torch.float32)
                    b = b.to(torch.float32)
                    temp = temp.to(torch.float32)
                temp += a * b
                
                if dt[0] == "FP8_CUSTOM":
                   
                    temp = fp8_quantizer(temp, torch.tensor(MANTISA_BITS))
                    
                if mac_flag == "true":
                    temp = temp.to(dt[0])

            Rs[i][j] = temp

    return Rs


def write_matrix(matrix_to_write, name, file_pointer, float_type):
    matrix_string = ''
    sz0 = matrix_to_write.size()[0]
    sz1 = matrix_to_write.size()[1]
    if 'ref' in name:
        file_pointer.write("PI_L2 OUT_TYPE %s[] = {" % name)
    elif 'A_mat' in name:
        file_pointer.write("PI_L2 MA_TYPE %s[] = {" % name)
    else:
        file_pointer.write("PI_L2 MB_TYPE %s[] = {" % name)
    if float_type == torch.float32:
        name = ")"
    elif float_type == torch.float16:
        name = ", dtype=torch.float16)"
    elif float_type == torch.bfloat16:
        name = ", dtype=torch.bfloat16)"
    for i in range(sz0):
        for j in range(sz1):
            matrix_string += str(matrix_to_write[i][j].item()).replace('tensor(', '').replace(name, '')
            matrix_string += ', '
    file_pointer.write("%s" % matrix_string)
    file_pointer.write("};\n")


def get_inital_config():
    # get arguments  and data format
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', default=10)
    parser.add_argument('--N', default=10)
    parser.add_argument('--P', default=10)
    parser.add_argument('--std', default=1)

    parser.add_argument('--MAC_flag', default="false")
    parser.add_argument('--float_type', default='FP8')
    args = parser.parse_args()

    M = int(args.M)
    N = int(args.N)
    P = int(args.P)
    std = float(args.std)

    mac_flag = str(args.MAC_flag)
    bits = args.float_type.split(",")
    return M, N, P, std, bits, mac_flag


def select_dtypes(user_dtypes, num_param):
    types_dict = {
        "FP32": torch.float32,
        "FP16": torch.float16,
        "FP16ALT": torch.bfloat16,
        "FP8_CUSTOM": "FP8_CUSTOM"     
    }
    dtypes = []
    if len(user_dtypes) == 1:
        for i in range(num_param):
            dtypes.append(types_dict[user_dtypes[0]])
    elif len(user_dtypes) == num_param:
        for i in range(num_param):
            dtypes.append(types_dict[user_dtypes[i]])
    else:
        for i in range(len(user_dtypes)):
            dtypes.append(types_dict[user_dtypes[i]])
        if 'FP32' in user_dtypes:
            for i in range(len(user_dtypes), num_param):
                dtypes.append(types_dict["FP32"])
        elif 'FP16' in user_dtypes:
            for i in range(len(user_dtypes), num_param):
                dtypes.append(types_dict["FP16"])
        elif 'FP8_CUSTOM' in user_dtypes:
            for i in range(len(user_dtypes), num_param):
                dtypes.append(types_dict["FP8_CUSTOM"])
        else:
            for i in range(len(user_dtypes), num_param):
                dtypes.append(types_dict["FP16ALT"])
        
    return dtypes

def check_cast(datatypes):
    result = len(set(datatypes)) == 1  
    if result : #All Elements in List are Equal
        return "false"
    else: #All Elements in List are Not Equal
        if torch.float32 in datatypes:
            return "false"
        else:
            return "true"

def save_data_into_hfile(M, N, P, A_mat, B_mat, res):
    # Generate header file
    f = open('data.h', 'w')
    f.write('\
#define M %s\n\
#define N %s\n\
#define P %s\n\n' % (M, N, P))
    write_matrix(A_mat, 'A_mat', f, A_mat.dtype)
    write_matrix(B_mat, 'B_mat', f, B_mat.dtype)
    write_matrix(res, 'ref', f, res.dtype)

    f.close()


def main():
    M, N, P, std, bits, mac_flag = get_inital_config()

    # Create reference matrices

    
    # Parameters: mean and standard deviation
    mean = 0
    A_ref = torch.normal(mean, std, (M, N))
    B_ref = torch.normal(mean, std, (N, P))
    desired_std = 0.1
    A_ref_Scaled = A_ref *(desired_std/std)
    B_ref_Scaled = B_ref *(desired_std/std)
    
    # calculate reference output
    ref_main = matrix_mult(Xs=A_ref, Ys=B_ref, dt=[torch.float32,torch.float32,torch.float32], mac_flag=mac_flag, cast_flag="false",cast_to="false", MANTISA_BITS=0)
    ref_scaled = matrix_mult(Xs=A_ref_Scaled, Ys=B_ref_Scaled, dt=[torch.float32,torch.float32,torch.float32], mac_flag=mac_flag, cast_flag="false",cast_to="false", MANTISA_BITS=0)


    # set the data types based on the parser input
    datatypes = select_dtypes(bits, 3)

    cast_flag = check_cast(datatypes[0:2])
    #print("data types",datatypes)
    cast_to = "FP16"
    #print(f"cast flag is {cast_flag}")

    output_folder = os.path.join(os.getcwd(), "matrix", str(M), str(std))
    os.makedirs(output_folder, exist_ok=True)
    
    if datatypes[0] == "FP8_CUSTOM":
        for MANTISA_BITS in range(2,3):
          A_mat = matrix_init_custom(A_ref, MANTISA_BITS)
          B_mat = matrix_init_custom(B_ref, MANTISA_BITS)
          A_mat_scaled = matrix_init_custom(A_ref_Scaled, MANTISA_BITS)
          B_mat_scaled = matrix_init_custom(B_ref_Scaled, MANTISA_BITS)

          datatypes[2] = torch.float32
          res = matrix_mult(Xs=A_mat, Ys=B_mat, dt=datatypes, mac_flag=mac_flag, cast_flag=cast_flag, cast_to = cast_to, MANTISA_BITS=MANTISA_BITS)
          res_scaled = matrix_mult(Xs=A_mat_scaled, Ys=B_mat_scaled, dt=datatypes, mac_flag=mac_flag, cast_flag=cast_flag, cast_to = cast_to, MANTISA_BITS=MANTISA_BITS)
          res_app = ((res_scaled).float() / ((desired_std / std)**2)).float()
          output_file =  os.path.join(output_folder,f"error_metric_{MANTISA_BITS}_{M}_{std}.txt")
          error_metric(ref_main, res_app, output_file)

    else:
        A_mat = matrix_init(A_ref, dt=datatypes[0])
        B_mat = matrix_init(B_ref, dt=datatypes[1])
        res = matrix_mult(Xs=A_mat, Ys=B_mat, dt=datatypes, mac_flag=mac_flag, cast_flag=cast_flag, cast_to = cast_to, MANTISA_BITS=0)
        A_mat_scaled = matrix_init(A_ref_Scaled, dt=datatypes[0])
        B_mat_scaled = matrix_init(B_ref_Scaled, dt=datatypes[1])
        res_scaled = matrix_mult(Xs=A_mat_scaled, Ys=B_mat_scaled, dt=datatypes, mac_flag=mac_flag, cast_flag=cast_flag, cast_to = cast_to, MANTISA_BITS=0)
        res_app = res_scaled / ((desired_std / std)**2)
        output_file =  os.path.join(output_folder,f"error_metric_{datatypes[0]}_{M}_{std}.txt")
        error_metric(ref_main, res_app, output_file)


    save_data_into_hfile(M, N, P, A_mat, B_mat, res)

    return None


if __name__ == "__main__":
    main()
    pass
