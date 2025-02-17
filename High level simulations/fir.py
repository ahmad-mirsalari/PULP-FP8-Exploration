#!/bin/python3

import os
import argparse
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
    size = true.size(dim=0)
    mse_loss = sum_squared_error / size
    return mse_loss


def matrix_init(IN, dt):
    # iterate through rows of IN
    temp = torch.zeros(len(IN), dtype=dt)
    for k in range(int(IN.shape[0])):
        temp[k] = IN[k]
    return temp
def matrix_init_custom(IN, MANTISA_BITS):
    temp = torch.zeros((IN.shape[0]), dtype=torch.float32)
    # iterate through rows of IN
    for i in range(IN.shape[0]):
       temp[i] = fp8_quantizer(IN[i], torch.tensor(MANTISA_BITS))
    return temp

def convolve(Xs, Fs, dt, outlen, mac_flag, vec_flag, cast_flag, cast_to, MANTISA_BITS):
    Rs = torch.zeros(outlen, dtype=dt[2])
    if vec_flag == "false":
        # iterate through rows of X
        for i in range(len(Xs) - len(Fs)):
            sum = torch.tensor([0], dtype=dt[2])
            # iterate through columns of Y
            for j in range(len(Fs)):
                # iterate through rows of Y
                a = Xs[i + j]
                b = Fs[len(Fs) - 1 - j]

                if mac_flag == "true":
                    a = a.type(torch.float32)
                    b = b.type(torch.float32)
                    sum = sum.type(torch.float32)
                if cast_flag == "true":
                    if cast_to == "FP16":
                        a = a.type(torch.float16)
                        b = b.type(torch.float16)
                    elif cast_to == "FP16ALT":
                        a = a.type(torch.bfloat16)
                        b = b.type(torch.bfloat16)
                sum += a *  b
                if dt[0] == "FP8_CUSTOM":
                    sum = fp8_quantizer(sum, torch.tensor(MANTISA_BITS))
                if mac_flag == "true":
                    sum = sum.type(dt[2])
            Rs[i] = sum
    else:  # vectorization

        if dt == torch.float8:
            vec_step = 4
        else:
            vec_step = 2
        a = torch.zeros(vec_step, dtype=dt)
        b = torch.zeros(vec_step, dtype=dt)
        # iterate through rows of X
        for i in range(len(Xs) - len(Fs)):
            sum = torch.zeros(vec_step, dtype=dt)
            # iterate through columns of Y
            for j in range(0, len(Fs), vec_step):
                # iterate through rows of Y
                for k in range(vec_step):
                    a[k] = Xs[i + j + k]
                    b[k] = Fs[len(Fs) - 1 - j - k]
                '''a = Xs[i + j]
                a1 = Xs[i + j + 1]
                b = Fs[len(Fs) - 1 - j]
                b1 = Fs[len(Fs) - 2 - j]'''
                if mac_flag == "true":
                    a = a.type(torch.float32)
                    b = b.type(torch.float32)
                    sum = sum.type(torch.float32)
                for k in range(vec_step):
                    sum[k] += a[k] * b[k]

                '''sum += a * b
                sum1 += a1 * b1'''
                if mac_flag == "true":
                    sum = sum.type(dt)
            for k in range(0,vec_step):
                Rs[i] += sum[k]
            #Rs[i] = sum + sum1
    return Rs


def write_matrix(matrix_to_write, name, len, file_pointer, float_type):
    matrix_string = ''
    if 'Buffer0' in name:
        file_pointer.write("DATA_LOCATION OUT_TYPE %s[%s];\n" % (name, len))
    else:    
        sz0 = matrix_to_write.size()[0]
        if 'check' in name:
            file_pointer.write("PI_L2 OUT_TYPE %s[] = {" % name)
        elif 'UnitImpulse' in name:
            file_pointer.write("DATA_LOCATION INP_TYPE %s[%s]= {" % (name, len))
        else:
            file_pointer.write("DATA_LOCATION FIL_TYPE %s[%s] = {" % (name, len))

        if float_type == torch.float32:
            rem_part = ")"
        elif float_type == torch.float16:
            rem_part = ", dtype=torch.float16)"
        elif float_type == torch.float8:
            rem_part = ", dtype=torch.float8)"
        elif float_type == torch.bfloat16:
            rem_part = ", dtype=torch.bfloat16)"
        for i in range(sz0):
            matrix_string += str(matrix_to_write[i].item()).replace('tensor(', '').replace(rem_part, '')
            matrix_string += ', '
        file_pointer.write("%s" % matrix_string)
        file_pointer.write("};\n")


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


def get_inital_config():
    #  get arguments  and data format
    parser = argparse.ArgumentParser()
    parser.add_argument('--LENGTH', default=512)
    parser.add_argument('--ORDER', default=100)
    parser.add_argument('--std', default=1)
    parser.add_argument('--MAC_flag', default="true")
    parser.add_argument('--vec_flag', default="false")
    parser.add_argument('--float_type', default='FP32')
    args = parser.parse_args()

    LENGTH = int(args.LENGTH)
    ORDER = int(args.ORDER)
    outlen = LENGTH - ORDER

    std = float(args.std)

    mac_flag = str(args.MAC_flag)
    vec_flag = str(args.vec_flag)
    bits = args.float_type.split(",")
    if (outlen % 2) != 0:
        print("The Length and order should be even")
        raise SystemExit
    return LENGTH, ORDER, std, outlen, bits, mac_flag, vec_flag


def save_data_into_hfile(LENGTH, ORDER, res, filter_conv, input_conv):
    # Generate header file
    f = open('data.h', 'w')
    f.write('\
#include "config.h"\n\n\
#define LENGTH %s\n\
#define ORDER %s\n\
        \n\n' % (LENGTH, ORDER))

    buffer0 = torch.zeros(1)
    write_matrix(res, 'Buffer0', 'LENGTH-ORDER', f, res.dtype)
    write_matrix(filter_conv, 'Filter0', 'ORDER', f, filter_conv.dtype)
    write_matrix(input_conv, 'UnitImpulse', 'LENGTH', f, input_conv.dtype)

    write_matrix(res, 'check', '', f, res.dtype)
    f.close()
def check_cast(datatypes):
    result = len(set(datatypes)) == 1  
    if result : #All Elements in List are Equal
        return "false"
    else: #All Elements in List are Not Equal
        if torch.float32 in datatypes:
            return "false"
        else:
            return "true"

def error_metric(ref, res, output_file):

    # calculate manually because metrics doesn't supprt bfloat16
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

def main():
    # get initial config from parser
    LENGTH, ORDER, std, outlen, bits, mac_flag, vec_flag = get_inital_config()
    mean = 0
    input_ref = torch.normal(mean, std, (LENGTH,1))
    filter_ref = torch.normal(mean, std, (ORDER,1))

    

    # calculate reference output
    ref = convolve(Xs=input_ref, Fs=filter_ref, dt=[torch.float32,torch.float32,torch.float32], outlen=outlen, mac_flag="false", vec_flag="false", cast_flag="false", cast_to="false", MANTISA_BITS=0)

    # set the data types based on the parser input
    datatypes = select_dtypes(bits, 3)

    cast_flag = check_cast(datatypes[0:2])
    cast_to = "FP16ALT"

    output_folder = os.path.join(os.getcwd(), "FIR", str(ORDER), str(std))
    os.makedirs(output_folder, exist_ok=True)
    
    # convert matrices to the desired data types
    if datatypes[0] == "FP8_CUSTOM":
        for MANTISA_BITS in range(1,7):
          input_conv = matrix_init_custom(input_ref, MANTISA_BITS)
          filter_conv = matrix_init_custom(filter_ref, MANTISA_BITS)
          datatypes[2] = torch.float32
          res = convolve(Xs=input_conv, Fs=filter_conv, dt=datatypes, outlen=outlen, mac_flag=mac_flag, vec_flag=vec_flag, cast_flag=cast_flag, cast_to=cast_to,MANTISA_BITS=MANTISA_BITS)
          output_file =  os.path.join(output_folder,f"error_metric_{MANTISA_BITS}_{ORDER}_{std}.txt")
          error_metric(ref, res,output_file)

    else:
      input_conv = matrix_init(input_ref, dt=datatypes[0])
      filter_conv = matrix_init(filter_ref, dt=datatypes[1])  
  
      res = convolve(Xs=input_conv, Fs=filter_conv, dt=datatypes, outlen=outlen, mac_flag=mac_flag, vec_flag=vec_flag, cast_flag=cast_flag, cast_to=cast_to,MANTISA_BITS=0)
      output_file =  os.path.join(output_folder,f"error_metric_{datatypes[0]}_{ORDER}_{std}.txt")
    return None


if __name__ == "__main__":
    main()
    pass