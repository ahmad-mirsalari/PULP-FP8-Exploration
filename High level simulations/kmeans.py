import pandas as pd
import os
import torch
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from fp_quantization import fp8_quantizer
'''
If you are working offline in local computer, just set folder_addr to '.' or absolute address of project folder in your computer, 
but if you are working in online, you may set folder_addr to the path of the project in the Internet.
'''
folder_addr = "."


class KMeansClustering:
    def __init__(self, X, num_clusters, dt, mac_flag, vec_flag, MANTISA_BITS):
        self.K = num_clusters
        self.max_iterations = 300
        self.plot_figure = False
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]
        self.dt = dt
        self.mac_flag = mac_flag
        self.vec_flag = vec_flag
        self.mantissa_bits = MANTISA_BITS

    def initialize_random_centroids(self, X):
        centroids = torch.zeros((self.K, self.num_features), dtype=self.dt[1])

        for k in range(self.K):
            centroid = X[k]  # np.random.random(self.num_features,)#X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid
        return centroids

    def mysum(self, point, centroids, mac_flag, vec_flag):

        if vec_flag == "false":
            diff = torch.zeros(self.K, dtype=self.dt[1])
            for i in range(centroids.shape[0]):
                temp = torch.zeros(1, dtype=self.dt[1])
                for j in range(centroids.shape[1]):
                    a = point[j]
                    b = centroids[i][j]
                    if mac_flag == "true":
                        a = a.type(torch.float32)
                        b = b.type(torch.float32)
                        temp = temp.type(torch.float32)
                    temp += (a - b) ** 2
                    if self.dt[0] == "FP8_CUSTOM":
                        temp = fp8_quantizer(temp, torch.tensor(self.mantissa_bits))
                    if mac_flag == "true":
                        temp = temp.type(self.dt[1])
                #temp = temp ** (1 / 2)
                diff[i] = temp
        else:
            diff = torch.zeros(self.K, dtype=self.dt[1])
            if self.dt == torch.float8:
                vec_step = 4
            else:
                vec_step = 2
            full_chunks = int(centroids.shape[1] / vec_step) # number of full chunks
            processed_elements = vec_step * full_chunks #
            for i in range(centroids.shape[0]):
                temp = torch.zeros(vec_step, dtype=self.dt[1])
                a = torch.zeros(vec_step, dtype=self.dt[1])
                b = torch.zeros(vec_step, dtype=self.dt[1])
                for j in range(0, processed_elements, vec_step):
                    for k in range(vec_step):
                        a[k] = point[j + k]
                        b[k] = centroids[i][j + k]
                    
                    # a = point[j]
                    # b = centroids[i][j]
                    # a1 = point[j + 1]
                    # b1 = centroids[i][j + 1]
                    if mac_flag == "true":
                        a = a.type(torch.float32)
                        b = b.type(torch.float32)
                        temp = temp.type(torch.float32)
                    for k in range(vec_step):
                        te = (a[k] - b[k])
                        # print(f"i:{i}, j:{j}, k:{k}, te:{te}")
                        temp[k] +=  te * te
                        # print(f"i:{i}, j:{j}, k:{k}, temp:{temp[k]}")
                    # temp += (a - b) ** 2
                    # temp1 += (a1 - b1) ** 2
                    if mac_flag == "true":
                        temp = temp.type(self.dt[1])
                
                for j in range(processed_elements, centroids.shape[1]):
                    a[0] = point[j]
                    b[0] = centroids[i][j]
                    if mac_flag == "true":
                        a = a.type(torch.float32)
                        b = b.type(torch.float32)
                        temp = temp.type(torch.float32)
                    temp[0] += (a[0] - b[0]) ** 2
                    if mac_flag == "true":
                        temp = temp.type(self.dt[1])

                #temp = (temp + temp1) ** (1 / 2)
                for j in range(vec_step):
                    diff[i] += temp[j]
                # diff[i] = temp
        return diff

    def mymean(self, inp):
        mean = torch.zeros(inp.shape[1])
        for i in range(inp.shape[1]):
            temp = torch.zeros(1, dtype=self.dt[1])
            for j in range(inp.shape[0]):
                temp += inp[j][i]
            if self.mac_flag == "true":
                        temp = temp.type(torch.float32)
            mean[i] = temp / inp.shape[0]
            if self.mac_flag == "true":
                        mean = mean.type(self.dt[1])
        return mean

    def create_clusters(self, X, centroids):
        # Will contain a list of the points that are associated with that specific cluster
        clusters = [[] for _ in range(self.K)]
        # Loop through each point and check which is the closest cluster
        for point_idx, point in enumerate(X):
            '''closest_centroid = torch.argmin(
                ((torch.sum((point - centroids) ** 2, dim=1)) ** (1 / 2))
            )'''
            temp = self.mysum(point, centroids, self.mac_flag, self.vec_flag)
            if self.mac_flag == "true":
                        temp = temp.type(torch.float32)
            closest_centroid = torch.argmin(temp)
            clusters[closest_centroid].append(point_idx)
        return clusters

    def calculate_new_centroids(self, clusters, X):
        centroids = torch.zeros((self.K, self.num_features), dtype=self.dt[1])
        for idx, cluster in enumerate(clusters):
            # new_centroid = torch.mean(X[cluster], dim=0)
            new_centroid = self.mymean(X[cluster])
            centroids[idx] = new_centroid

        return centroids

    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred

    def fit(self, X):
        centroids = self.initialize_random_centroids(X)
        for it in range(self.max_iterations):
            
            clusters = self.create_clusters(X, centroids)
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)
            diff = centroids - previous_centroids
            if not diff.any():
                print("Termination criterion satisfied")
                break

        # Get label predictions
        print("FINAL Iteration: ", it)
        y_pred = self.predict_cluster(clusters, X)

        return centroids, y_pred

    def predict_single_point(self, point, centroids):
        """
        Predicts the closest cluster for a single data point based on the provided centroids.
        """
        distances = self.mysum(point, centroids, self.mac_flag, self.vec_flag)
        if self.mac_flag == "true":
                        distances = distances.type(torch.float32)
        closest_centroid_idx = torch.argmin(distances).item()
        return closest_centroid_idx


def write_matrix(matrix_to_write, name, len, file_pointer, float_type, test_flag="false"):
    matrix_string = ''
    if 'check' in name or 'ref' in name:
        file_pointer.write("DATA_LOCATION FLOAT %s[] = {" % (name))
    elif 'tests' in name:
        file_pointer.write("DATA_LOCATION FLOAT %s[N_TESTS][N_COORDS] = {" % (name))
    else:
        if test_flag == "true":
            file_pointer.write("DATA_LOCATION FLOAT %s[1][1] = {" % (name))
        else:
            file_pointer.write("DATA_LOCATION FLOAT  %s[N_OBJECTS][N_COORDS] = {" % (name))

    if float_type == torch.float32:
        rem_part = ")"
    elif float_type == torch.float16:
        rem_part = ", dtype=torch.float16)"
    elif float_type == torch.float8:
        rem_part = ", dtype=torch.float8)"
    elif float_type == torch.bfloat16:
        rem_part = ", dtype=torch.bfloat16)"
    

    if 'ref'   in name:
        sz0 = matrix_to_write.shape[0]

        for i in range(sz0):
            matrix_string += str(matrix_to_write[i])
            matrix_string += ', '
        file_pointer.write("%s" % matrix_string)
        file_pointer.write("};\n")
        return
    else:
        sz0, sz1 = matrix_to_write.shape
        if 'check' in name:
            for i in range(sz0):
                for j in range(sz1):
                    if float_type == torch.float32:
                        matrix_string += str(matrix_to_write[i][j].item()).replace('tensor(', '').replace(rem_part, '')
                    else:
                        matrix_string += str(matrix_to_write[i][j].item()).replace('tensor(', '').replace(rem_part, '')
                    matrix_string += ','
            file_pointer.write("%s" % matrix_string)
            file_pointer.write("};\n")
        else:
            for i in range(sz0):
                file_pointer.write("{")
                for j in range(sz1):
                    if float_type == torch.float32:
                        matrix_string = str(matrix_to_write[i][j].item()).replace('tensor(', '').replace(rem_part, '')
                    else:
                        matrix_string = str(matrix_to_write[i][j].item()).replace('tensor(', '').replace(rem_part, '')
                    matrix_string += ','
                    file_pointer.write("%s" % matrix_string)
                file_pointer.write("},\n")

            file_pointer.write("};\n")


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

def mean_squared_error(true, pred):
    squared_error = torch.square(true - pred)
    sum_squared_error = torch.sum(squared_error)
    size = true.size(dim=0) * true.size(dim=1)
    mse_loss = sum_squared_error / size
    return mse_loss


def relative_absolute_error(true, pred):
    true_mean = torch.mean(true)
    squared_error_num = torch.sum(torch.abs(true - pred))
    squared_error_den = torch.sum(torch.abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss

def error_metric(ref, res):

    # calculate manually because metrics doesn't supprt bfloat16
    d = ref - res
    mse_f = torch.mean(d**2)
    mae_f = torch.mean(abs(d))
    rmse_f = torch.sqrt(mse_f)
    r2_f = 1-(torch.sum(d**2)/torch.sum((ref-torch.mean(ref))**2))
    print("MAE:",mae_f.item())
    print("MSE:", mse_f.item())
    print("RMSE:", rmse_f.item())
    print("R-Squared:", r2_f.item())
    rae = relative_absolute_error(ref, res)
    print("RAE is", rae.item())

def get_inital_config():
    # get input size and datatypes
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', default=128)
    parser.add_argument('--features', default=8)
    parser.add_argument('--num_clusters', default=8)
    parser.add_argument('--MAC_flag', default="true")
    parser.add_argument('--vec_flag', default="false")
    parser.add_argument('--float_type', default='FP32')
    parser.add_argument('--test', default="false")
    args = parser.parse_args()

    bits = args.float_type.split(",")
    input_size = int(args.input_size)
    num_clusters = int(args.num_clusters)
    num_features = int(args.features)
    mac_flag = str(args.MAC_flag)
    vec_flag = str(args.vec_flag)
    test_flag = str(args.test)
    if num_features < 1 or num_features > 8:
        sys.exit("ValueError: num_features is not supported for this dataset")

    return input_size, num_features, num_clusters, bits, mac_flag, vec_flag, test_flag


def load_data(input_size, num_features):
    # load data
    filepath = folder_addr + '/dataset-kmeans/data.csv'
    dataset = pd.read_csv(filepath)
    X = dataset.iloc[0:input_size, 0:num_features].values * 1
    input_fp32 = torch.from_numpy(X)
    X_train = input_fp32.type(torch.float32)
    test_size = input_size + 100
    X = dataset.iloc[input_size:test_size, 0:num_features].values  * 1  
    input_fp32 = torch.from_numpy(X)
    X_test = input_fp32.type(torch.float32)
    return X_test, X_train


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


def save_data_into_hfile(x_train, num_clusters, centers, x_test, y_test_custom, test_flag='false'):
    f = open('data_def.h', 'w')
    f.write('\
# define N_CLUSTERS %s\n\
# define N_TESTS %s\n\
# define N_OBJECTS %s\n\
# define N_COORDS %s\n\n' % (num_clusters,x_test.shape[0] ,x_train.shape[0], x_train.shape[1]))
    if test_flag == 'true':
        x_train = torch.zeros((1,1), dtype=x_train.dtype)
    write_matrix(x_train, 'objects', '', f, x_train.dtype, test_flag)
    write_matrix(x_test, 'tests', '', f, x_test.dtype)
    f.close()

    g = open('out_ref.h', 'w')
    g.write('\
#ifndef __CHECKSUM_H__ \n\
#define __CHECKSUM_H__\n\
#include "config.h"\n\
#include "pmsis.h"\n\n')
    write_matrix(centers, 'check', '', g, centers.dtype)
    write_matrix(y_test_custom, 'ref', '', g, y_test_custom.dtype)
    g.write('\
#endif \n')
    g.close()


def main():
    input_size, num_features, num_clusters, bits, mac_flag, vec_flag, test_flag = get_inital_config()

    X_test, X_train = load_data(input_size, num_features)
    y_test_fp32 = np.zeros(X_test.shape[0])
    y_test_custom = np.zeros(X_test.shape[0])
    print("Kmeans is started in FP32 data type")
    Kmeans = KMeansClustering(X_train, num_clusters, dt=[torch.float32,torch.float32], mac_flag=mac_flag, vec_flag=vec_flag, MANTISA_BITS=0)
    cent_fp32, _ = Kmeans.fit(X_train)
    for i in range(X_test.shape[0]):
        y_test_fp32[i] = Kmeans.predict_single_point( X_test[i], cent_fp32)
           
    # set the data types based on the parser input
    datatypes = select_dtypes(bits, 2)
    # change the datatypes

    fig, axs = plt.subplots(7, 1, figsize=(6, 12))

    data = X_train.flatten().tolist()
    std = torch.std(X_train)
    mean = torch.mean(X_train)
    print("STD of X_tain is:", std)
    print("Mean of X_tain is:", mean)
        # Create a histogram for the distribution
    axs[0].hist(data, bins=30)
    axs[0].set_title(f"REF")
    axs[0].set_xlabel("Values")
    axs[0].set_ylabel("Frequency")
    
    if datatypes[0] == "FP8_CUSTOM":
        for MANTISA_BITS in range(1,7):
          x_train = matrix_init_custom(X_train, MANTISA_BITS)
          datatypes[1] = torch.float32
          print("Kmeans is started in the desired data type", datatypes[0])
          Kmeans = KMeansClustering(x_train, num_clusters, dt=datatypes, mac_flag=mac_flag, vec_flag=vec_flag, MANTISA_BITS=MANTISA_BITS)
          cent_des, _ = Kmeans.fit(x_train)
          # print("Centroids in the desired data-type:", cent_type.numpy())
          print("MANTISA_BITS",MANTISA_BITS)
          axs[MANTISA_BITS].hist(data, bins=30)
          axs[MANTISA_BITS].set_title(f"MANTISA_BITS = {MANTISA_BITS}")
          axs[MANTISA_BITS].set_xlabel("Values")
          axs[MANTISA_BITS].set_ylabel("Frequency")
          error_metric(cent_fp32,cent_des)
          
          '''#TEST PHASE
          x_test = matrix_init_custom(X_train, MANTISA_BITS)
          for i in range(x_test.shape[0]):
              y_test_custom[i] = Kmeans.predict_single_point( x_test[i], cent_des)    
          
      
          
          C_false = sum(x != y for x, y in zip(y_test_custom, y_test_fp32))
          print("Number of differences:", C_false)
          print("Classification error:", (C_false / y_test_fp32.shape[0]) * 100)'''
      
          #save_data_into_hfile(x_train, num_clusters, cent_des, x_test, y_test_custom, test_flag)
          print("############################## Done! ###################################")
        plt.show()


    else:
        x_train = matrix_init(X_train, dt=datatypes[0])
        print("Kmeans is started in the desired data type", datatypes[0])
        Kmeans = KMeansClustering(x_train, num_clusters, dt=datatypes, mac_flag=mac_flag, vec_flag=vec_flag, MANTISA_BITS=0)
        cent_des, _ = Kmeans.fit(x_train)
        # print("Centroids in the desired data-type:", cent_type.numpy())
    
        error_metric(cent_fp32,cent_des)
    
        '''# test the model
 
        x_test = matrix_init(X_test, dt=datatypes[0])
        for i in range(x_test.shape[0]):
            y_test_custom[i] = Kmeans.predict_single_point( x_test[i], cent_des)    
        # y_test_custom = Kmeans.predict_single_point( x_test[0], cent_des)
    
        
        C_false = sum(x != y for x, y in zip(y_test_custom, y_test_fp32))
        print("Number of differences:", C_false)
        print("Classification error:", (C_false / y_test_fp32.shape[0]) * 100)'''
    
        #save_data_into_hfile(x_train, num_clusters, cent_des, x_test, y_test_custom, test_flag)
        print("############################## Done! ###################################")
    


if __name__ == "__main__":
    main()
    pass