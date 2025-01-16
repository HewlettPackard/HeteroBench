"""
(C) Copyright [2024] Hewlett Packard Enterprise Development LP

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the Software),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import sys
import time

# cpu_impl.h
if len(sys.argv) == 9:
    path_to_data = sys.argv[1]
    iterations = int(sys.argv[2])
    num_features = int(sys.argv[3])
    num_samples = int(sys.argv[4])
    num_training = int(sys.argv[5])
    num_testing = int(sys.argv[6])
    step_size = int(sys.argv[7])
    num_epochs = int(sys.argv[8])
else:
    print("Did not provide command line arguments correctly. Using default values")
    print("Usage: python main.py <iterations> <num_features> <num_samples> <num_training> <num_testing> <step_size> <num_epochs>")
    iterations = 20
    num_features = 10000
    num_samples = 5000
    num_training = 4500
    num_testing = 500
    step_size = 60000
    num_epochs = 5

data_set_size = num_features * num_samples

FeatureType = np.float32
DataType = np.float32
LabelType = np.int8

# dotProduct.cpp
def dotProduct(param, feature):
    result = 0
    for i in range(num_features):
        result += param[i] * feature[i]
    return result

# Sigmoid.cpp
def Sigmoid(exponent):
    return 1.0 / (1.0 + np.exp(-exponent))

# computeGradient.cpp
def computeGradient(grad, feature, scale):
    for i in range(num_features):
        grad[i] = scale * feature[i]

# updateParameter.cpp
def updateParameter(param, grad, step_size):
    for i in range(num_features):
        param[i] += step_size * grad[i]

# main.cpp
class DataSet:
    def __init__(self, data_points, labels, parameter_vector, num_data_points, num_features):
        self.data_points = data_points
        self.labels = labels
        self.parameter_vector = parameter_vector
        self.num_data_points = num_data_points
        self.num_features = num_features

def dotProduct_host(param_vector, data_point_i, num_features):
    result = 0.0
    for i in range(num_features):
        result += param_vector[i] * data_point_i[i]
    return result

def getPrediction(parameter_vector, data_point_i, num_features, threshold=0):
    parameter_vector_dot_x_i = dotProduct_host(parameter_vector, data_point_i, num_features)
    return 1 if parameter_vector_dot_x_i > threshold else 0

def computeErrorRate(
    data_set, 
    cumulative_true_positive_rate, 
    cumulative_false_positive_rate, 
    cumulative_error
    ):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(data_set.num_data_points):
        prediction = getPrediction(
            data_set.parameter_vector, 
            data_set.data_points[i * data_set.num_features:(i + 1) * data_set.num_features], 
            data_set.num_features)
        if prediction != data_set.labels[i]:
            if prediction == 1:
                false_positives += 1
            else:
                false_negatives += 1
        else:
            if prediction == 1:
                true_positives += 1
            else:
                true_negatives += 1

    error_rate = (false_positives + false_negatives) / data_set.num_data_points
    cumulative_true_positive_rate += true_positives / (true_positives + false_negatives)
    cumulative_false_positive_rate += false_positives / (false_positives + true_negatives)
    cumulative_error += error_rate

    return error_rate

def check_results(param_vector, data_points, labels):
    with open("./benchmarks/spam_filter/python/output.txt", "w") as ofile:
        ofile.write(f"\nmain parameter vector: \n")
        for i in range(30):
            ofile.write(f"m[{i}]: {param_vector[i]} | ")
        ofile.write("\n")
            
        training_tpr = 0.0
        training_fpr = 0.0
        training_error = 0.0
        testing_tpr = 0.0
        testing_fpr = 0.0
        testing_error = 0.0

        training_set = DataSet(
            data_points,
            labels,
            param_vector,
            num_training,
            num_features
        )
        computeErrorRate(training_set, training_tpr, training_fpr, training_error)

        testing_set = DataSet(
            data_points[num_features * num_training:],
            labels[num_training:],
            param_vector,
            num_testing,
            num_features
        )
        computeErrorRate(testing_set, testing_tpr, testing_fpr, testing_error)

        training_tpr *= 100.0
        training_fpr *= 100.0
        training_error *= 100.0
        testing_tpr *= 100.0
        testing_fpr *= 100.0
        testing_error *= 100.0

        ofile.write(f"Training TPR: {training_tpr}\n")
        ofile.write(f"Training FPR: {training_fpr}\n")
        ofile.write(f"Training Error: {training_error}\n")
        ofile.write(f"Testing TPR: {testing_tpr}\n")
        ofile.write(f"Testing FPR: {testing_fpr}\n")
        ofile.write(f"Testing Error: {testing_error}\n")

def SgdLR_sw(data, label, theta):
    gradient = np.zeros(num_features, dtype=FeatureType)

    # 1 warm up iteration
    print("Running 1 warm up iteration ...")
    for training_id in range(num_training):
        dot = dotProduct(theta, data[num_features * training_id:num_features * (training_id + 1)])
        prob = Sigmoid(dot)
        computeGradient(gradient, data[num_features * training_id:num_features * (training_id + 1)], (prob - label[training_id]))
        updateParameter(theta, gradient, -step_size)
    print("Done")

    # check results
    print("Checking results ...")
    check_results(theta, data, label)
    print("Done")

    print(f"Running {iterations} iterations ...")
    start_whole_time = time.time()

    dotProduct_time = 0
    Sigmoid_time = 0
    computeGradient_time = 0
    updateParameter_time = 0

    for epoch in range(iterations):
        for training_id in range(num_training):
            start_iteration_time = time.time()
            dot = dotProduct(theta, data[num_features * training_id:num_features * (training_id + 1)])
            dotProduct_time += time.time() - start_iteration_time

            start_iteration_time = time.time()
            prob = Sigmoid(dot)
            Sigmoid_time += time.time() - start_iteration_time

            start_iteration_time = time.time()
            computeGradient(gradient, data[num_features * training_id:num_features * (training_id + 1)], (prob - label[training_id]))
            computeGradient_time += time.time() - start_iteration_time

            start_iteration_time = time.time()
            updateParameter(theta, gradient, -step_size)
            updateParameter_time += time.time() - start_iteration_time

    run_whole_time = time.time() - start_whole_time
    print(f"1 warm up iteration and {iterations} iterations")
    print(f"Single iteration time: {run_whole_time / iterations * 1000:.2f} ms")
    print(f"dotProduct time: {dotProduct_time / iterations * 1000:.2f} ms")
    print(f"Sigmoid time: {Sigmoid_time / iterations * 1000:.2f} ms")
    print(f"computeGradient time: {computeGradient_time / iterations * 1000:.2f} ms")
    print(f"updateParameter time: {updateParameter_time / iterations * 1000:.2f} ms")

def main():
    print("=======================================")
    print("Running spam_filter benchmark Python")
    print("=======================================")

    data_points = np.zeros(data_set_size, dtype=DataType)
    labels = np.zeros(num_samples, dtype=LabelType)
    param_vector = np.zeros(num_features, dtype=FeatureType)

    data_points_filepath = f"{path_to_data}/shuffledfeats.dat"
    labels_filepath = f"{path_to_data}/shuffledlabels.dat"

    with open(data_points_filepath, "r") as data_file:
        for i in range(data_set_size):
            line = data_file.readline().strip()
            if line:  # Check if the line is not empty
                data_points[i] = float(line)

    with open(labels_filepath, "r") as label_file:
        for i in range(num_samples):
            line = label_file.readline().strip()
            if line:  # Check if the line is not empty
                labels[i] = int(line)


    SgdLR_sw(data_points, labels, param_vector)

if __name__ == "__main__":
    main()