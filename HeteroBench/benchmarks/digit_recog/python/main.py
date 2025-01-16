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
from numba import jit

# cpu_impl.h
if len(sys.argv) == 8:
    input_path = sys.argv[1]
    iterations = int(sys.argv[2])
    num_training = int(sys.argv[3])
    class_size = int(sys.argv[4])
    digit_width = int(sys.argv[5])
    k_const = int(sys.argv[6])
    para_factor = int(sys.argv[7])
else:
    print("Did not provide command line arguments correctly. Using default values")
    print("Usage: python main.py <iterations> <num_training> <class_size> <digit_width> <k_const> <para_factor>")
    iterations = 2000
    num_training = 18000
    class_size = 1800
    digit_width = 4
    k_const = 3
    para_factor = 40

DigitType = np.uint64
LabelType = np.uint8

m1 = np.uint64(0x5555555555555555)
m2 = np.uint64(0x3333333333333333)
m4 = np.uint64(0x0f0f0f0f0f0f0f0f)

num_test = iterations

# popcount.cpp
#@jit(nopython=True, cache=True)
def popcount(diff): # Need to make sure all the data type is same
    diff -= (diff >> DigitType(1)) & m1
    diff = (diff & m2) + ((diff >> DigitType(2)) & m2)
    diff = (diff + (diff >> DigitType(4))) & m4
    diff += diff >> DigitType(8)
    diff += diff >> DigitType(16)
    diff += diff >> DigitType(32)
    return diff & DigitType(0x7f)

# update_knn.cpp
#@jit(nopython=True, cache=True)
def update(training_set, test_set, dists, labels, label):
    dist = 0
    for i in range(digit_width):
        diff = DigitType(test_set[i] ^ training_set[i])
        dist += popcount(diff)

    max_dist = 0
    max_dist_id = k_const + 1

    for k in range(k_const):
        if dists[k] > max_dist:
            max_dist = dists[k]
            max_dist_id = k
    if dist < max_dist:
        dists[max_dist_id] = dist
        labels[max_dist_id] = label

#@jit(nopython=True, cache=True)
def update_knn(training_set, test_set, dists, labels):
    for i in range(num_training):
        label = i / class_size
        # update(training_set[i * digit_width], test_set, dists, labels, label)
        # In python, the shape is already 2D, so no need to multiply by digit_width.
        update(training_set[i], test_set, dists, labels, label)

# knn_vote.cpp
#@jit(nopython=True, cache=True)
def knn_vote(labels):
    max_vote = 0
    votes = np.zeros(10, dtype=np.int32)
    for i in range(k_const):
        index = labels[i]
        votes[index] += 1
    for i in range(10):
        current_votes = votes[i]
        if current_votes > max_vote:
            max_vote = current_votes
            max_label = i
    return max_label

# main.cpp
def check_results(result, expected, cnt):
    correct_cnt = 0

    with open("./benchmarks/digit_recog/python/outputs.txt", "w") as ofile:
        for i in range(cnt):
            if result[i] != expected[i]:
                ofile.write(f"Test {i}: expected = {int(expected[i])}, result = {int(result[i])}\n")
            else:
                correct_cnt += 1

        ofile.write(f"\n\t {correct_cnt} / {cnt} correct!\n")

def load_data():
    testing_data = np.fromfile(f"{input_path}/test_set.dat", dtype=DigitType)
    
    # expected = np.fromfile(f"{input_path}/expected.dat", dtype=LabelType)
    # print(f'exptected: {expected}')

    # Need to rewrite the expected data loading as the expected data is not correct if directly loaded from file in uint8.
    # Need to load the data as string and then convert to uint8
    expected_data_list = []
    with open(f"{input_path}/expected.dat", 'r') as file:
        for line in file:
            part = line.strip().strip(',')
            if part.isdigit() and 0 <= int(part) <= 9:
                expected_data_list.append(np.uint8(part))
            else:
                print(f"Warning: Unexpected value '{part}' in file {input_path}/expected.dat, line {line.strip()}")

    expected = np.array(expected_data_list, dtype=np.uint8)
    
    training_data_0 = np.fromfile(f"{input_path}/training_set_0.dat", dtype=DigitType)
    training_data_1 = np.fromfile(f"{input_path}/training_set_1.dat", dtype=DigitType)
    training_data_2 = np.fromfile(f"{input_path}/training_set_2.dat", dtype=DigitType)
    training_data_3 = np.fromfile(f"{input_path}/training_set_3.dat", dtype=DigitType)
    training_data_4 = np.fromfile(f"{input_path}/training_set_4.dat", dtype=DigitType)
    training_data_5 = np.fromfile(f"{input_path}/training_set_5.dat", dtype=DigitType)
    training_data_6 = np.fromfile(f"{input_path}/training_set_6.dat", dtype=DigitType)
    training_data_7 = np.fromfile(f"{input_path}/training_set_7.dat", dtype=DigitType)
    training_data_8 = np.fromfile(f"{input_path}/training_set_8.dat", dtype=DigitType)
    training_data_9 = np.fromfile(f"{input_path}/training_set_9.dat", dtype=DigitType)

    training_data = np.concatenate((training_data_0, training_data_1, training_data_2, \
                                    training_data_3, training_data_4, training_data_5, \
                                    training_data_6, training_data_7, training_data_8, training_data_9))
    
    # reshape testing data to (-1, 4), if it cannot, padding with zeros
    if testing_data.size < num_test * digit_width:
        testing_data = np.pad(testing_data, (0, num_test * digit_width - testing_data.size), mode='constant')
    else:
        testing_data = testing_data[:num_test * digit_width]
    testing_data = testing_data.reshape(num_test, digit_width)

    # reshape training data to (-1, 4), if it cannot, padding with zeros
    if training_data.size < num_training * digit_width:
        training_data = np.pad(training_data, (0, num_training * digit_width - training_data.size), mode='constant')
    else:
        training_data = training_data[:num_training * digit_width]
    training_data = training_data.reshape(num_training, digit_width)

    return testing_data, expected, training_data

def DigitRec_sw(training_set, test_set, results):
    # nearest neighbor set
    dists = np.empty(k_const, dtype=int)
    labels = np.empty(k_const, dtype=int)

    print(f"Running {num_test} iterations ...")

    start_whole_time = time.time()

    start_update_knn_time = 0
    start_knn_vote_time = 0

    for t in range(num_test):
        for i in range(k_const):
            dists[i] = 256
            labels[i] = 0

        start_iteration_time = time.time()
        update_knn(training_set, test_set[t], dists, labels)
        start_update_knn_time += time.time() - start_iteration_time

        start_iteration_time = time.time()
        max_label = knn_vote(labels)
        results[t] = max_label
        start_knn_vote_time += time.time() - start_iteration_time

    print("Done")

    run_whole_time = time.time() - start_whole_time
    print(f"Total {num_test} iterations")
    print(f"Single iteration time: {(run_whole_time / num_test) * 1000:.2f} ms")
    print(f"Update knn time: {(start_update_knn_time / num_test) * 1000:.2f} ms")
    print(f"Knn vote time: {(start_knn_vote_time / num_test) * 1000:.2f} ms")

def main():
    print("=======================================")
    print("Running digit_recog benchmark Python")
    print("=======================================")

    testing_data, expected, training_data = load_data()

    result = np.zeros(num_test, dtype=LabelType)

    DigitRec_sw(training_data, testing_data, result)

    print("Checking results ...")
    check_results(result, expected, num_test)
    print("Done")

if __name__ == "__main__":
    main()