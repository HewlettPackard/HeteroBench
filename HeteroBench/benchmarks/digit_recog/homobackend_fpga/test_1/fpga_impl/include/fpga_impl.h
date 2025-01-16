#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// dataset information
// #define NUM_TRAINING 18000
// #define CLASS_SIZE 1800
// #define NUM_TEST 2000
// #define DIGIT_WIDTH 4

// typedefs
typedef ap_uint<256>    WholeDigitType;
typedef unsigned char   LabelType;

// parameters
// #define K_CONST 3
// #define PARA_FACTOR 40

// types and constants used in the functions below
const unsigned long long m1  = 0x5555555555555555; //binary: 0101...
const unsigned long long m2  = 0x3333333333333333; //binary: 00110011..
const unsigned long long m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...

// void popcount(DigitType diff, int* popcount_result);
void DigitRec_hw(WholeDigitType global_training_set[NUM_TRAINING], WholeDigitType global_test_set[NUM_TEST], LabelType global_results[NUM_TEST]);
