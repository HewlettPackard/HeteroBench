#include <iostream>

// dataset information
// #define NUM_TRAINING 18000
// #define CLASS_SIZE 1800
// #define NUM_TEST 2000
// #define DIGIT_WIDTH 4

// typedefs
typedef unsigned long long DigitType;
typedef unsigned char      LabelType;

// parameters
// #define K_CONST 3
// #define PARA_FACTOR 40

// types and constants used in the functions below
const unsigned long long m1  = 0x5555555555555555; //binary: 0101...
const unsigned long long m2  = 0x3333333333333333; //binary: 00110011..
const unsigned long long m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...

void popcount(DigitType diff, int* popcount_result);
void update_knn( const DigitType* training_set, const DigitType* test_set, int dists[K_CONST], int labels[K_CONST]);
void knn_vote(int lables[K_CONST], LabelType* max_label);
