#ifndef __GPU_IMPL_H__
#define __GPU_IMPL_H__

#include "math.h"

// dataset information
// const int NUM_FEATURES  = 1024;
// const int NUM_SAMPLES   = 5000;
// const int NUM_TRAINING  = 4500;
// const int NUM_TESTING   = 500;
// const int STEP_SIZE     = 60000;
// const int NUM_EPOCHS    = 5;
const int DATA_SET_SIZE = NUM_FEATURES * NUM_SAMPLES;

#define VFTYPE_WIDTH  64
#define VDTYPE_WIDTH  64

// features / parameters
typedef float FeatureType;
typedef int64_t VectorFeatureType;

// training data
typedef float DataType; // may need to change this to 16-bit float
typedef int64_t VectorDataType;

// label
typedef int8_t LabelType;
typedef int32_t VectorLabelType;

#define PARA_FACTOR 32

FeatureType dotProduct(FeatureType param[NUM_FEATURES],
                       DataType    feature[NUM_FEATURES]);

FeatureType Sigmoid(FeatureType exponent);

void computeGradient(
    FeatureType grad[NUM_FEATURES],
    DataType    feature[NUM_FEATURES],
    FeatureType scale);

void updateParameter(
    FeatureType param[NUM_FEATURES],
    FeatureType grad[NUM_FEATURES],
    FeatureType step_size);

#endif
