/*
 * (C) Copyright [2024] Hewlett Packard Enterprise Development LP
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the Software),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
 
#include "acc_impl.h"

using namespace std;

#pragma acc routine
void popcount(DigitType diff, int* popcount_result)
{
    diff -= (diff >> 1) & m1;             //put count of each 2 bits into those 2 bits
    diff = (diff & m2) + ((diff >> 2) & m2); //put count of each 4 bits into those 4 bits 
    diff = (diff + (diff >> 4)) & m4;        //put count of each 8 bits into those 8 bits 
    diff += diff >>  8;  //put count of each 16 bits into their lowest 8 bits
    diff += diff >> 16;  //put count of each 32 bits into their lowest 8 bits
    diff += diff >> 32;  //put count of each 64 bits into their lowest 8 bits
    *popcount_result = diff & 0x7f;
}