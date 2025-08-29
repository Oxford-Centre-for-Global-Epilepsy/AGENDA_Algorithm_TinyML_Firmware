#include "avg_pooling.h"

AvgPool::AvgPool(int dim)
    : dim_(dim), count_(0)
{
    sum_ = new float[dim_];
    reset();
}

// Reset all the feature vector to zero
void AvgPool::reset() {
  for (int i=0; i<dim_; ++i) {
    sum_[i] = 0;
  }
  count_ = 0;
}

// Add the feature vector to aveage pooling summation
void AvgPool::add(const float* vec) {
  for (int i=0; i<dim_; ++i) {
    sum_[i] += vec[i];
  }
  ++count_;   // increment the counter
}

// Get the averaged feature vector
void AvgPool::finalize(float* out) {
    float inv = 1.0f / static_cast<float>(count_);
    for (int i = 0; i < dim_; i++) {
        out[i] = sum_[i] * inv;
    }
}