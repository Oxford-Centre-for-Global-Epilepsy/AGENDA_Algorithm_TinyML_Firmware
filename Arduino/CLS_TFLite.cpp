#include "CLS_TFLite.h"
using namespace tflite;

bool CLS_TFLite::begin(const unsigned char* model_data,
                       MicroOpResolver& resolver,
                       uint8_t* tensor_arena, size_t tensor_arena_size) {
  resolver_ = &resolver;
  arena_ = tensor_arena;
  arena_size_ = tensor_arena_size;

  model_ = ::tflite::GetModel(model_data);
  if (!model_) return false;
  if (model_->version() != TFLITE_SCHEMA_VERSION) return false;

  new (&interp_storage_) MicroInterpreter(model_, *resolver_, arena_, arena_size_);
  interp_ = reinterpret_cast<MicroInterpreter*>(&interp_storage_);

  if (interp_->AllocateTensors() != kTfLiteOk) return false;
  return true;
}

// /////////////////////////
// ///// INPUT METHODS /////
// /////////////////////////

// Get pointer to input tensor (float data expected)
TfLiteTensor* CLS_TFLite::input(int idx) {
  return interp_ ? interp_->input(idx) : nullptr;
}

// /////////////////////
// ///// INFERENCE /////
// /////////////////////

bool CLS_TFLite::invoke() {
  if (!interp_) return false;
  return interp_->Invoke() == kTfLiteOk;
}

// //////////////////////////
// ///// OUTPUT METHODS /////
// //////////////////////////

// Get pointer to output tensor (float data expected)
TfLiteTensor* CLS_TFLite::output(int idx) {
  return interp_ ? interp_->output(idx) : nullptr;
}

// Copy float output tensor to user buffer
void CLS_TFLite::getOutput(float* out_buffer, int out_len) {
  TfLiteTensor* t = interp_->output(0);
  const float* src = t->data.f;
  int n = min(out_len, static_cast<int>(t->bytes / sizeof(float)));
  for (int i = 0; i < n; ++i) {
    out_buffer[i] = src[i];
  }
}
