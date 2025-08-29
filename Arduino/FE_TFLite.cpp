#include "FE_TFLite.h"
using namespace tflite;

bool FE_TFLite::begin(const unsigned char* model_data,
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

// This function gets the quantisation parameter of input tensor
void FE_TFLite::getInputQuantisation(float& scale, int& zeroPoint) {
  TfLiteTensor* t = interp_->input(0);   // get first input tensor
  scale = t->params.scale;         // quantization scale
  zeroPoint = t->params.zero_point;         // zero point
}

// This function gets the pointer to the input tensor
TfLiteTensor* FE_TFLite::input(int idx)  { return interp_ ? interp_->input(idx)  : nullptr; }

// /////////////////////
// ///// INFERENCE /////
// /////////////////////

bool FE_TFLite::invoke() {
  if (!interp_) return false;
  return interp_->Invoke() == kTfLiteOk;
}

// //////////////////////////
// ///// OUTPUT METHODS /////
// //////////////////////////

// This function gets the quantisation parameter of output tensor
void FE_TFLite::getOutputQuantisation(float& scale, int& zeroPoint) {
  TfLiteTensor* t = interp_->output(0);   // get first input tensor
  scale = t->params.scale;         // quantization scale
  zeroPoint = t->params.zero_point;         // zero point
}

// This function gets the pointer to the output tensor
TfLiteTensor* FE_TFLite::output(int idx) { return interp_ ? interp_->output(idx) : nullptr; }

// This function directly get the values in the output tensor
void FE_TFLite::getOutput(float *out_buffer, int out_len) {
  TfLiteTensor* t = interp_->output(0);
  float scale = t->params.scale;
  int zp = t->params.zero_point;
  const uint8_t* src = t->data.uint8;

  int n = min(out_len, static_cast<int>(t->bytes));
  for (int i = 0; i < n; ++i) {
    out_buffer[i] = scale * (static_cast<int>(src[i]) - zp);
  }
}
