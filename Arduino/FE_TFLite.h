#ifndef FE_TFLITE_H
#define FE_TFLITE_H

#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

class FE_TFLite {
public:
  bool begin(const unsigned char* model_data,
             tflite::MicroOpResolver& resolver,
             uint8_t* tensor_arena, size_t tensor_arena_size);

  // Functions for filling input buffer
  void getInputQuantisation(float& scale, int& zeroPoint);
  TfLiteTensor* input(int idx = 0);

  // Functions for invoking inference
  bool invoke();

  // Functions for fetching from output buffer
  void getOutputQuantisation(float& scale, int& zeroPoint);
  TfLiteTensor* output(int idx = 0);
  void getOutput(float* out_buffer, int out_len);

private:
  const tflite::Model* model_ = nullptr;
  tflite::MicroOpResolver* resolver_ = nullptr;   // non-owning
  uint8_t* arena_ = nullptr;
  size_t arena_size_ = 0;

  alignas(tflite::MicroInterpreter) uint8_t interp_storage_[sizeof(tflite::MicroInterpreter)];
  tflite::MicroInterpreter* interp_ = nullptr;
};

#endif
