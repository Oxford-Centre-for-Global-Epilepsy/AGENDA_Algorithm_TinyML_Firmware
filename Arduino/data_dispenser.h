#pragma once
#include <Arduino.h>
#include <SD.h>

// A class for reading and quantizing CSV input data into a fixed-format buffer for inference
class InputDispenser {
public:
    // Constructor
    // - filepath: path to the CSV file on SD card
    // - numColumns: number of expected values per row (i.e., input features)
    // - scale & zeroPoint: parameters for TFLM-style quantization
    // - floatBuffer / uint8Buffer: pre-allocated target buffers for float or uint8
    InputDispenser(const char* filepath, size_t numColumns, size_t numRows, 
                   float scale = 1.0f, int zeroPoint = 0,
                   float* floatBuffer = NULL, uint8_t* uint8Buffer = NULL);

    ~InputDispenser();

    // Opens the file and validates header
    bool begin();

    // SD File operations for the dispenser
    bool isEOF();
    void rewind();
    bool streamNext();

    // Set up the data dispensing functionality
    void setOutputBuffer(float* floatPtr, uint8_t* uint8Ptr);
    void setQuantisation(float quantScale, int quantZeroPoint);


private:
    const char* path;       // path to the CSV file
    size_t columns;         // number of expected values per row
    size_t rows;            // number of expected rows
    float* lineBuffer;      // Internal reusable buffer to hold parsed float values for each row

    float scale;            // quantization scale
    int zeroPoint;          // quantization zero point

    float* floatBuffer;     // pointer to float32 buffer
    uint8_t* uint8Buffer;     // pointer to uint8 quantized buffer

    int currentRowIndex;    // current row being written

    File file;              // SD file handle

    // Parses a CSV line into the internal float lineBuffer
    bool parseCsvLine(const String& line, float* outBuf) const;

    // Quantizes internal line buffer to uint8 using TFLM formula
    void quantise(uint8_t* out) const;
};
