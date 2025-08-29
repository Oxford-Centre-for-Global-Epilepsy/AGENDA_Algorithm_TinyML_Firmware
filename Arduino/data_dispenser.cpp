#include "data_dispenser.h"

// Constructor initializes members and allocates internal line buffer
InputDispenser::InputDispenser(const char* filepath, size_t numColumns, size_t numRows,
                               float scale, int zeroPoint,
                               float* floatBuffer, uint8_t* uint8Buffer)
    : path(filepath), columns(numColumns), rows(numRows), scale(scale), zeroPoint(zeroPoint),
      floatBuffer(floatBuffer), uint8Buffer(uint8Buffer),
      currentRowIndex(0) {
    lineBuffer = new float[columns];
}

// Destructor releases allocated memory
InputDispenser::~InputDispenser() {
    delete[] lineBuffer;
}

// Opens the CSV file from SD card and checks header for correct column count
bool InputDispenser::begin() {
    file = SD.open(path);
    if (!file) {
        Serial.print("[InputDispenser] Failed to open file: ");
        Serial.println(path);
        return false;
    }

    // Read header line and count commas to validate column count
    String header = file.readStringUntil('\n');
    int commaCount = 0;
    for (char c : header) {
        if (c == ',') commaCount++;
    }

    if (static_cast<size_t>(commaCount + 1) != columns) {
        Serial.println("[InputDispenser] Column count mismatch in header");
        return false;
    }

    return true;
}

// Checks whether end-of-file has been reached or file is invalid
bool InputDispenser::isEOF() {
    return !file || !file.available();
}

// Rewinds the file to the start and skips the header line
void InputDispenser::rewind() {
    if (file) {
        file.close();
        file = SD.open(path);
        if (file) {
            file.readStringUntil('\n'); // Skip header
            currentRowIndex = 0;
        }
    }
}

// Streams the next row of CSV into either float or quantized buffer
bool InputDispenser::streamNext() {
    if (!file || !file.available()) return false;

    String line = file.readStringUntil('\n');
    if (!parseCsvLine(line, lineBuffer)) {
        Serial.println("[InputDispenser] Failed to parse CSV line");
        return false;
    }

    if (static_cast<size_t>(currentRowIndex) >= rows) {
        currentRowIndex = 0;
    }

    bool output_flag = false;

    // For each channel in this time sample
    for (size_t ch = 0; ch < columns; ++ch) {
        size_t idx = ch * rows + currentRowIndex; // channel-major indexing

        if (floatBuffer) {
            floatBuffer[idx] = lineBuffer[ch];
            output_flag = true;
        }

        if (uint8Buffer) {
            int32_t q = static_cast<int32_t>(roundf(lineBuffer[ch] / scale) + zeroPoint);
            q = max(0, min(255, q));  // Clamp to uint8 range
            uint8Buffer[idx] = static_cast<uint8_t>(q);
            output_flag = true;
        }
    }

    if (!output_flag) {
        Serial.println("[InputDispenser] No output buffer set");
        return false;
    }

    currentRowIndex++;
    return true;
}

// Updates the output buffer pointers and resets row index
void InputDispenser::setOutputBuffer(float* floatPtr, uint8_t* uint8Ptr) {
    floatBuffer = floatPtr;
    uint8Buffer = uint8Ptr;
    currentRowIndex = 0;
}

// Set the Quantisation Parameters
void InputDispenser::setQuantisation(float quantScale, int quantZeroPoint) {
    scale = quantScale;
    zeroPoint = quantZeroPoint;
}

// Parses a line of CSV-formatted floats into a buffer
bool InputDispenser::parseCsvLine(const String& line, float* outBuf) const {
    int lastIdx = 0;
    for (size_t i = 0; i < columns; ++i) {
        int idx = line.indexOf(',', lastIdx);
        String token = (idx == -1) ? line.substring(lastIdx) : line.substring(lastIdx, idx);
        outBuf[i] = token.toFloat();
        if (idx == -1 && i < columns - 1) return false; // premature end
        lastIdx = idx + 1;
    }
    return true;
}

// Quantizes internal lineBuffer using scale and zero point
void InputDispenser::quantise(uint8_t* out) const {
    for (size_t i = 0; i < columns; ++i) {
        int32_t q = static_cast<int32_t>(roundf(lineBuffer[i] / scale) + zeroPoint);
        q = max(0, min(255, q));  // Clamp to uint8 range
        out[i] = static_cast<uint8_t>(q);
    }
}
