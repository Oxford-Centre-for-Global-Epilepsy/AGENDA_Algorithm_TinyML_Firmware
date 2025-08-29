#include <Arduino.h>
#include <SD.h>
#include "data_dispenser.h"

#include "FE_TFLite.h"
#include "CLS_TFLite.h"
#include "avg_pooling.h"

#include "eegnet.h"
#include "classifier.h"

// ===== Preparing Data Dispenser =====
constexpr size_t NUM_CHANNELS = 16;
constexpr size_t NUM_ROWS = 128;
float inputTensor[NUM_ROWS * NUM_CHANNELS];  // Flat buffer (always in channel-first format by TFLite Default)

constexpr int SD_CS_PIN = BUILTIN_SDCARD;

float dispenserQuantScale = 1.0;            // Setup the input quantisation scale (need reset)
int dispenserQuantZeroPoint = 128;          // Setup the output quantisation scale (need reset)

InputDispenser dispenser("/G0001.csv", NUM_CHANNELS, NUM_ROWS, dispenserQuantScale, dispenserQuantZeroPoint, inputTensor, nullptr);

// ===== Preparing FE Model =====
tflite::MicroMutableOpResolver<9> resolver;
void setup_resolver() {
  resolver.AddQuantize();
  resolver.AddConv2D();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddDepthwiseConv2D();
  resolver.AddHardSwish();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();  // for MLP
}

constexpr int feTensorArenaSize = 80*1024;
alignas(16) uint8_t fe_tensor_arena[feTensorArenaSize];
size_t NUM_OUTPUTS_FE = 16;

FE_TFLite EEGNet;

// ===== Prepare Average Pooling =====
AvgPool pooler(NUM_OUTPUTS_FE);

// ===== Prepare CLS Model =====
constexpr int clsTensorArenaSize = 16*1024;
alignas(16) uint8_t cls_tensor_arena[clsTensorArenaSize];
size_t NUM_OUTPUTS_CLS = 2;

CLS_TFLite Classifier;

// ===== Prepare Misc Helpers =====
size_t frame_counter = 0;
size_t frame_target = 5000;

bool file_depletion = false;

// ////////////////////////////////////
// ///// Performance Benchmarking /////
// ////////////////////////////////////

const char* dataDir = "/test_data_0820";  // can be changed depending on which fold

uint8_t parseLabelFromName(const char* fname) {
    return (fname[0] == 'N') ? 0 : 1;
}

// Function that fetches all the validation file names
constexpr size_t MAX_FILES = 400;
constexpr size_t MAX_NAME_LEN = 13;  // 12 chars + '\0'

char fileList[MAX_FILES][MAX_NAME_LEN];
size_t numFiles = 0;

size_t collectFileList(const char* folder) {
    numFiles = 0;
    File root = SD.open(folder);

    while (true) {
        File entry = root.openNextFile();
        if (!entry) break;
        if (!entry.isDirectory() && numFiles < MAX_FILES) {
            strncpy(fileList[numFiles], entry.name(), MAX_NAME_LEN - 1);
            fileList[numFiles][MAX_NAME_LEN - 1] = '\0';
            numFiles++;
        }
        entry.close();
    }

    return numFiles;
}

constexpr int NUM_CLASSES = 2;
int confusion[NUM_CLASSES][NUM_CLASSES];

void resetConfusion() {
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            confusion[i][j] = 0;
        }
    }
}

void reportConfusion() {
    Serial.println("Confusion Matrix:");
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            Serial.printf("%4d ", confusion[i][j]);
        }
        Serial.println();
    }

    int correct = 0, total = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_CLASSES; j++) {
            total += confusion[i][j];
            if (i == j) correct += confusion[i][j];
        }
    }

    float acc = (total > 0) ? (100.0f * correct / total) : 0.0f;
    Serial.printf("Overall accuracy: %.2f%% (%d/%d)\n", acc, correct, total);
}

int runInferenceOnFile(const char* filename, int trueLabel) {
    // Create new dispenser on heap
    InputDispenser* dispenser = new InputDispenser(
        filename,
        NUM_CHANNELS,
        NUM_ROWS,
        dispenserQuantScale,
        dispenserQuantZeroPoint,
        inputTensor,
        nullptr
    );

    if (!dispenser->begin()) {
        Serial.printf("[ERROR] Failed to begin dispenser for %s\n", filename);
        delete dispenser;
        return -1;
    }

    // Link EEGNet and dispenser
    dispenser->setOutputBuffer(inputTensor, EEGNet.input()->data.uint8);
    EEGNet.getInputQuantisation(dispenserQuantScale, dispenserQuantZeroPoint);
    dispenser->setQuantisation(dispenserQuantScale, dispenserQuantZeroPoint);

    // Reset pooling
    pooler.reset();
    size_t local_frame_counter = 0;
    bool local_file_depletion = false;

    // Read frames
    while (local_frame_counter < frame_target && !local_file_depletion) {
        for (size_t i = 0; i < NUM_ROWS; ++i) {
            if (!dispenser->streamNext()) {
                if (dispenser->isEOF()) {
                    local_file_depletion = true;
                } else {
                    Serial.printf("[ERROR] Failed to stream row from %s\n", filename);
                    delete dispenser;
                    return -1;
                }
                break;
            }
        }

        if (!EEGNet.invoke()) {
            Serial.println("[ERROR] FE inference failed");
            delete dispenser;
            return -1;
        }

        float output_buffer[NUM_OUTPUTS_FE];
        EEGNet.getOutput(output_buffer, NUM_OUTPUTS_FE);
        pooler.add(output_buffer);
        local_frame_counter++;
    }

    // Pool finalize
    float feature_vector[NUM_OUTPUTS_FE];
    pooler.finalize(feature_vector);

    // Run classifier
    TfLiteTensor* cls_input = Classifier.input();
    for (int i = 0; i < NUM_OUTPUTS_FE; i++) {
        cls_input->data.f[i] = feature_vector[i];
    }

    if (!Classifier.invoke()) {
        Serial.println("[ERROR] Classifier inference failed");
        delete dispenser;
        return -1;
    }

    float results[NUM_OUTPUTS_CLS];
    Classifier.getOutput(results, NUM_OUTPUTS_CLS);

    int pred = (results[1] > results[0]) ? 1 : 0;
    Serial.printf("%s -> pred=%d, truth=%d\n", filename, pred, trueLabel);

    delete dispenser; // destroy to free resources
    return (pred == trueLabel) ? 1 : 0;
}

// //////////////////////
// ///// Setup Time /////
// //////////////////////

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    // --- Prepare Data Dispenser ---
    if (!SD.begin(SD_CS_PIN)) {
        Serial.println("[Main] SD initialization failed!");
        while (true);
    }

    // if (!dispenser.begin()) {
    //     Serial.println("[Main] Failed to begin dispenser.");
    //     while (true);
    // }

    // --- Setup EEGNet Model ---
    setup_resolver();

    if (!EEGNet.begin(eegnet, resolver, fe_tensor_arena, feTensorArenaSize)) {
      Serial.println("[Main] Failed to begin EEGNet");
      while (true);
    }

    // // --- Link EEGNet and Dispenser ---
    // dispenser.setOutputBuffer(inputTensor, EEGNet.input()->data.uint8);
    
    // EEGNet.getInputQuantisation(dispenserQuantScale, dispenserQuantZeroPoint);
    // dispenser.setQuantisation(dispenserQuantScale, dispenserQuantZeroPoint);

    Serial.println("[Main] Experiment started.");

    // --- Setup Classifier Model ---
    if (!Classifier.begin(classifier, resolver, cls_tensor_arena, clsTensorArenaSize)) {
        Serial.println("[Main] Failed to begin Classifier");
        while (true);
    }

    numFiles = collectFileList(dataDir);
    Serial.printf("Collected %u files from root\n", (unsigned)numFiles);
}

void loop() {
    resetConfusion();

    for (size_t i = 0; i < numFiles; i++) {
        const char* fname = fileList[i];
        uint8_t truth = parseLabelFromName(fname);

        char fullPath[32];
        snprintf(fullPath, sizeof(fullPath), "%s/%s", dataDir, fname);

        int res = runInferenceOnFile(fullPath, truth);
        if (res >= 0) {
            // If res == 1 → pred == truth
            if (res == 1) {
                confusion[truth][truth]++; 
            } else {
                // Wrong prediction → since binary, pred must be the opposite
                uint8_t pred = (truth == 0) ? 1 : 0;
                confusion[truth][pred]++;
            }
        }
    }

    // After all files
    reportConfusion();

    while (true); // stop
}


