# AGENDA Epilepsy Diagnosis Algorithm (Embedded)

Greetings, visitors. This is the repository for all relevant TinyML implementation codes on the AGENDA project. You will find three main folders:

## "/Arduino"

This folder contains the Arduino codes for running the model testing pipeline. Many of the codes, I believe, could be directly reused for integration into the complete AGENDA firmware project.

Non-reusable codes include:

- **teensy_test.ino**: This is the main Arduino file that calls the different components to run testing. It could serve as a reference for how to use the inference engine, but no code is directly reusable.
- **data_dispenser.cpp / .h**: This code is for moving the patient data from SD storage to the input buffer. It could serve as a reference for quantization procedures, but no code is directly reusable.

Reusable with caution:

- **eegnet.h**: This is the header file that contains the feature extractor model. See "/Models" for more detail.
- **classifier.h**: This is the header file that contains the classifier model. See "/Models" for more detail.

Reusable components:

- **CLS_TFLite.cpp / .h**: This code contains the wrapper for the classifier head. It's meant to be cleanly dropped in and used. See "teensy_test.ino" for example usage.
- **avg_pooling.cpp / .h**: This code contains the wrapper for average pooling (not a TFLite implementation). It's meant to be cleanly dropped in and used. See "teensy_test.ino" for example usage.
- **FE_TFLite.cpp / .h**: This code contains the wrapper for the feature extractor. It's meant to be cleanly dropped in and used. See "teensy_test.ino" for example usage.

## "/Models"

This folder contains the header files that hold the TinyML models. They are organized by folds, and each is selected based on validation performance.

To be able to directly drop in and use, the output of the model conversion factory should be processed following these steps:

- Rename the model files to "eegnet.h" and "classifier.h".
- Delete the "#include ..." line at the start of each file.
- Rename the model variable to "eegnet" and "classifier".
- Delete the model length variable at the end of each file.
- For optimal performance, add "alignas(16)" before the model variable declaration.

## "/Datasets"

This folder contains the dataset that can be directly dropped onto the SD card and used for testing. 

*NOT YET — WAITING TO FIGURE OUT HOW TO SECURELY PLACE PATIENT DATA*.

To use it, drop the whole folder of ".csv" files onto the SD card, and change line 64 in "teensy_test.ino" (*dataDir*) to match the name of the dataset folder.


## Miscellaneous Notes

- To use the code smoothly, install the Arduino libraries **"ArduTFLite"** and **"Chirale_TensorFlowLite"**.
- To minimize the memory footprint, opt in only for the following resolvers:
  - AddQuantize()
  - AddConv2D()
  - AddAdd()
  - AddMul()
  - AddDepthwiseConv2D()
  - AddHardSwish()
  - AddAveragePool2D()
  - AddReshape()
  - AddFullyConnected()


