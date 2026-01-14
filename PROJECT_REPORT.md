# Project Analysis & Clean-up Report

## 1. Clean-up Actions Summary
- **Irrelevant folders removed**: `Approach-1`, `Approach-2`, `Approach-3` have been moved to an `archive/` directory to declutter the workspace.
- **Primary source identified**: `Current--Approach` contains the active `MouthMap.ipynb` which holds the core logic.

## 2. Technical Analysis of "Current--Approach"

### Architecture
Your model implements the **LipNet** architecture (Spatio-Temporal Convolutional Neural Network):
1.  **Input**: 3D Video Tensor `(Batch, 75 frames, 46 height, 140 width, 1 channel)`
2.  **Visual Feature Extraction (STCNN)**:
    - 3x layers of `Conv3D` + `Activation(Relu)` + `MaxPool3D`
    - Filter progression: 128 -> 256 -> 75
    - This extracts spatial features (shapes of lips) and temporal features (movement across frames).
3.  **Sequence Modeling (RNN)**:
    - 2x layers of `Bidirectional LSTM` (128 units each)
    - Orthogonal initialization (standard for RNNs to prevent exploding gradients).
    - `Dropout(0.5)` for regularization.
4.  **Output**: `Dense` layer with **CTC (Connectionist Temporal Classification)** Loss via `Softmax` activation.

### Current "Score" & Performance
Based on the execution logs found in `MouthMap.ipynb`:
- **Training Epoch**: The logs show widely varying loss, starting high (~99) and decreasing to ~52 by epoch 10.
- **Prediction Quality**:
    - *Early Epochs*: The model output was gibberish (e.g., `josssssss...`). This is normal for CTC models early in training; they tend to output repeated characters (like `s` or blank tokens) before learning the alignment.
    - *40th Epoch (from README)*: The README claims a "40th epoch model checkpoint" is available on Kaggle. This implies the model *did* converge to a usable state in a separate run, even if the current local notebook log only shows early epochs.
- **Metric**: The primary metric is **CTC Loss**. A lower loss indicates better alignment between the predicted character sequence and the target text. You do not have a separate "Accuracy" or "Word Error Rate (WER)" metric printed in the training loop explicitly, but visual inspection of predictions (`tf.strings.reduce_join`) serves as the qualitative score.

## 3. Resume Framing
To put this on your resume effectively, describe it as:
> **MouthMap - Deep Learning Lip Reading System**
> *Implemented an end-to-end lip reading pipeline using TensorFlow/Keras. Architecture combines 3D Convolutional Neural Networks (STCNN) for spatiotemporal feature extraction and Bidirectional LSTMs with CTC Loss for sequence alignment. Achieved sentence-level prediction on the GRID corpus, deployed via a Streamlit web application.*

## 4. Next Steps for "Productionizing"
To finish the "fix" and make this a proper software project:
1.  **Extract Code**: The logic inside `Current--Approach/MouthMap.ipynb` is solid but "trapped". I will now move the `load_video`, `build_model`, and training loops into standard Python files (`src/*.py`).
2.  **Standardize Config**: Move hardcoded paths (`/kaggle/working/`, `data/s1/`) into a config file so it runs on your machine without edits.
3.  **Testing**: Add a simple test to verify the model builds with the correct input shape.

I will now proceed to creating the `src` python package with the extracted logic.
