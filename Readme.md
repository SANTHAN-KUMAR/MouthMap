<div align="center">
  <img src="https://raw.githubusercontent.com/d-kavinraja/MouthMap/main/Img-src/Lip%20Movement.gif" alt="MouthMap Lip Reading Animation" width="600"/>
  <h1>MouthMap: Lip Reading with Deep Learning</h1>
  <p><strong>Translating Silence into Sentences with AI 🤫➡️✍️</strong></p>
  
  <p>
    <a href="https://www.python.org/downloads/release/python-380/"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"></a>
    <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow Version"></a>
    <a href="https://github.com/yourusername/MouthMap/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
    <a href="https://github.com/yourusername/MouthMap/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg" alt="Pull Requests Welcome"></a>
    <a href="https://www.kaggle.com/models/santhankarnala/40th-epoch-model-checkpoint/Keras/default/1"><img src="https://img.shields.io/badge/Kaggle%20Model-Available-blue" alt="Kaggle Model"></a>
  </p>
</div>

MouthMap is an advanced deep learning project that interprets lip movements from video and transcribes them into text. Using a sophisticated architecture of 3D Convolutional Neural Networks (CNNs) and Bidirectional LSTMs with a CTC loss function, MouthMap pushes the boundaries of silent speech recognition.

---

## 🌟 Key Features

-   🎥 **Video-to-Text Transcription**: Converts lip movements in videos directly into coherent text sentences.
-   🧠 **State-of-the-Art Architecture**: Built on a powerful 3D CNN + Bi-LSTM model to capture complex spatio-temporal features.
-   ⚙️ **End-to-End Pipeline**: Provides a complete workflow from video preprocessing and data loading to model training and inference.
-   🚀 **Real-time Potential**: Engineered for efficiency, laying the groundwork for live transcription applications.
-   🧩 **Modular & Customizable**: The code is well-structured, making it easy to adapt, extend, and experiment with.

---

## 📲 Live Demo & Exported Model

Our app is available to try! While accuracy with custom videos is still being improved, you can test its capabilities with the provided samples.

-   **Try the App**: `https://mouthmap.streamlit.app/`
-   **Download the Trained Model**: A pre-trained model checkpoint from the 40th epoch is available on Kaggle: [**Download from Kaggle Models**](https://www.kaggle.com/models/santhankarnala/40th-epoch-model-checkpoint/Keras/default/1)

---

## 🏗️ Model Architecture

MouthMap processes video frames by first extracting features through a series of 3D convolutional layers, which are ideal for capturing both spatial details (lip shape) and temporal changes (movement). The output is then passed to a Bidirectional LSTM network to understand the sequential context of the speech, and finally, a Dense layer predicts the text.

![Model Architecture](./Img-src/Model%20Architecture.png)

| Layer Type                | Details                                           | Purpose                                            |
| :------------------------ | :------------------------------------------------ | :------------------------------------------------- |
| **Input Layer** | `(75, 46, 140, 1)`                                | 75 frames of 46x140 grayscale video.               |
| **3x Conv3D + MaxPool3D** | Kernels: `(3,3,3)`, Filters: `128 -> 256 -> 75`    | Hierarchical feature extraction from video data.   |
| **Reshape Layer** | Flattens spatial dimensions.                      | Prepares data for sequential processing by LSTMs.  |
| **2x Bidirectional LSTM** | `128` units each, with `50%` Dropout.             | Captures temporal dependencies in speech patterns. |
| **Dense Layer** | `41` units + `Softmax` activation.                | Outputs character probabilities for CTC decoding.  |

---

## 🚀 Getting Started

Follow these steps to set up the project on your local machine.

### Prerequisites

-   Python 3.8+
-   TensorFlow 2.x
-   FFmpeg (for video processing, often handled by OpenCV but good to have)

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/MouthMap.git](https://github.com/yourusername/MouthMap.git)
    cd MouthMap
    ```

2.  **Create a Virtual Environment & Install Dependencies**
    It's highly recommended to use a virtual environment.
    ```bash
    # Create and activate the virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install the required packages
    pip install -r requirements.txt
    ```

3.  **Download the Dataset**
    Run the following Python script to download and extract the dataset.
    ```python
    import gdown
    import os

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Download and extract
    url = '[https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL](https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL)'
    output = 'data/data.zip'
    gdown.download(url, output, quiet=False)
    gdown.extractall(output, 'data/')
    ```

---

## 🧪 Quick Inference Example

Here’s how you can use the pre-trained model to transcribe a video. Make sure to download the model weights from the [Kaggle link](#-live-demo--exported-model) and place them in a `models` directory.

```python
import tensorflow as tf
import numpy as np
from typing import List
# Note: You will need to import your project's custom functions
# For example: from your_project_utils import load_video, num_to_char
# And from your_project_model import build_model, CTCLoss

# 1. Build the model and load weights
# Ensure build_model() and CTCLoss are defined as in the original repository
model = build_model() 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=CTCLoss)
model.load_weights('./models/checkpoint.weights.h5') # Update path to your downloaded weights

# 2. Load and preprocess a sample video
# Ensure load_video() is defined as in the original repository
video_path = './data/s1/bbal6n.mpg' 
frames = load_video(video_path)
video_tensor = tf.expand_dims(frames, axis=0) # Add batch dimension

# 3. Predict the sequence of character probabilities
yhat = model.predict(video_tensor)

# 4. Decode the output to get the final text
# Ensure num_to_char is defined (from the StringLookup layer)
decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
predicted_text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')

print(f"✅ Predicted Text: {predicted_text}")
# Example Output -> Original: "bin blue at l six now"
