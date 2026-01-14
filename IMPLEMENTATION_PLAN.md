# MouthMap Improvement & Implementation Plan

## 1. Project Analysis & Resume Assessment

**Current Status:**
The project demonstrates advanced Deep Learning skills (3D CNNs, RNNs, CTC Loss) and creates a tangible product (Streamlit app). However, the *repository structure* and *code organization* reflect an experimental/research phase rather than a production-ready engineering project.

**Identified Flaws:**
1.  **Directory Structure**: Folders like `Approach-1`, `Approach-2`, `Current--Approach` clutter the root and confuse the user. It looks like a "work in progress" rather than a finished product.
2.  **Notebook Dependency**: core logic (Data loading, Model architecture, Training) is locked inside `.ipynb` files. This makes the code hard to test, import, or reuse.
3.  **Reproducibility**: Lack of a clear `requirements.txt` (or `pyproject.toml`) linked to the specific working version.
4.  **No Testing**: There are no unit tests to verify data shapes, model output, or utility functions.
5.  **Hardcoded Paths**: Notebooks often contain paths specific to the local machine or Google Drive (e.g., `/kaggle/working/`).

**Resume Potential:**
To make this "Resume-Ready", the project needs to look like a **Python Package** or a **Microservice**. It should demonstrate that you know how to write clean, modular, and maintainable software, not just run experiments in Jupyter.

---

## 2. Implementation Plan

### Phase 1: Restructuring & Cleanup
*Goal: Create a standard, professional directory structure.*

*   [ ] **Archive Old Experiments**: Move `Approach-1`, `Approach-2`, `Approach-3` into an `archive/` or `experiments/` directory (or delete them if not needed).
*   [ ] **Standard Layout**: Create the following structure:
    ```text
    MouthMap/
    ├── .github/workflows/   # CI/CD (Optional but recommended)
    ├── data/                # Data storage (gitignored)
    ├── notebooks/           # For experiments ONLY
    ├── src/                 # Source code package
    │   ├── __init__.py
    │   ├── config.py        # Constants
    │   ├── data_loader.py   # Video & Alignment loading
    │   ├── model.py         # Architecture definition
    │   ├── train.py         # Training loop script
    │   ├── inference.py     # Prediction logic
    │   └── utils.py         # Helpers (Vocab, CTC decode)
    ├── tests/               # Unit tests
    ├── app.py               # Streamlit entry point (cleaned up)
    ├── requirements.txt     # Dependencies
    └── README.md            # Documentation
    ```

### Phase 2: Code Modularization
*Goal: Move logic from 'Current--Approach/MouthMap.ipynb' to 'src/'.*

*   [ ] **Configuration (`src/config.py`)**: Centralize constants like `BATCH_SIZE`, `FRAME_COUNT`, `VOCAB`.
*   [ ] **Data Pipeline (`src/data_loader.py`)**: Extract `load_video`, `load_alignments`, and `mappable_function`. Ensure they handle paths robustly using `os.path`.
*   [ ] **Model Architecture (`src/model.py`)**: Extract the `Sequential` model build and `CTCLoss` into clean functions.
*   [ ] **Utilities (`src/utils.py`)**: Move vocabulary generation and `num_to_char` decoding logic here.

### Phase 3: Professional Standards
*Goal: Ensure code quality and consistency.*

*   [ ] **Type Hinting**: Add Python type hints (`List`, `Tuple`, `tf.Tensor`) to all functions.
*   [ ] **Docstrings**: Add Google or NumPy style docstrings explaining inputs and outputs.
*   [ ] **Error Handling**: Add checks (e.g., if a video file is missing or empty).

### Phase 4: Application & Training Scripts
*Goal: Make the code executable via command line.*

*   [ ] **Refactor `app.py`**: Import functions from `src/` instead of redefining them.
*   [ ] **Create `train.py`**: A script to retrain the model from CLI (e.g., `python src/train.py --epochs 50`).

### Phase 5: Documentation
*Goal: Make it easy for recruiters/users to understand.*

*   [ ] **Update README**: Point to the new structure. Add a "Project Structure" section.
*   [ ] **Installation Guide**: Clear steps to install deps and run the app.

---

## 3. Immediate Next Steps

I recommend we start with **Phase 1 and 2** immediately.
1.  I can create the directory structure.
2.  I will extract the code from your notebook into the `src/` files.
3.  I will verify the imports work.

Shall I proceed with **Phase 1 (Restructuring)**?
