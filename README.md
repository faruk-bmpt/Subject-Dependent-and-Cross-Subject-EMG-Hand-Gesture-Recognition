# Subject-Dependent and Cross-Subject EMG Hand-Gesture Recognition: A Multi-Perspective Evaluation Framework

A modular Python pipeline for evaluating EMG-based hand-gesture
recognition using subject-dependent and cross-subject experiments.

## Repository contents

  -----------------------------------------------------------------------
  File / block                        Purpose
  ----------------------------------- -----------------------------------
  `artifact_manager.py`               Manages experiment artifacts and
                                      outputs.

  `BLOCK_A_DATA1.py`                  Checks dataset integrity and
                                      recording inventory.

  `BLOCK_B_MEM1.py`                   Supports memory and resource
                                      management.

  `BLOCK_C_LOG1.py`                   Handles experiment logging and run
                                      tracking.

  `BLOCK_D_PRE1.py`                   Preprocesses EMG recordings.

  `BLOCK_0_OPT1.py`                   Evaluates the design/search space
                                      for pipeline settings.

  `BLOCK_1_LEAK1.py`                  Checks for potential data leakage.

  `BLOCK_2_FE1.py`                    Extracts and selects EMG features.

  `BLOCK_3_GEN1.py`                   Evaluates cross-subject
                                      generalization, including LOSO and
                                      calibration experiments.

  `BLOCK_4_PER1.py`                   Evaluates personalized,
                                      subject-dependent recognition.

  `BLOCK_5_AB1.py`                    Runs feature or pipeline ablation
                                      experiments.

  `BLOCK_6_XAI1.py`                   Provides model explainability
                                      analyses.

  `BLOCK_7_STAT1.py`                  Performs statistical validation and
                                      comparisons.

  `BLOCK_8_BEN1.py`                   Benchmarks models and
                                      configurations.

  `BLOCK_9_CM1.py`                    Analyzes confusion matrices and
                                      gesture-level performance.

  `BLOCK_10_DEP1.py`                  Assesses deployment-related
                                      performance.

  `BLOCK_11_VIS1.py`                  Visualizes feature-space or learned
                                      representations.

  `BLOCK_12_ERR1.py`                  Characterizes recognition errors.

  `BLOCK_13_ROB1.py`                  Evaluates robustness under tested
                                      conditions.

  `common.py`                         Shared helper functions.

  `config.py`                         Central configuration settings.

  `state_manager.py`                  Tracks pipeline state.

  `block0_run.txt`                    Run notes or status text for Block
                                      0.

  `requirements.txt`                  Python dependencies.

  `run_pipeline.sh`                   Shell entry point for running the
                                      pipeline.
  -----------------------------------------------------------------------

## Models

The pipeline specification includes Linear Discriminant Analysis (LDA),
k-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest
(RF), and XGBoost (XGB).

## How to use

1.  **Prepare the environment.** Use a compatible Python environment and
    install the listed dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

2.  **Review the configuration.** Open `config.py` and check paths,
    settings, and experiment options before running.

3.  **Prepare the dataset.** Ensure the EMG data are available in the
    format and location expected by the pipeline.

4.  **Run the pipeline.** From a compatible shell, run:

    ``` bash
    bash run_pipeline.sh
    ```

5.  **Check outputs.** Review generated logs, metrics, and artifacts to
    confirm which blocks completed successfully.

Run from the repository root. The exact data paths, operating-system
requirements, and output locations depend on the configuration and
scripts in the repository.

## Evaluation scope

The repository is organized to examine preprocessing, feature
extraction, leakage control, subject-independent and subject-dependent
performance, calibration, ablation, explainability, statistical
comparison, benchmarking, gesture-level errors, deployment,
visualization, and robustness.

**Note:** The block descriptions summarize the apparent roles indicated
by the filenames and pipeline organization. Check each script and its
run logs to confirm implementation details and completion; this README
does not claim that every block has been executed successfully.
