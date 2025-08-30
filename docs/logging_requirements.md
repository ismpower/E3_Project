# Logging Requirements for E3 Engine

This document outlines the essential information that must be captured and logged for various processes within the E3 Engine development, ensuring a complete and auditable history of all research activities, data operations, and model runs.

All logs should include a timestamp (UTC, ISO 8601 format) and the user/process ID that initiated the action.

## 1. Model Training Runs

**Purpose:** To track the full lineage and performance of every trained model.

**Required Fields:**
* `run_id`: Unique identifier for each training run (e.g., UUID).
* `model_name`: Name/version of the E3 model being trained (e.g., `E3v2_SrRbIAr`).
* `timestamp`: Date and time the training run started.
* `researcher_id`: Identifier of the researcher initiating the run.
* `dataset_id`: Identifier of the specific dataset version used for training.
* `dataset_path`: Path or URI to the dataset used.
* `hyperparameters`: JSON object containing all hyperparameters (e.g., learning rate, epochs, batch size, optimizer type, GNN layers/units).
* `gpu_config`: Details of GPUs used (e.g., `RTX 3080 x2`).
* `training_metrics`: JSON object with final training metrics (e.g., loss, R² score, Mean Absolute Error (MAE) for $\tau$ and $t_{break}$).
* `validation_metrics`: JSON object with final validation metrics on the validation split (e.g., loss, R² score, MAE for $\tau$ and $t_{break}$).
* `model_save_path`: Path where the trained model artifact is saved.
* `duration_seconds`: Total training duration in seconds.
* `status`: (e.g., `completed`, `failed`, `interrupted`).
* `error_message`: (Optional) Details if `status` is `failed`.

## 2. Data Ingestion & Processing

**Purpose:** To provide full provenance and transformation history for all data points.

**Required Fields:**
* `ingestion_id`: Unique identifier for each ingestion event.
* `timestamp`: Date and time of ingestion.
* `source_name`: Name of the external data source (e.g., `Materials Project`, `Killian_2007_Sr_Exp`).
* `source_url`: URL or identifier of the original data location.
* `original_file_name`: Original filename if applicable.
* `original_file_hash`: SHA256 hash of the raw input file before any processing.
* `input_format`: Original format of the ingested data (e.g., `CIF`, `JSON`, `CSV`).
* `processed_dataset_id`: Identifier for the resulting processed dataset version.
* `processed_file_path`: Path where the standardized graph representation is saved.
* `processing_parameters`: JSON object of parameters used during standardization (e.g., atom featurization method, graph construction rules).
* `quality_flags`: List of any flags raised during ingestion/processing (e.g., `malformed_entry`, `missing_metadata`, `out_of_range`).
* `num_records_ingested`: Number of records successfully ingested.
* `num_records_quarantined`: Number of records flagged and quarantined.
* `error_log_path`: Path to a detailed error log if issues occurred.

## 3. Validation Tests

**Purpose:** To record the results and context of all model validation tests.

**Required Fields:**
* `test_id`: Unique identifier for the validation test.
* `timestamp`: Date and time the test was executed.
* `test_type`: (e.g., `LOOCV`, `external_dataset_validation`, `predictive_inference`).
* `model_id`: Identifier of the specific trained model being validated.
* `dataset_id`: Identifier of the dataset used for validation.
* `test_metrics`: JSON object with metrics relevant to the test (e.g., R² score, MAE, RMSE, precision, recall, F1-score depending on task).
* `test_parameters`: Any specific parameters for this test (e.g., cross-validation folds, discrepancy thresholds).
* `passed`: Boolean indicating if the test met predefined success criteria.
* `report_path`: Path to the detailed validation report/notebook output.
* `anomalies_detected`: (Optional) List of any anomalies or unexpected behaviors observed during validation.
