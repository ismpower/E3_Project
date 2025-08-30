# Data Acquisition and Verification Protocol (DAVP) Pipeline Design

This document outlines the architectural design for the DAVP pipeline, covering data ingestion, standardization, error handling, and metadata tracking.

## 1. Input Formats Specification

The DAVP pipeline must support the ingestion and parsing of data from the following formats. Each format requires specific parsers and validation routines.

### 1.1. CIF (Crystallographic Information File)
* **Purpose:** Primarily for crystal structures from sources like Materials Project and OQMD.
* **Key Information to Extract:**
    * Atomic species and their fractional coordinates.
    * Unit cell parameters (a, b, c, alpha, beta, gamma).
    * Space group information.
    * Any associated material properties (e.g., formation energy, band gap) if embedded.
* **Parsing Considerations:**
    * Robust parsers for diverse CIF syntax variations.
    * Handling of different data blocks.
    * Error checking for incomplete or malformed CIF entries.

### 1.2. SMILES (Simplified Molecular-Input Line-Entry System)
* **Purpose:** For molecular structures, typically from PubChem and DrugBank.
* **Key Information to Extract:**
    * Molecular graph (atoms and bonds).
    * Stereochemistry if encoded.
    * Implicit hydrogen atoms.
* **Parsing Considerations:**
    * Use of well-established cheminformatics libraries (e.g., RDKit, OpenBabel) for parsing and validation.
    * Handling of various SMILES conventions (isomeric, canonical, etc.).

### 1.3. PDB (Protein Data Bank)
* **Purpose:** For protein and biomolecular structures, from sources like RCSB PDB.
* **Key Information to Extract:**
    * Atomic coordinates (XYZ for atoms, heteroatoms).
    * Residue names and chain identifiers.
    * Bond connectivity (though often inferred).
    * Ligand information.
* **Parsing Considerations:**
    * Handling of multi-model files.
    * Robust parsing of fixed-column format.
    * Extraction of relevant metadata from PDB header.

### 1.4. JSON (JavaScript Object Notation)
* **Purpose:** General-purpose structured data, potentially from custom experimental outputs or API responses.
* **Key Information to Extract:**
    * Flexible schema, requires dynamic or configurable parsing based on expected JSON structure.
    * Must accommodate nested structures.
* **Parsing Considerations:**
    * Schema validation (e.g., using JSON Schema) where possible.
    * Graceful handling of missing or unexpected fields.

### 1.5. CSV (Comma Separated Values)
* **Purpose:** Tabular data, common for experimental results or simple property lists.
* **Key Information to Extract:**
    * Column headers and corresponding data types.
    * Numerical and categorical values.
* **Parsing Considerations:**
    * Robust handling of delimiters, quoted fields, and missing values.
    * Type inference for columns or explicit schema definition.
    * Identification of key columns for linking to graph structures (e.g., element names, IDs).

## 2. Standardized Graph Representation

All ingested data, regardless of its original source format, must be converted into a standardized graph representation. This graph will be the primary input for the E3 Engine's GNN model. The representation should be a single, cohesive data structure that captures both the inherent properties of the system and its contextual environment.

### 2.1. Graph Structure Components

The graph representation will be composed of the following key components:

* **Nodes:** Each node represents an individual atom or ion in the system.
    * `node_id`: Unique identifier.
    * `element_type`: One-hot encoded vector representing the atomic species (e.g., Argon, Strontium, Rubidium, Iodine).
    * `position`: 3D coordinates (x, y, z) of the atom/ion.
    * `node_features`: A feature vector containing context-dependent properties:
        * `local_temperature` ($T_e$).
        * `local_density` ($n_0$).
        * `coulomb_coupling_parameter` ($\Gamma$).

* **Edges:** Edges encode pairwise interactions between nodes.
    * `source_node_id`, `target_node_id`.
    * `interaction_type`: Categorical label (e.g., `bond`, `screening`, `coulomb_force`).
    * `edge_features`: A feature vector representing the interaction:
        * `distance`: Euclidean distance between the two nodes.
        * `screening_length`: Characteristic length scale of plasma screening effects.
        * `effective_potential`: Value of the interaction potential at that distance.

* **Global Properties:** Global properties of the entire system.
    * `system_id`: Unique identifier for the system (e.g., `Sr_Exp_Run_8`).
    * `timestamp`: Time of the original measurement.
    * `total_energy`, `total_entropy`, etc. (if available).

### 2.2. Output Format Specifications

The pipeline will output this standardized graph representation in a format suitable for the E3 GNN model, such as a PyTorch Geometric `Data` object or a similar, easy-to-parse structure like a JSON file that conforms to this schema.


## 3. Error Handling Protocol

A robust error handling protocol is critical to maintain data quality and ensure that only valid, well-formed data enters the E3 Engine's processing pipeline. This protocol focuses on early detection, clear flagging, and isolation of problematic data.

### 3.1. Error Detection Mechanisms

The pipeline will employ several mechanisms to detect errors during the ingestion and standardization process:

* **Parsing Errors:** Failure to parse the raw input file according to its specified format (e.g., malformed CIF syntax, unclosed SMILES string, invalid JSON structure).
* **Schema Validation:** Data not conforming to the expected schema for a given input type or the standardized graph representation (e.g., missing required fields, incorrect data types).
* **Data Range/Constraint Checks:** Values falling outside physically plausible or defined ranges (e.g., negative temperatures, atomic coordinates outside a reasonable cell, bond lengths too short/long).
* **Consistency Checks:** Inconsistencies between derived and explicit information (e.g., calculated molecular weight not matching provided weight).

### 3.2. Flagging and Quarantine of Malformed Files

Upon detection of an error, the following actions will be taken:

* **Flagging:** The record or file will be immediately flagged with one or more `quality_flags` within its metadata. These flags will indicate the type of error detected (e.g., `parsing_error`, `schema_mismatch`, `out_of_range_value`).
* **Quarantine:** Malformed files or problematic records will not proceed into the main processing stream. Instead, they will be moved to a designated "quarantine" area (e.g., a specific directory or a separate database table for review). This ensures that problematic data does not corrupt downstream processes.
    * **Quarantine Location:** Define a specific directory structure for quarantined files, possibly categorized by error type or source.
    * **Quarantine Logging:** Each quarantined item will have a corresponding entry in the data ingestion log, linking to its quarantine location and detailed error message.
* **Partial Processing (if applicable):** For files with multiple independent records, if some records are valid and others are malformed, valid records may proceed while malformed ones are quarantined (e.g., in a large CSV file, only problematic rows are skipped/quarantined).

### 3.3. Error Logging and Reporting

All detected errors and quarantine actions must be meticulously logged according to the "Data Ingestion & Processing" section of the `logging_requirements.md` document.

* **Error Messages:** Detailed error messages, including file paths, line numbers (if applicable), and specific reasons for failure, must be captured.
* **Automated Alerts:** The system should be capable of generating automated alerts (e.g., email notifications, internal dashboard alerts) for significant volumes of errors or critical failures, prompting human intervention.
* **Human Review:** A clear process for human review of quarantined data and error logs must be established to debug issues, refine parsers, or decide on data exclusion.


## 4. Metadata Tracking System

The metadata tracking system is a crucial element of the DAVP, providing a comprehensive "data lineage" for every standardized graph representation produced by the pipeline. This metadata is stored alongside the data itself and is critical for reproducibility, debugging, and auditing.

### 4.1. Core Metadata Fields

The metadata will be captured at multiple stages and must include:

* **Source Provenance:**
    * `source_id`: Unique identifier for the original data source.
    * `source_name`: Human-readable name (e.g., `Materials_Project`).
    * `source_url`: Link to the original data entry.
    * `download_timestamp`: The date and time the data was ingested from the source.
    * `original_file_hash`: A hash of the original raw file to ensure data integrity.

* **Processing Parameters:**
    * `pipeline_version`: Version of the DAVP pipeline used for processing.
    * `processing_id`: Unique identifier for the specific processing run.
    * `processing_parameters`: A JSON object detailing all parameters used during the conversion from the input format to the standardized graph representation (e.g., featurization method, bond length cutoffs, etc.).

* **Quality Flags:**
    * `quality_flags`: A list of strings or a dictionary of flags generated by the error handling protocol.
    * `review_status`: A field to indicate if the data has been manually reviewed (`pending`, `reviewed_and_accepted`, `reviewed_and_rejected`).
    * `error_message`: A summary of any errors that occurred during processing.

### 4.2. Metadata Integration

This metadata will be stored in a structured format (e.g., JSON) within the standardized graph representation itself. This ensures that the data is "self-describing" and that its lineage is always linked directly to the data object, facilitating its use downstream by the E3 Engine.
