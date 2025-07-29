# E3 Engine: Elemental Embedding Engine

## Overview

The **Elemental Embedding Engine (E3)** is a physics-informed Graph Neural Network designed to predict multi-stage entropic relaxation dynamics in non-equilibrium plasmas. The E3 engine resolves the long-standing anomaly in ultracold neutral plasma expansion by learning context-dependent representations of elemental behavior.

## Key Features

- **Physics-Informed GNN**: Learns context-dependent elemental embeddings
- **Multi-Stage Prediction**: Predicts both tau (relaxation time) and t_break (buffer time)
- **Cross-Chemical Validation**: Trained on 4 chemical families (Noble Gas, Alkali Metal, Halogen, Alkaline Earth)
- **High Performance**: Achieves R² = 0.9993 on validation data

## Repository Structure

### 🎯 E3 Engine Core
- `e3_engine/core/` - Main E3 GNN implementation
- `e3_engine/models/` - Neural network architectures
- `e3_engine/training/` - Training configurations and scripts
- `e3_engine/validation/` - Model validation tools

### 📊 Demonstrations & Results
- `notebooks/e3_demos/` - E3 Engine demonstrations
- `notebooks/e3_training/` - Model training workflows
- `notebooks/e3_validation/` - Performance validation
- `results/archipelago_visualizations/` - Project Archipelago outputs

### 🔬 Supporting Bruno Framework
- `bruno_framework/theory/` - Theoretical physics foundation
- `bruno_framework/constants/` - Bruno constant calculations
- `bruno_framework/applications/` - Astrophysical applications

### 📁 Data & Processing
- `data/experimental/` - Ultracold plasma experimental data
- `data/processed/` - E3-ready datasets
- `scripts/data_processing/` - Data preparation tools

## Quick Start

### 1. Training the E3 Model
```bash
python e3_engine/core/entropy_first_model.py
```

### 2. Running E3 Demonstrations
```bash
jupyter notebook notebooks/e3_demos/Project_Archipelago.ipynb
```

### 3. Validating Model Performance
```bash
jupyter notebook notebooks/e3_validation/Historical_Entropy_Audit.ipynb
```

## Key Publications & Results

- **Model Performance**: R² = 0.9993 across four chemical families
- **Discovery**: Two-regime physics with temperature crossover at ~10-25 K
- **Innovation**: Context-aware elemental embeddings for entropic prediction

## Bruno Theoretical Framework

The E3 Engine is supported by the Bruno Collapse Framework, which provides:
- **Bruno Constant (κ)**: Empirical scaling relationship (1340 ± 60 × 10⁻⁶ K⁻¹s⁻¹)
- **Thermodynamic Foundation**: Entropy-first physics approach
- **Astrophysical Applications**: Supernova and multimessenger predictions

## Citation

If you use the E3 Engine in your research, please cite:

```bibtex
@article{chajar2025e3,
  title={The Elemental Embedding Engine (E3): A Physics-Informed Graph Neural Network for Predicting Multi-Stage Entropic Relaxation in Non-Equilibrium Plasmas},
  author={Chajar, Ismail},
  journal={Journal of Computational Physics},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Author**: Ismail Chajar
- **Institution**: EthI.C Lab
- **Email**: i.chajar@ethic-lab.ca
- **Location**: Saint-Jean-sur-Richelieu, Quebec, Canada
