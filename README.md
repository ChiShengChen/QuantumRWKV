# Quantum RWKV Project

This project explores and implements a Quantum-enhanced Recurrent Weighted Key Value (RWKV) model for time series prediction tasks. It provides both classical and quantum versions of the RWKV model and compares their performance on various time series datasets.

## Project Structure

The repository is organized as follows:

```
Quantum_rwkv/v2
├── rwkv.py                             # Classical RWKV model implementation
├── quantum_rwkv.py                     # Quantum-enhanced RWKV model implementation
├── test_rwkv.py                        # Unit tests for the RWKV model components
│
├── test_classical_*.py                 # Scripts to train/test classical RWKV on different datasets
├── test_quantum_*.py                   # Scripts to train/test quantum RWKV on different datasets
│
├── run_all_timeseries_tasks.py         # Main script to run all time series experiments
│
├── results/                            # General results directory (may contain summaries or plots)
├── results_<dataset>_classical/        # Results for classical model on <dataset> (e.g., plots, metrics)
└── results_<dataset>_quantum/          # Results for quantum model on <dataset> (e.g., plots, metrics)

```

## Key Components

*   **`rwkv.py`**: Contains the implementation of the standard RWKV language model.
*   **`quantum_rwkv.py`**: Implements the quantum-enhanced version of the RWKV model, potentially leveraging quantum circuits for certain components.
*   **`test_classical_*.py` / `test_quantum_*.py`**: A suite of test scripts designed to evaluate the performance of the classical and quantum RWKV models respectively. Each script typically focuses on a specific time series dataset, such as:
    *   ARMA
    *   Chaotic Logistic Map
    *   Damped Oscillation
    *   Noisy Damped Oscillation
    *   Piecewise Regime
    *   Sawtooth Wave
    *   Square/Triangle Wave
    *   Trend, Seasonality, Noise
    *   General Waveforms
*   **`test_rwkv.py`**: Contains unit tests to verify the core functionalities of the `RWKVModel` implementation, such as forward passes and state management.
*   **`run_all_timeseries_tasks.py`**: A utility script to execute all defined time series prediction tasks for both classical and quantum models. This is likely the main entry point for running experiments.
*   **`results_*` directories**: These directories store the outputs of the experiments, including performance metrics, generated plots, and model checkpoints for each dataset and model type (classical or quantum).
*   **`comparison_summary.csv`**: A CSV file that aggregates and summarizes the key performance metrics from different experiments, allowing for a direct comparison between the classical and quantum approaches.
*   **`quantum_circuit.png` / `quantum_circuit_high_level.png`**: Images depicting the quantum circuit design used in the `quantum_rwkv.py` model.

## Setup

(Please add instructions here on how to set up the environment, e.g., Python version, required libraries, and how to install them. For example: `pip install -r requirements.txt`)

## Usage

To run all the time series prediction experiments and generate results:

```bash
python run_all_timeseries_tasks.py
```

Individual test scripts can also be run:
```bash
python test_classical_waveform.py
python test_quantum_waveform.py
# etc.
```

Results, including plots and metrics, will be saved in the respective `results_*` directories. The `comparison_summary.csv` file will be updated with the outcomes.

## Visualizations

The project includes visualizations such as:
*   `quantum_circuit.png`: A detailed view of the quantum circuit.
*   `quantum_circuit_high_level.png`: A simplified, high-level diagram of the quantum circuit.
*   Various `*.png` files in the root directory and `results_*` directories, showcasing prediction comparisons (e.g., `waveform_prediction_comparison_quantum_rwkv.png`).

## Contributing

(Optional: Add guidelines for contributing to the project.)

## License

(Optional: Specify the license for the project.) 
