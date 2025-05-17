import importlib
import traceback

# List of (module_name, function_name) pairs to run
TASKS = [
    ("test_classical_learning", "run_classical_simple_learning_test"),
    ("test_classical_learning", "run_classical_waveform_prediction_test"),
    ("test_classical_arma", "run_classical_arma_prediction"),
    ("test_classical_chaotic_logistic", "run_classical_chaotic_logistic_prediction"),
    ("test_classical_damped_oscillation", "run_classical_damped_oscillation_prediction"),
    ("test_classical_noisy_damped_oscillation", "run_classical_noisy_damped_oscillation_prediction"),
    ("test_classical_piecewise_regime", "run_classical_piecewise_regime_prediction"),
    ("test_classical_sawtooth_wave", "run_classical_sawtooth_wave_prediction"),
    ("test_classical_waveform", "run_classical_waveform_prediction_test"),
    ("test_classical_square_triangle_wave", "run_classical_square_triangle_wave_prediction"),
    ("test_classical_trend_seasonality_noise", "run_classical_trend_seasonality_noise_prediction"),
    ("test_quantum_learning", "run_simple_learning_test"),
    ("test_quantum_learning", "run_quantum_waveform_learning_test"),
    ("test_quantum_arma", "run_quantum_arma_prediction"),
    ("test_quantum_chaotic_logistic", "run_quantum_chaotic_logistic_prediction"),
    ("test_quantum_damped_oscillation", "run_quantum_damped_oscillation_prediction"),
    ("test_quantum_noisy_damped_oscillation", "run_quantum_noisy_damped_oscillation_prediction"),
    ("test_quantum_piecewise_regime", "run_quantum_piecewise_regime_prediction"),
    ("test_quantum_sawtooth_wave", "run_quantum_sawtooth_wave_prediction"),
    ("test_quantum_square_triangle_wave", "run_quantum_square_triangle_wave_prediction"),
    ("test_quantum_trend_seasonality_noise", "run_quantum_trend_seasonality_noise_prediction"),
    ("test_quantum_waveform", "run_waveform_prediction_test"),
]

SUCCESS = []
FAIL = []

for module_name, func_name in TASKS:
    print(f"\n===== Running {module_name}.{func_name} =====\n")
    try:
        mod = importlib.import_module(module_name)
        func = getattr(mod, func_name)
        func()
        SUCCESS.append(f"{module_name}.{func_name}")
    except Exception as e:
        print(f"[ERROR] {module_name}.{func_name} failed: {e}")
        traceback.print_exc()
        FAIL.append(f"{module_name}.{func_name}")

print("\n================ SUMMARY ================")
print(f"SUCCESS ({len(SUCCESS)}):")
for s in SUCCESS:
    print(f"  [OK] {s}")
print(f"\nFAIL ({len(FAIL)}):")
for f in FAIL:
    print(f"  [FAIL] {f}")
print("========================================\n") 