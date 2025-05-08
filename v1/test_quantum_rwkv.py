import torch
# Make sure to import from the new quantum_rwkv.py file
from quantum_rwkv import ModelConfig, QuantumRWKVModel 

def test_quantum_model_forward():
    # Test parameters
    batch_size = 2
    seq_len = 5 # Shorter sequence for potentially slower quantum simulation
    vocab_size = 50
    n_embd_test = 32 # Must be divisible by n_head_test
    n_head_test = 4  # For RWKVTimeMixing
    n_layer_test = 2 # Fewer layers for faster test
    
    # Quantum-specific parameters for ModelConfig
    n_qubits_test = 4 
    q_depth_test = 1 # Shallow depth for faster test
    
    # Create a model configuration for the quantum model
    config = ModelConfig(
        n_embd=n_embd_test,
        n_head=n_head_test, # Classical TimeMixing still needs this
        n_layer=n_layer_test,
        vocab_size=vocab_size,
        block_size=seq_len + 10, 
        n_intermediate=n_embd_test * 2, # Less relevant for QFFN, but part of config
        layer_norm_epsilon=1e-5,
        n_qubits=n_qubits_test,
        q_depth=q_depth_test
    )
    
    print(f"Quantum Model Config: {config}\n")
    
    # Instantiate the quantum model
    # This might take a bit longer due to PennyLane VQC initialization
    try:
        model = QuantumRWKVModel(config)
        model.eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error instantiating QuantumRWKVModel: {e}")
        print("Ensure PennyLane is installed and configured correctly.")
        print("Try: pip install pennylane pennylane-lightning[torch] (or other backends)")
        raise
        
    print(f"QuantumRWKVModel instantiated successfully.\n")
    
    # Create dummy input token indices (Batch, SeqLen)
    dummy_idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Dummy input idx shape: {dummy_idx.shape}\n")
    
    # --- First forward pass (no initial states) ---
    print("--- First forward pass (states=None) for Quantum Model ---")
    try:
        logits1, states1 = model(dummy_idx, states=None)
    except Exception as e:
        print(f"Error during first forward pass of QuantumRWKVModel: {e}")
        raise

    print(f"Logits shape (Quantum): {logits1.shape}")
    assert logits1.shape == (batch_size, seq_len, vocab_size)
    
    print(f"Number of layers for states (Quantum): {len(states1)}")
    assert len(states1) == n_layer_test
    
    for i, layer_state in enumerate(states1):
        wkv_state, cm_state = layer_state
        aa, bb, pp = wkv_state
        print(f"  Layer {i} States (Quantum):")
        print(f"    WKV state (aa, bb, pp) shapes: {aa.shape}, {bb.shape}, {pp.shape}")
        print(f"    Channel Mix state shape: {cm_state.shape}")
        assert aa.shape == (batch_size, n_embd_test)
        assert bb.shape == (batch_size, n_embd_test)
        assert pp.shape == (batch_size, n_embd_test)
        assert cm_state.shape == (batch_size, n_embd_test)
    print("\nSuccessfully completed first forward pass and state checks (Quantum).\n")

    # --- Second forward pass (using states from the first pass) ---
    print("--- Second forward pass (using states from first pass) for Quantum Model ---")
    next_dummy_idx = torch.randint(0, vocab_size, (batch_size, 1)) # Next single token
    print(f"Next dummy input idx shape (Quantum): {next_dummy_idx.shape}\n")
    
    try:
        logits2, states2 = model(next_dummy_idx, states=states1)
    except Exception as e:
        print(f"Error during second forward pass of QuantumRWKVModel: {e}")
        raise
        
    print(f"Logits shape (Quantum, second pass): {logits2.shape}")
    assert logits2.shape == (batch_size, 1, vocab_size)
    
    print(f"Number of layers for states (Quantum, second pass): {len(states2)}")
    assert len(states2) == n_layer_test
    
    for i, layer_state in enumerate(states2):
        wkv_state, cm_state = layer_state
        aa, bb, pp = wkv_state
        print(f"  Layer {i} States (Quantum, second pass):")
        print(f"    WKV state (aa, bb, pp) shapes: {aa.shape}, {bb.shape}, {pp.shape}")
        print(f"    Channel Mix state shape: {cm_state.shape}")
        assert aa.shape == (batch_size, n_embd_test)
        assert bb.shape == (batch_size, n_embd_test)
        assert pp.shape == (batch_size, n_embd_test)
        assert cm_state.shape == (batch_size, n_embd_test)
    print("\nSuccessfully completed second forward pass with states and checks (Quantum).\n")
    
    print("All quantum model tests passed!")

if __name__ == '__main__':
    test_quantum_model_forward() 