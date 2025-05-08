import torch
from rwkv import ModelConfig, RWKVModel

def test_model_forward():
    # Test parameters
    batch_size = 2
    seq_len = 10
    vocab_size = 50
    n_embd_test = 32 # Must be divisible by n_head_test
    n_head_test = 4
    n_layer_test = 3
    
    # Create a model configuration
    config = ModelConfig(
        n_embd=n_embd_test,
        n_head=n_head_test,
        n_layer=n_layer_test,
        vocab_size=vocab_size,
        block_size=seq_len + 10, # Larger than seq_len for testing
        n_intermediate=n_embd_test * 2, # Smaller intermediate for faster test
        layer_norm_epsilon=1e-5
    )
    
    print(f"Model Config: {config}\n")
    
    # Instantiate the model
    model = RWKVModel(config)
    model.eval() # Set to evaluation mode
    
    print(f"RWKVModel instantiated successfully.\n")
    
    # Create dummy input token indices (Batch, SeqLen)
    dummy_idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Dummy input idx shape: {dummy_idx.shape}\n")
    
    # --- First forward pass (no initial states) ---
    print("--- First forward pass (states=None) ---")
    logits1, states1 = model(dummy_idx, states=None)
    
    print(f"Logits shape: {logits1.shape}")
    assert logits1.shape == (batch_size, seq_len, vocab_size)
    
    print(f"Number of layers for states: {len(states1)}")
    assert len(states1) == n_layer_test
    
    for i, layer_state in enumerate(states1):
        wkv_state, cm_state = layer_state
        aa, bb, pp = wkv_state
        print(f"  Layer {i} States:")
        print(f"    WKV state (aa, bb, pp) shapes: {aa.shape}, {bb.shape}, {pp.shape}")
        print(f"    Channel Mix state shape: {cm_state.shape}")
        assert aa.shape == (batch_size, n_embd_test)
        assert bb.shape == (batch_size, n_embd_test)
        assert pp.shape == (batch_size, n_embd_test)
        assert cm_state.shape == (batch_size, n_embd_test)
    print("\nSuccessfully completed first forward pass and state checks.\n")

    # --- Second forward pass (using states from the first pass) ---
    # This simulates processing the next chunk of a sequence, or a single next token
    print("--- Second forward pass (using states from first pass) ---")
    # Let's create a new dummy input, e.g., for the next single token
    next_dummy_idx = torch.randint(0, vocab_size, (batch_size, 1)) # Next token
    print(f"Next dummy input idx shape: {next_dummy_idx.shape}\n")
    
    logits2, states2 = model(next_dummy_idx, states=states1)
    
    print(f"Logits shape (second pass): {logits2.shape}")
    assert logits2.shape == (batch_size, 1, vocab_size) # Output for the single next token
    
    print(f"Number of layers for states (second pass): {len(states2)}")
    assert len(states2) == n_layer_test
    
    for i, layer_state in enumerate(states2):
        wkv_state, cm_state = layer_state
        aa, bb, pp = wkv_state
        print(f"  Layer {i} States (second pass):")
        print(f"    WKV state (aa, bb, pp) shapes: {aa.shape}, {bb.shape}, {pp.shape}")
        print(f"    Channel Mix state shape: {cm_state.shape}")
        assert aa.shape == (batch_size, n_embd_test)
        assert bb.shape == (batch_size, n_embd_test)
        assert pp.shape == (batch_size, n_embd_test)
        assert cm_state.shape == (batch_size, n_embd_test)
    print("\nSuccessfully completed second forward pass with states and checks.\n")
    
    print("All tests passed!")

if __name__ == '__main__':
    test_model_forward() 