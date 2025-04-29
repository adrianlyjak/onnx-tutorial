from kokoro import KModel
import time
import torch
import onnx
from torch.export import Dim

model = KModel(None, None).eval()

print(model)

# Constants from the model
batch_size = 1
context_length = 512  # Based on position embeddings size
style_dim = 256  # Total style dimension (128 + 128) based on ref_s splitting
embedding_dim = 128  # From the model architecture

# Create dummy inputs with proper padding to context_length
dummy_seq_length = 12
input_ids = torch.zeros(
    (batch_size, context_length), dtype=torch.long
)  # Initialize with padding
input_ids[0, :dummy_seq_length] = torch.LongTensor(
    [0] + [1] * (dummy_seq_length - 2) + [0]
)  # Add content

# Style reference tensor
ref_s = torch.randn(batch_size, style_dim)  # [batch_size, 256]

# Test the inputs
start_time = time.time()
print("trying dummy inputs")
print(f"input_ids shape: {input_ids.shape}")
print(f"ref_s shape: {ref_s.shape}")
# Use named arguments to match the model's expectations
output = model(phonemes=input_ids, ref_s=ref_s)
print(f"time for dummy inputs: {time.time() - start_time}")

# Create inputs tuple for ONNX with named arguments
dummy_inputs = ({"phonemes": input_ids, "ref_s": ref_s},)

print("Starting ONNX export...")
start_time = time.time()

try:
    # Define dynamic dimensions using torch.export.Dim
    batch = Dim("batch", min=1, max=32)  # Allow batch size 1-32
    seq = Dim(
        "sequence", min=1, max=context_length
    )  # Allow sequence length up to context_length

    torch.onnx.export(
        model,
        dummy_inputs,
        "kokoro.onnx",
        opset_version=20,
        dynamic_shapes={
            "phonemes": {0: batch, 1: seq},
            "ref_s": {0: batch},
        },
        dynamo=True,
        report=True,
        input_names=["phonemes", "ref_s"],
        output_names=["output"],
    )
    print(f"Export completed in {time.time() - start_time:.2f} seconds")

    # Try to load and check the model
    print("Validating exported model...")
    model = onnx.load("kokoro.onnx")
    onnx.checker.check_model(model)
    print("Model validation successful")

except Exception as e:
    print(f"Export failed after {time.time() - start_time:.2f} seconds")
    print(f"Error: {str(e)}")
    raise
