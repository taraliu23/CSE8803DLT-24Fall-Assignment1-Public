import torch
import torch.nn as nn

d_model = 64
n_head = 8
dropout = 0.1

batch_size = 2
seq_len = 16


class CausalSelfAttention(nn.Module):
	"""
	A vanilla multi-head masked self-attention layer with a projection at the end.
	"""

	def __init__(self, d_model=64, n_head=8, dropout=0.1):
		super().__init__()
		assert d_model % n_head == 0
		self.q_proj = nn.Linear(d_model, d_model)
		self.k_proj = nn.Linear(d_model, d_model)
		self.v_proj = nn.Linear(d_model, d_model)
		self.o_proj = nn.Linear(d_model, d_model)
		self.residual_dropout = nn.Dropout(dropout)

		self.n_head = n_head

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Implement the multi-head masked self-attention layer.
		You should not use network modules other than what defined in the __init__ function.

		Input & output shape: (batch_size, sequence_length, d_model)
		"""
		batch_size, seq_len, d_model = x.shape
		print(f"Input shape: {x.shape}")  

		# Step 1: Linear projections to get queries, keys, and values
		Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
		K = self.k_proj(x)  # (batch_size, seq_len, d_model)
		V = self.v_proj(x)  # (batch_size, seq_len, d_model)
		print(f"Q, K, V shapes after projection: {Q.shape}, {K.shape}, {V.shape}")

		# Step 2: Calculate head_dim here (instead of relying on __init__ function)
		head_dim = d_model // self.n_head
		scale = head_dim ** -0.5  # Calculate scale in forward method

		# Step 3: Reshape and split into heads
		Q = Q.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)  # (batch_size, n_head, seq_len, head_dim)
		K = K.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)  # (batch_size, n_head, seq_len, head_dim)
		V = V.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)  # (batch_size, n_head, seq_len, head_dim)
		print(f"Q, K, V shapes after splitting into heads: {Q.shape}, {K.shape}, {V.shape}")

		# Step 4: Scaled dot-product attention
		scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch_size, n_head, seq_len, seq_len)
		print(f"Attention scores shape: {scores.shape}")

		# Step 5: Apply causal mask (prevent attending to future tokens)
		causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)  # Lower triangular mask
		scores = scores.masked_fill(causal_mask == 0, float('-inf'))  # Apply mask
		print(f"Attention scores after masking shape: {scores.shape}")

		# Step 6: Softmax over the last dimension (seq_len_k)
		attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, n_head, seq_len, seq_len)
		print(f"Attention weights shape: {attn_weights.shape}")

		# Step 7: Compute the attention output
		attn_output = torch.matmul(attn_weights, V)  # (batch_size, n_head, seq_len, head_dim)
		print(f"Attention output per head shape: {attn_output.shape}")

		# Step 8: Concatenate heads and apply final linear projection
		attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_len, n_head, head_dim)
		attn_output = attn_output.view(batch_size, seq_len, d_model)  # (batch_size, seq_len, d_model)
		print(f"Attention output after concatenating heads shape: {attn_output.shape}")

		# Step 9: Apply output projection and dropout
		output = self.o_proj(attn_output)  
		output = self.residual_dropout(output)
		print(f"Final output shape: {output.shape}")

		return output


		# --- TODO: end of your code ---
		# raise NotImplementedError

# Load the model
causal_self_attention = CausalSelfAttention(d_model=d_model, n_head=n_head, dropout=dropout)
causal_self_attention.load_state_dict(torch.load("causal_self_attention.pt"))


# Test the model
x = torch.load("x.pt")

y = causal_self_attention(x)
y_expected = torch.load("y.pt")

assert y.shape == y_expected.shape, f"Expected shape: {y_expected.shape}, but got: {y.shape}"
assert torch.sum(torch.abs(y - y_expected) < 1e-5) > 0.78 * batch_size * seq_len * d_model, "The output is incorrect."

print("The output is correct.")
