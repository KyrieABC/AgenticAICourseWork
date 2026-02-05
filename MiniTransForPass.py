#Main PyTorch tensor library for GPU/CPU computation
import torch
import math
#Neural network module containing layers, activation
import torch.nn as nn
#Functional interface for activation functions, loss functions
import torch.nn.functional as F

class MiniTransformer(nn.Module):
    """
    A small-scale implementation of the Transformer Architecture
    Original "Attention is all you need" with reduced dimension
    """
    def __init__(self,vocab_size,d_model=64,n_heads=2,ff_dim=128,max_len=100,n_layers=2):
        #Initialize parent nn.Module class, set up hooks, parameters registration
        super().__init__()
        
        self.d_model=d_model
        
        #Token embedding layer
        #Create Lookup table that maps token indices to dense vectors
        #Each row represents a trainable vector for one vocab item
        self.token_embedding=nn.Embedding(vocab_size,d_model)
        
        #Positional Encoding:
        #Fixed (non-learnable) encoding that adds sequence position information
        #Since Transformers process all tokens in parallel, they need explicity position info
        self.positional_encoding=self.create_positional_encoding(max_len, d_model)
        
        #Encoder layer stack
        #Create a list of n_layers identical Transformer encoder layers
        #nn.ModuleList ensures all parameters are registered with PyTorch
        #Each Layer processes the sequence with self-attention then feed-forward
        self.encoder_layers=nn.ModuleList([
            TransformerEncoderLayer(d_model,n_heads,ff_dim) #Create one layer
            for _ in range(n_layers) #Repeat n_layers times
        ])
        
        #Final Layer Normalization
        #Normalizes activation across the feature dimension (d_model dimension)
        #Stabilizes training, reduces sensitivity to parameter initialization
        #Applied after all encoder layers, before final projection
        self.layer_norm=nn.LayerNorm(d_model)
        
        #Output Projection Layer
        #Linear transfermation from hidden dimension back to vocabulary size
        #Produce logits (unnormalized scores) for each vocabulary token
        #Used for next token prediction or classification
        self.output_projection=nn.Linear(d_model,vocab_size)
    
    def create_positional_encoding(self,max_len,d_model):
        """
        Generates fixed sinusoidal positional encodings as per original Transformer paper.
        Each position gets a unique encoding vector that the model can use to learn
        relative or absolute position information.
        
        Mathematical formula for position pos and dimension i:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))  # Even dimensions
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) # Odd dimensions
        
        Args:
            max_len: Maximum sequence length to precompute encodings for
            d_model: Dimensionality of the encoding vectors
            
        Returns:
            Tensor of shape [1, max_len, d_model] - batch dimension added for broadcasting
        """
        
        #Initialize zero matrix to hold positional encodings
        #Shape: [max_len,d_model] - each row is encoding for one position
        pe=torch.zeros(max_len,d_model)
        
        #Create positional indices (0,max_len-1)
        #Unsqueeze adds a dimension: [max_len]->[max_len,1] for broadcasting
        position=torch.arrange(0,max_len).unsqueeze(1)
        
        #Precompute division terms for sinusodial functions
        #Create tensor of even indicesL [0,2,4,...]
        #Compute: 1000*(2i/d_model) using log for numerical stability
        #Actually computing: e^(2i*-log(10000)/d_model)=1/10000*(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            -(math.log(10000.0) / d_model))
        
        # Apply sine function to even dimensions (0, 2, 4, ...)
        # position * div_term: [max_len, 1] × [d_model/2] -> [max_len, d_model/2] via broadcasting
        # pe[:, 0::2]: Slice all rows, every other column starting from 0 (even columns)
        pe[:, 0::2] = torch.sin(position * div_term)  # Fill even columns with sine values
        
        # Apply cosine function to odd dimensions (1, 3, 5, ...)
        # Similar computation for odd columns
        pe[:, 1::2] = torch.cos(position * div_term)  # Fill odd columns with cosine values
        
        # Add batch dimension: [max_len, d_model] -> [1, max_len, d_model]
        # This allows broadcasting across batch dimension in forward pass
        # unsqueeze(0) adds dimension at position 0 (batch dimension)
        return pe.unsqueeze(0)
        
    def forward(self, input_ids):
        """
        Forward pass through the entire Mini Transformer.
        This is the main function called during inference and training.
        
        Args:
            input_ids: Integer tensor of token indices
                       Shape: [batch_size, seq_len]
                       Example: [[5, 23, 10, 0], [12, 8, 42, 1]] for batch_size=2, seq_len=4
            
        Returns:
            logits: Float tensor of unnormalized scores for each vocabulary token at each position
                    Shape: [batch_size, seq_len, vocab_size]
                    These can be converted to probabilities via softmax
        """
        # Get batch size and sequence length from input tensor shape
        # input_ids.shape returns torch.Size([batch_size, seq_len])
        batch_size, seq_len = input_ids.shape
        
        # ==================== STEP 1: TOKEN EMBEDDING ====================
        # Convert discrete token indices to continuous vector representations
        # self.token_embedding is a lookup table: returns vector for each token index
        # Shape transformation: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        # Each integer token index replaced with a d_model-dimensional float vector
        token_embeds = self.token_embedding(input_ids)
        
        # ==================== STEP 2: ADD POSITIONAL ENCODING ====================
        # Extract positional encodings for current sequence length
        # self.positional_encoding shape: [1, max_len, d_model]
        # Slice to get encodings for actual sequence length: [1, seq_len, d_model]
        pos_enc = self.positional_encoding[:, :seq_len, :]
        
        # Add positional encoding to token embeddings (element-wise addition)
        # pos_enc broadcasted from [1, seq_len, d_model] to [batch_size, seq_len, d_model]
        # This combines token semantic information with position information
        # Shape remains: [batch_size, seq_len, d_model]
        x = token_embeds + pos_enc
        
        # ==================== STEP 3: PROCESS THROUGH ENCODER LAYERS ====================
        # Sequentially pass through each Transformer encoder layer
        # Each layer applies self-attention then feed-forward with residual connections
        # Output of each layer becomes input to next layer
        for layer in self.encoder_layers:
            x = layer(x)  # Shape preserved: [batch_size, seq_len, d_model]
            # After each layer, x contains increasingly abstract representations
            # that incorporate information from across the sequence
        
        # ==================== STEP 4: FINAL LAYER NORMALIZATION ====================
        # Normalize the final hidden states across feature dimension (d_model)
        # Stabilizes values before final projection, helps training convergence
        # Shape: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model] (normalized)
        x = self.layer_norm(x)
        
        # ==================== STEP 5: OUTPUT PROJECTION ====================
        # Linear transformation from hidden dimension to vocabulary size
        # self.output_projection contains weight matrix [d_model, vocab_size] and bias [vocab_size]
        # Matrix multiplication: [batch_size, seq_len, d_model] × [d_model, vocab_size] 
        #                    -> [batch_size, seq_len, vocab_size]
        # Each position in sequence gets a score for each vocabulary token
        logits = self.output_projection(x)
        
        return logits  # Return unnormalized scores (logits)

class TransformerEncoderLayer(nn.Module):
    """
    A single Transformer encoder layer implementing:
    1. Multi-head self-attention with residual connection
    2. Position-wise feed-forward network with residual connection
    3. Layer normalization before each sub-layer (Pre-LN variant)
    
    This is the fundamental building block of the Transformer.
    """
    
    def __init__(self, d_model, n_heads, ff_dim):
        """
        Initialize a single encoder layer with all necessary components.
        
        Args:
            d_model: Hidden dimension size (consistent throughout model)
            n_heads: Number of attention heads
            ff_dim: Hidden dimension in feed-forward network
        """
        super().__init__()  # Initialize parent nn.Module
        
        # MULTI-HEAD SELF-ATTENTION MECHANISM
        # Allows each position to attend to all positions in the sequence
        # Multiple heads learn different types of attention patterns
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        
        # LAYER NORMALIZATION 1: Applied before self-attention (Pre-LN architecture)
        # Normalizes across the d_model dimension for each token independently
        # Stabilizes activations, helps with gradient flow
        self.norm1 = nn.LayerNorm(d_model)
        
        # LAYER NORMALIZATION 2: Applied before feed-forward network
        # Same purpose as norm1, applied after attention, before FFN
        self.norm2 = nn.LayerNorm(d_model)
        
        # POSITION-WISE FEED-FORWARD NETWORK
        # Applied independently to each token position
        # Two linear transformations with ReLU activation in between
        # Typically expands dimension (d_model -> ff_dim) then contracts back
        self.ffn = nn.Sequential(  # Sequential container applies layers in order
            # First linear layer: Project from d_model to ff_dim (typically larger)
            # Weight matrix shape: [d_model, ff_dim], bias shape: [ff_dim]
            nn.Linear(d_model, ff_dim),
            
            # ReLU activation: Introduces non-linearity, allows model to learn complex patterns
            # ReLU(x) = max(0, x) - simple, computationally efficient
            nn.ReLU(),  # Applied element-wise
            
            # Second linear layer: Project back from ff_dim to d_model
            # Weight matrix shape: [ff_dim, d_model], bias shape: [d_model]
            nn.Linear(ff_dim, d_model)
        )
        # Note: Original paper uses activation: max(0, xW1 + b1)W2 + b2
        
    def forward(self, x):
        """
        Forward pass through a single Transformer encoder layer.
        Implements: LayerNorm -> Self-Attention -> Residual -> LayerNorm -> FFN -> Residual
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
               Contains token representations from previous layer or embeddings
            
        Returns:
            Output tensor of same shape [batch_size, seq_len, d_model]
            Contains transformed representations with contextual information
        """
        # ============ RESIDUAL BLOCK 1: SELF-ATTENTION ============
        # Pre-LN architecture: Normalize BEFORE the sub-layer
        
        # STEP 1a: Layer normalization on input
        # Normalizes across feature dimension for each token
        # Reduces internal covariate shift, stabilizes training
        norm_x = self.norm1(x)  # Shape: [batch_size, seq_len, d_model]
        
        # STEP 1b: Multi-head self-attention
        # All three arguments are same tensor for self-attention
        # Query, Key, Value all come from normalized input
        # Attention mechanism computes weighted sum of values based on query-key similarity
        attn_output = self.self_attention(
            query=norm_x,   # What we're looking for
            key=norm_x,     # What we compare against
            value=norm_x    # What we aggregate
        )  # Shape: [batch_size, seq_len, d_model]
        
        # STEP 1c: Residual (skip) connection
        # Add original input x (not normalized) to attention output
        # Helps gradient flow in deep networks, prevents vanishing gradients
        # Preserves original information while adding transformed information
        x = x + attn_output  # Shape: [batch_size, seq_len, d_model]
        # Now x contains: original input + attention-transformed information
        
        # ============ RESIDUAL BLOCK 2: FEED-FORWARD NETWORK ============
        # Similar structure: Normalize -> Transform -> Add residual
        
        # STEP 2a: Layer normalization after attention
        # Normalize the combined (input + attention) representation
        norm_x = self.norm2(x)  # Shape: [batch_size, seq_len, d_model]
        
        # STEP 2b: Position-wise feed-forward network
        # Apply two linear layers with ReLU activation
        # Each token processed independently (no cross-token interaction here)
        ffn_output = self.ffn(norm_x)  # Shape: [batch_size, seq_len, d_model]
        
        # STEP 2c: Second residual connection
        # Add attention-output to FFN output
        x = x + ffn_output  # Shape: [batch_size, seq_len, d_model]
        # Now x contains: (input + attention) + feed-forward transformation
        
        return x  # Return transformed representations for next layer or final output

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in original Transformer paper.
    Implements: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Multiple heads allow model to jointly attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(self, d_model, n_heads):
        """
        Initialize multi-head attention with linear projections.
        
        Args:
            d_model: Total dimension of the model
            n_heads: Number of parallel attention heads
        """
        super().__init__()
        
        # Store configuration parameters as instance variables
        self.n_heads = n_heads  # Number of attention heads
        self.d_model = d_model  # Total model dimension
        
        # Ensure d_model is divisible by n_heads
        # Each head gets d_model // n_heads dimensions
        self.head_dim = d_model // n_heads  # Dimension per head
        
        # Assert that dimensions divide evenly
        # If not true, we would have incomplete heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # QUERY PROJECTION: Linear layer to compute query representations
        # Learns weight matrix W_q of shape [d_model, d_model]
        # Projects input to same dimension, then split across heads
        self.w_q = nn.Linear(d_model, d_model)
        
        # KEY PROJECTION: Linear layer to compute key representations
        # Learns weight matrix W_k of shape [d_model, d_model]
        # Different parameters from query projection
        self.w_k = nn.Linear(d_model, d_model)
        
        # VALUE PROJECTION: Linear layer to compute value representations
        # Learns weight matrix W_v of shape [d_model, d_model]
        # Different parameters from query and key projections
        self.w_v = nn.Linear(d_model, d_model)
        
        # OUTPUT PROJECTION: Linear layer to combine heads
        # Learns weight matrix W_o of shape [d_model, d_model]
        # Combines information from all heads back to original dimension
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention across multiple heads.
        
        Args:
            query: Tensor of shape [batch_size, query_seq_len, d_model]
                   Contains representations we want to match against keys
            key: Tensor of shape [batch_size, key_seq_len, d_model]
                 Contains representations we compare queries against
            value: Tensor of shape [batch_size, value_seq_len, d_model]
                   Contains information we want to aggregate
            mask: Optional tensor for masking certain attention scores
                  Shape: [batch_size, 1, seq_len, seq_len] or similar
                  Used for padding masking or causal (look-ahead) masking
                  
        Returns:
            attention_output: Tensor of shape [batch_size, query_seq_len, d_model]
        """
        # Extract batch size and sequence lengths from input tensors
        # query.shape returns torch.Size([batch_size, query_seq_len, d_model])
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key.shape  # key_seq_len (usually same as query_len for self-attention)
        _, value_len, _ = value.shape  # value_seq_len (usually same as key_len)
        
        # ============ STEP 1: LINEAR PROJECTION AND HEAD SPLITTING ============
        # Project inputs to query, key, value representations
        # Then reshape to separate heads for parallel computation
        
        # QUERY projection and reshaping:
        # 1. Linear projection: [batch_size, query_len, d_model] -> same shape
        # 2. View: Reshape to [batch_size, query_len, n_heads, head_dim]
        #    This groups features into n_heads separate groups, each with head_dim features
        Q = self.w_q(query)  # Linear projection
        Q = Q.view(batch_size, query_len, self.n_heads, self.head_dim)  # Reshape for heads
        
        # KEY projection and reshaping (same process as query):
        K = self.w_k(key)
        K = K.view(batch_size, key_len, self.n_heads, self.head_dim)
        
        # VALUE projection and reshaping (same process):
        V = self.w_v(value)
        V = V.view(batch_size, value_len, self.n_heads, self.head_dim)
        
        # ============ STEP 2: PREPARE FOR BATCHED MATRIX MULTIPLICATION ============
        # Transpose to bring head dimension to position 1 (after batch)
        # From: [batch_size, seq_len, n_heads, head_dim]
        # To:   [batch_size, n_heads, seq_len, head_dim]
        # This allows us to compute attention per head using batched matmul
        
        Q = Q.transpose(1, 2)  # Swap dimensions 1 and 2
        K = K.transpose(1, 2)  # Swap dimensions 1 and 2
        V = V.transpose(1, 2)  # Swap dimensions 1 and 2
        # Now shape: [batch_size, n_heads, seq_len, head_dim]
        
        # ============ STEP 3: COMPUTE ATTENTION SCORES ============
        # Compute Q • K^T (dot product between queries and keys)
        # This measures similarity/compatibility between query and key positions
        
        # K.transpose(-2, -1): Transpose last two dimensions
        # K shape: [batch_size, n_heads, key_len, head_dim]
        # After transpose: [batch_size, n_heads, head_dim, key_len]
        # matmul(Q, K^T): [batch_size, n_heads, query_len, head_dim] × 
        #                 [batch_size, n_heads, head_dim, key_len]
        # Result: [batch_size, n_heads, query_len, key_len]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # SCALE ATTENTION SCORES: Divide by sqrt(head_dim)
        # This prevents softmax from having extremely small gradients
        # When head_dim is large, dot products can become large in magnitude
        # Scaling keeps values in a range where softmax has reasonable gradients
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # ============ STEP 4: APPLY MASK (IF PROVIDED) ============
        # Masking prevents attention to certain positions
        # Common masks: Padding mask (ignore padding tokens), Causal mask (no looking ahead)
        
        if mask is not None:
            # mask should have shape that can be broadcast to attention_scores
            # Typically: [batch_size, 1, 1, key_len] or [batch_size, 1, query_len, key_len]
            # mask == 0 positions get -inf (completely ignored after softmax)
            # mask == 1 positions keep their original scores
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
            # float('-inf') ensures e^-inf = 0 in softmax
        
        # ============ STEP 5: SOFTMAX TO GET ATTENTION PROBABILITIES ============
        # Convert scores to probabilities that sum to 1 for each query
        # softmax along last dimension (key_len dimension)
        # This gives distribution over keys for each query position
        attention_probs = F.softmax(attention_scores, dim=-1)
        # Shape: [batch_size, n_heads, query_len, key_len]
        # attention_probs[i, h, j, k] = probability that query j attends to key k in head h of batch i
        
        # ============ STEP 6: WEIGHTED SUM OF VALUES ============
        # Multiply attention probabilities by values
        # For each query position, compute weighted average of values
        # Weight = attention probability for that key
        
        # attention_probs: [batch_size, n_heads, query_len, key_len]
        # V: [batch_size, n_heads, value_len, head_dim] (value_len usually equals key_len)
        # matmul: [batch_size, n_heads, query_len, key_len] × [batch_size, n_heads, value_len, head_dim]
        # Result: [batch_size, n_heads, query_len, head_dim]
        attention_output = torch.matmul(attention_probs, V)
        
        # ============ STEP 7: COMBINE HEADS BACK TOGETHER ============
        # Need to combine the n_heads separate outputs into single d_model-dimensional output
        
        # STEP 7a: Transpose to bring head dimension back to position 2
        # From: [batch_size, n_heads, query_len, head_dim]
        # To:   [batch_size, query_len, n_heads, head_dim]
        attention_output = attention_output.transpose(1, 2)
        
        # STEP 7b: Make tensor contiguous in memory for view() operation
        # transpose() may create non-contiguous tensor, view() requires contiguous
        attention_output = attention_output.contiguous()
        
        # STEP 7c: Reshape to combine head_dim * n_heads back into d_model
        # View as: [batch_size, query_len, d_model]
        # This concatenates the head_dim-dimensional outputs from each head
        attention_output = attention_output.view(batch_size, query_len, self.d_model)
        
        # ============ STEP 8: FINAL LINEAR PROJECTION ============
        # Apply output projection to combine information from all heads
        # Learns to weight different heads' contributions appropriately
        output = self.w_o(attention_output)  # Shape: [batch_size, query_len, d_model]
        
        return output

# ============================================================================
# COMPREHENSIVE USAGE EXAMPLE WITH DETAILED EXPLANATIONS
# ============================================================================

def run_comprehensive_example():
    """
    Demonstrate complete workflow from initialization to forward pass
    with detailed explanations at each step.
    """
    
    print("=" * 70)
    print("MINI TRANSFORMER FORWARD PASS - COMPLETE DEMONSTRATION")
    print("=" * 70)
    
    # ==================== STEP 1: SET HYPERPARAMETERS ====================
    print("\n1. SETTING MODEL HYPERPARAMETERS:")
    VOCAB_SIZE = 10000   # Vocabulary: 10,000 unique tokens (typical for small tasks)
    BATCH_SIZE = 4       # Process 4 sequences in parallel (batch dimension)
    SEQ_LEN = 32         # Each sequence has 32 tokens
    print(f"   Vocabulary size: {VOCAB_SIZE:,} tokens")
    print(f"   Batch size: {BATCH_SIZE} sequences")
    print(f"   Sequence length: {SEQ_LEN} tokens per sequence")
    
    # ==================== STEP 2: INITIALIZE MODEL ====================
    print("\n2. INITIALIZING MINI TRANSFORMER MODEL:")
    
    # Create model instance with specified dimensions
    model = MiniTransformer(
        vocab_size=VOCAB_SIZE,  # Must match actual vocabulary
        d_model=64,      # Hidden dimension: 64 features per token
        n_heads=2,       # 2 parallel attention heads (each gets 32 dimensions)
        ff_dim=128,      # Feed-forward hidden: 128 (2x d_model)
        max_len=512,     # Maximum sequence length for positional encoding
        n_layers=2       # Stack 2 encoder layers
    )
    
    print(f"   Model architecture:")
    print(f"   - Embedding dimension: {model.d_model}")
    print(f"   - Attention heads: {model.encoder_layers[0].self_attention.n_heads}")
    print(f"   - Layers: {len(model.encoder_layers)}")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== STEP 3: CREATE SAMPLE INPUT ====================
    print("\n3. CREATING SAMPLE INPUT DATA:")
    
    # Create dummy input: Random token indices
    # torch.randint generates random integers in range [0, VOCAB_SIZE)
    # Shape: [BATCH_SIZE, SEQ_LEN] = [4, 32]
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    print(f"   Input tensor shape: {input_ids.shape}")
    print(f"   Input values (first 3 tokens of first sequence): {input_ids[0, :3].tolist()}")
    print(f"   Each value is a token index (0 to {VOCAB_SIZE-1})")
    
    # ==================== STEP 4: SET MODEL TO EVALUATION MODE ====================
    print("\n4. SETTING MODEL TO EVALUATION MODE:")
    
    # Switch model to evaluation mode (affects dropout, batch norm if present)
    model.eval()  # Disables dropout, uses running statistics for batch norm
    
    print("   Model.eval() called - dropout/batchnorm layers behave differently")
    print("   No gradient computation during forward pass (faster, less memory)")
    
    # ==================== STEP 5: PERFORM FORWARD PASS ====================
    print("\n5. PERFORMING FORWARD PASS (INFERENCE):")
    
    # torch.no_grad() context manager disables gradient computation
    # Reduces memory usage, speeds up computation for inference
    with torch.no_grad():  # All operations inside have requires_grad=False
        print("   Entered torch.no_grad() context")
        print("   Starting forward propagation through network...")
        
        # Call the model's forward method (model(input_ids) calls model.forward(input_ids))
        logits = model(input_ids)
        
        print("   Forward pass completed successfully!")
    
    # ==================== STEP 6: ANALYZE OUTPUT ====================
    print("\n6. ANALYZING MODEL OUTPUT:")
    
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Interpretation: [batch_size={logits.shape[0]}, ", end="")
    print(f"seq_len={logits.shape[1]}, vocab_size={logits.shape[2]}]")
    print(f"   For each of {BATCH_SIZE} sequences, for each of {SEQ_LEN} positions,")
    print(f"   we have {VOCAB_SIZE} scores (logits), one for each vocabulary token")
    
    # ==================== STEP 7: CONVERT LOGITS TO PROBABILITIES ====================
    print("\n7. CONVERTING LOGITS TO PROBABILITIES:")
    
    # Apply softmax along vocabulary dimension (last dimension)
    # softmax converts logits to probabilities that sum to 1 for each position
    # dim=-1 means apply along last dimension (vocabulary dimension)
    probs = F.softmax(logits, dim=-1)
    
    print(f"   Probabilities shape: {probs.shape} (same as logits)")
    
    # Check that probabilities sum to 1 for first token of first sequence
    prob_sum = probs[0, 0, :].sum().item()  # Sum over vocabulary for position 0, sequence 0
    print(f"   Sum of probabilities for first token: {prob_sum:.6f}")
    print(f"   (Should be very close to 1.0, minus small floating point errors)")
    
    # ==================== STEP 8: GET PREDICTIONS ====================
    print("\n8. MAKING PREDICTIONS (GREEDY DECODING):")
    
    # Greedy decoding: Choose token with highest probability at each position
    # torch.argmax returns indices of maximum values along specified dimension
    predicted_tokens = torch.argmax(logits, dim=-1)  # dim=-1: max over vocabulary
    
    print(f"   Predicted tokens shape: {predicted_tokens.shape}")
    print(f"   Interpretation: [batch_size={predicted_tokens.shape[0]}, ", end="")
    print(f"seq_len={predicted_tokens.shape[1]}]")
    print(f"   For each sequence, we predicted {SEQ_LEN} token indices")
    print(f"   First 5 predicted tokens in first sequence: {predicted_tokens[0, :5].tolist()}")
    
    # ==================== STEP 9: UNDERSTANDING ATTENTION MECHANISM ====================
    print("\n9. UNDERSTANDING THE ATTENTION MECHANISM:")
    print("   During forward pass, each encoder layer performs:")
    print("   a) Multi-head attention: Each head computes attention scores")
    print("      Q•K^T / sqrt(d_k) -> softmax -> attention weights")
    print("   b) Weighted sum: Attention weights × Values")
    print("   c) Head combination: Concatenate all heads, project with W_o")
    
    # ==================== STEP 10: VISUALIZING DATA FLOW ====================
    print("\n10. DATA FLOW THROUGH THE MODEL:")
    print("    Input IDs [4, 32]")
    print("        ↓ Token Embedding")
    print("    Token Embeddings [4, 32, 64]")
    print("        + Positional Encoding")
    print("    With Position Info [4, 32, 64]")
    print("        ↓ 2× Encoder Layers")
    print("    (Each: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual)")
    print("    Contextual Representations [4, 32, 64]")
    print("        ↓ Final LayerNorm")
    print("    Normalized Representations [4, 32, 64]")
    print("        ↓ Output Projection")
    print("    Logits [4, 32, 10000] ← FINAL OUTPUT")
    
    return model, input_ids, logits, predicted_tokens

# ============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# ============================================================================

def analyze_model_components(model):
    """Print detailed information about each component of the model."""
    print("\n" + "=" * 70)
    print("MODEL COMPONENT ANALYSIS")
    print("=" * 70)
    
    print("\n1. TOKEN EMBEDDING LAYER:")
    print(f"   Weight matrix shape: {model.token_embedding.weight.shape}")
    print(f"   Interpretation: [vocab_size, d_model] = [{model.token_embedding.weight.shape[0]}, {model.token_embedding.weight.shape[1]}]")
    print(f"   Each of {model.token_embedding.weight.shape[0]:,} tokens has a {model.token_embedding.weight.shape[1]}-dimensional vector")
    
    print("\n2. POSITIONAL ENCODING:")
    print(f"   Shape: {model.positional_encoding.shape}")
    print(f"   Max sequence length: {model.positional_encoding.shape[1]}")
    print(f"   Fixed (non-learnable) sinusoidal encoding")
    
    print("\n3. ENCODER LAYERS:")
    for i, layer in enumerate(model.encoder_layers):
        print(f"   Layer {i+1}:")
        print(f"     - Attention heads: {layer.self_attention.n_heads}")
        print(f"     - Head dimension: {layer.self_attention.head_dim}")
        print(f"     - FFN hidden dim: {layer.ffn[0].out_features}")
        
    print("\n4. OUTPUT PROJECTION:")
    print(f"   Weight matrix shape: {model.output_projection.weight.shape}")
    print(f"   Bias shape: {model.output_projection.bias.shape}")
    print(f"   Projects from d_model={model.d_model} to vocab_size={model.output_projection.out_features}")

def demonstrate_attention_patterns(model, input_ids):
    """Demonstrate how attention works with a simple example."""
    print("\n" + "=" * 70)
    print("ATTENTION PATTERN DEMONSTRATION")
    print("=" * 70)
    
    # Get embeddings for a very short sequence
    short_input = input_ids[0:1, 0:5]  # [batch_size=1, seq_len=5]
    print(f"\nShort sequence (5 tokens): {short_input[0].tolist()}")
    
    # Get embeddings (without positional encoding)
    embeddings = model.token_embedding(short_input)  # [1, 5, 64]
    
    # Create simple attention mask (no masking)
    batch_size, seq_len, _ = embeddings.shape
    mask = torch.ones(batch_size, 1, seq_len, seq_len)  # All positions attend to all
    
    # Get first encoder layer's attention module
    attention_layer = model.encoder_layers[0].self_attention
    
    # Manually compute attention step by step
    Q = attention_layer.w_q(embeddings)
    K = attention_layer.w_k(embeddings)
    V = attention_layer.w_v(embeddings)
    
    print(f"\nQuery projection shape: {Q.shape}")
    print(f"Key projection shape: {K.shape}")
    print(f"Value projection shape: {V.shape}")
    
    # Reshape for multi-head
    Q = Q.view(batch_size, seq_len, attention_layer.n_heads, attention_layer.head_dim)
    K = K.view(batch_size, seq_len, attention_layer.n_heads, attention_layer.head_dim)
    V = V.view(batch_size, seq_len, attention_layer.n_heads, attention_layer.head_dim)
        