import torch
import torch.nn.functional as F

def sampling_playground(logits, temperature=1.0, top_k=0, top_p=0.0, reward_bias=None):
    """
    A playground to transform raw logits into a next-token choice.
    """
    # 1. Apply RLHF/Reward Bias
    # In RLHF, the model's 'policy' is nudged by a reward signal.
    if reward_bias is not None:
        logits = logits + reward_bias 

    # 2. Apply Temperature
    # Higher T = more flat/random; Lower T = more sharp/deterministic
    logits = logits / max(temperature, 1e-5)

    # 3. Apply Top-K
    if top_k > 0:
        # Find the top k values and set everything else to -infinity
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        # torch.topk(logist, tok_k)[0]: Finds the values of the k largest logits
        # [...,-1,None]:grab the smallest value within top k group, any logit smaller than the threshold is flagged.
        logits[indices_to_remove] = -float('Inf')
        #Set the logits to negative infinity so e^(-infinity)=0

    # 4. Apply Top-P (Nucleus Sampling)
    if top_p > 0.0:
        #sorted_logits(The value): actual raw logits from original output re-arranged in descending order
        #sorted_indices(The map): indices(numbers are from the unsorted version) of the sorted tensor
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # torch.cumsum: create a running total: [0.4,0.3,0.2,0.1]->[0.4,0.7,0.9,1.0]
        # F.softmax: Converts the scores ito percentage between 0.0 to 1.0
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Identify tokens to remove (those that exceed the cumulative probability p)
        # For every index with value > top_p, its binary mask will be set to true. Otherwise false.
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Ensure we keep at least the very first token
        # Shifts the mask(indices) to the right. Make sure the first word that pushes the sum over limit is kept
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        # Safety to make sure the first most likely word is never removed, even if p is extremely low.
        sorted_indices_to_remove[..., 0] = 0
        #Ex: [0.4,0.3,0.2,0.1]-cumsum->[0.4.0.7,0.9,1.0]
        #If p=0.5, then initial mask: [False(0), True(1), True(1), True(1)]
        #After shift:[0,0,1,1]
        
        #To set all indices where the mask=1 to negative infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')

    # 5. Final Selection
    probabilities = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probabilities, 1)
    
    return next_token, probabilities

# --- Example Usage ---
vocab = ["Apple", "Banana", "Cat", "Dog", "Elephant"]
# Simulated raw scores (logits) from a model
raw_logits = torch.tensor([2.0, 1.5, 5.0, 4.8, -1.0]) 

# Settings
token_id, probs = sampling_playground(
    raw_logits, 
    temperature=0.7, 
    top_k=3, 
    top_p=0.9,
    reward_bias=torch.tensor([0, 0, 0, -2.0, 0]) # Negative reward for 'Dog'
)

print(f"Probabilities: {probs.tolist()}")
print(f"Selected: {vocab[token_id.item()]}")