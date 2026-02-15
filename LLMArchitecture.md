# **Pre-training**
- **The stage where a model "learns to speak" by consuming massive amounts of raw text from the internet, books, and code**
- **Through self-supervised learning (The model doesn't need humans to label the data)**
- **Mathematically, the model is trying to maximize the likelihood of the next token ($w_t$) given the previous tokens**
1. Input (w1,w2,w3,....)
2. Processing: The model passes this through layers of a Transformer architecture.
3. Prediction: It "guesses" the most likely outcome
4. Correction: If incorrect (how to determine), weights adjusted by backpropagation

## Tokenization
- The process of breaking down a stream of text into smaller units called tokens.
1. Normalization
  - The text is cleaned. This might involve removing extra whitespace, converting to lowercase, or handling Unicode characters
2. Pre-tokenization
  - The text is split into rough boundaries, usually by whitespace or punctuation
  - *This prevents the model from accidentally merging parts of two different sentences or words*
3. Training the Tokenizer (The Subword Algorithm)
  - Byte Pair Encoding (BPE)
  - The tokenizer looks at a massive corpus and counts which characters or character sequences appear together most frequently
  - *It then merges the most frequent pairs into a single new token.*
4. Encoding/Mapping
  - Every unique token in the final vocabulary is assigned a unique integer (ID)
### **Mathematical Reasoning:** 
Minimize the sequence length while keeping the vocabulary size manageable.
### BPE:
*aaabdaaabac -> ZabdZabac (Z=aa) -> ZYdZYac (Y=ab) -> WdzWac (W=ZY=aaab)*

## Scaling Laws
  - Describe the predictable relationship between a model's performance (measured as "loss" or error) and the resources used to train it
N: The number of parameters (the "brain" size)
D: The number of tokens trained on (the "knowledge" volume)
C: The total floating-point operations (FLOPs) used for training
**Mathematical Reasoning**
Power Laws:
  L(N,D) = E + A / N^a + B / D^b
  - L: Cross Entropy Loss. The lower the better.
  - E: Irreducible loss - the inherent noise in language that no model can solve
  - A, B, a, b: constants determined by empirical testing
Chinchilla Breakthrough:
  C = 6 * N * D
  - C: compute
  *Every doubling of model size, you should double the amount of data*
  - Compute- Optimal: N, D should be scaled in equal proportion

## Curriculum Learning
  - Involves presenting the model with "simpler" data first and gradually increasing the complexity as the model’s "competence" grows
  - Converges Faster: It finds the "low-hanging fruit" of language patterns quickly
  - Avoids Poor Local Minima: It builds a stable foundation before being "confused" by outliers or noise.
  - Reduces Total Compute: It may reach the same level of performance as a standard model using fewer total FLOPs.
1. Difficulty Scoring (The "Teacher")
  - Rank by sentence length, vocabulary frequency, perplexity
2. Pacing Function (The "Schedule")
  - Define a function c(t) that determines the model's competence at time t.
  Linear Pacing (constant rate) vs Root Pacing (increase quickly at first, then slows down)
3. Sampling (The "Classroom")
  - During training, the model only samples data where the difficulty score d(x) <= c(t)
  - As t increases, the "window" of data expands until the model is eventually training on the entire dataset.
### **Mathematical Reasoning**
  - Introduce a weight function W(x,t) that modifies the distribution at time t
***
***
# **LLM Inference**
  - **The process of using a fully trained model to generate a response to a specific input (a prompt)**
  - **Unlike training, which updates the model's internal weights, inference is read-only**
  - **The model's "knowledge" is frozen; simply performing a massive series of matrix multiplications to calculate probabilities**
1. Input Tokenization
2. Forward Pass
3. Logit Generation
4. Softmax & Sampling
5. Autoregression (The Loop): The chosen token is added back to the prompt, and the whole process repeats to find the next token

## KV Cache
  - Without KV Cache: For every new word, the model re-calculates the math for the entire prompt from scratch. This is redundant (O(n^2) complexity)
  - With KV Cache: The model calculates the mathematical "essence" (the Keys and Values) of a word once, saves it in GPU memory, and simply "plugs it in" for the next step (O(n) complexity)
1. Prefill Phase: It calculates the Key (K) and Value (V) vectors for every token in the prompt and stores them in a "cache" (a dedicated slice of VRAM).
2. Incremental Decoding: The model generates the first new token
3. The Shortcut: To generate the second new token, the model doesn't re-process the prompt. It only calculates K and V for the one token it just created
4. Concatenation: It fetches the stored Ks and Vs from the cache, appends the new one, and performs the Attention calculation
### **Mathematical Reasoning**
When generating the token at position t+1:
  - We only need the Query(t+1) for the current token to ask "What should come next?"
  - But we need the Keys(1,2,...,t+1) and value (1,2,...,t+1) of all previous tokens to answer that question

## Parallelism
  - Technique of distributing massive model weights and computational workloads across multiple GPUs to reduce latency and increase throughput (tokens per seconds)
Three types of parallelism
1. Tensor Parallelism
  - Splits individual weight matrices (tensors) across multiple GPUs
### **Mathematical Reasoning**
  Column Parallelism: 
    - W split vertially W=[W1|W2]
    Y=X[W1|W2] = [XW1|XW2]
    *Each GPU computes a portion of the out, and concatenate at the end.
  Row Parallelism:
    - W split horizontally and X needs to split as well. 
2. Pipeline Parallelism
  - Splits the model layer-wise.
  Steps:
    - **Distribute**: Load a chunk of layers into each GPU's memory
    - **Forward Pass**: GPU 1 processes the input and passes the activations to GPU 2
    - **Handoff**: GPU treats that data as its input, process, and passes it to GPU 3 
  *The main mathematical challenge is the "Pipeline Bubble"—the idle time where GPU N is waiting for GPU 1 to finish.*
  - To fix this, we use Micro-batching, splitting one request into tiny chunks so the pipeline stays full
3. Data Parallelism
  -  Makes N copies of the full model on N different GPUs
  *Used to handle high throughput (many users at once)*

## Decoding Strategies
  - Rules that determine how a model selects the next token from the list of probabilities it has generated
### **Mathematical Reasoning**
  - Before any strategy picks a words, we often apply Temperature (T). 
  - Low Temp: Makes high probabilities even higher, more deterministic
  - High Temp: Flattens the distribution, more creative
**Deterministic Strategies**: Focus on logic and consistency (Good for coding/math)
**Stochastic (Random) Strategies**: Focus on variety and "human-like" flow (Good for creative writing/chat)
1. Greedy Search: Always pick the token with highest probability
  - Weakness: *Can get stuck in repetitive loops because it never explores "less likely" paths that might lead to better sentence*
2. Beam Search: Keeps track of top N (beam width) most likely sequences at every step
  - Step 1: keep the top N words
  - Step 2: Look at all possible next words for all three starts
  - Keep only the top 3 total paths based on cumulative probability
### **Mathematical Reasoning**
  - Maximize the joint probability of the whole sequence
3. Top-K and Top-P Sampling (The creative way)
  - Introduces controlled randomness by "truncating" the tail of unlikely words
***
***
