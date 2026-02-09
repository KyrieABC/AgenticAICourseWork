# **RAG (Retrieval-Augmented Generation)**
- Takes an input and retrieves a set of relevant/supporting documents given a source (e.g., Wikipedia). 
- The documents are concatenated as context with the original input prompt and fed to the text generator which produces the final output.

***
***
***
## RAG Workflow
### 1. Data Ingestion & Chunking
Since documents can be massive, the system breaks them down into smaller, manageable pieces called chunks.
### 2. Embedding
These text chunks are converted into numerical representations called vectors or embeddings.
This process captures the semantic meaning of the text.
### 3. Vector Database Storage
The embeddings are stored in a specialized Vector Database (like Pinecone, Milvus, or Weaviate).
This acts as a searchable index where the AI can find information based on concepts rather than just keywords.
### 4. Retrieval
When a user asks a question, the system converts that question into a vector and searches the database for the chunks that are most mathematically similar to the query.
### 5. Augmentation & Generation
The system takes the user's original question and "stuffs" the retrieved text chunks into the prompt.
***
***

## Dense vs Sparse Retrieval 
 - Modern RAG systems usually use Hybrid Search.
### Dense Retrieval (Semantic Search)
An embedding model converts a sentence into a point in a multi-dimensional space. 
If two sentences are conceptually similar, they will be mathematically "close" to each other in that space.
*Weakness*: Can be "too fuzzy." If you search for a specific product ID or a very rare technical term, it might return something "similar" but not the exact match you need.
### 2. Sparse Retrieval (Keyword Search)
It counts how many times a word appears in a document vs. how often it appears in the whole library. Rare words get more weight.
*Weakness*: It has no "common sense." If you search for "feline" and the document says "cat," sparse retrieval will find nothing.
***
***
## Hybrid Search
### 1. Query Dualization
**When you ask a question, the system sends it to two different engines at once:**
  - The Sparse Engine
  - The Dense Engine
### 2.Parallel Retrieval
**The two engines search the database independently**
### 3. Fusion (Merging)
**Reciprocal Rank Fusion (RRF):**
RRF looks at the rank (position) of a document in both lists. 
If a document is #1 in the keyword search and #3 in the semantic search, it gets a massive "boost" in the final hybrid list.
### 4. Alpha Weighting
**You can tune the "Alpha" ($\alpha$) parameter to decide which engine to trust more:**
  - *$\alpha$ = 0.0*: Pure Keyword Search
  - *$\alpha$ = 1.0*: Pure semantic Search
  - *$\alpha$ = 0.5*: Equal 50/50 split
***
***

## Re-ranking (cross-encoders)
***Takes the top results from the initial search and examines them with much higher scrutiny to find the absolute best match.***
### 1. The Initial Search (Bi-Encoder)
**The system calculates the "similarity" between the query and the documents separately.**
  -It’s like comparing two pre-written summaries to see if they look alike.
*It is fast but loses the nuance of how the question specifically interacts with the answer.*
### 2. The Re-ranker (Cross-Encoder)
**The Cross-Encoder takes the Query and one Document chunk and feeds them into the model simultaneously.**
  -Instead of comparing two vectors, the model "reads" the query and the document together to see if the document actually answers the question.
  -It outputs a probability score between 0 and 1. 
*Only do it for the top 10–20 results found in the first step due to expensiveness.*
### 3. The Re-ordering
**The system re-sorts the documents based on the Cross-Encoder's high-precision scores.**
**Some Scenario to use Re-ranking:**
  - 1. Avoiding "Near Misses"
  - 2. Complex Legal/Policy Nuance

***
***
## Query Rewriting
***an LLM acts as a translator, turning a messy user query into a highly optimized search term before it ever touches the vector database.***
### 1. Intent Analysis
**A small, fast LLM (like GPT-4o-mini or a fine-tuned T5) analyzes the query to determine what the user is actually looking for.**
### 2. Transformation (The Rewrite) 
  - Expansion: Adding synonyms or related technical terms to increase the "surface area" of the search.
  - Decomposition: If the question is complex, the rewriter breaks it into 2-3 smaller, simpler sub-questions.
  - HyDE (Hypothetical Document Embedding): The LLM generates a "fake" answer to the question first, then uses that fake answer to search for real documents that look like it.
### 3. Retrieval with the "New" Query
**The rewritten query is embedded and sent to the vector database.**
*Because the query is now more "document-like," the retrieval hits are significantly more accurate.*

***
***
***
## Rag Indexing
**The primary goal of indexing is to convert unstructured data (like PDFs, emails, or manuals) into a Vector Index.**
  -*Standardization*: It cleans the data by removing irrelevant formatting, headers, or "noise" from the raw files
  -*Granularity (Chunking)*:It breaks large documents into smaller, meaningful "chunks."
  -*Context Preservation*: Good indexing uses "overlaps" between chunks so that the end of one paragraph and the start of the next remain connected.
  -*Semantic Mapping (Embedding)*: It uses an AI model to turn text into a list of numbers (vectors).
  - *Efficient Retrieval*: It organizes these vectors using specialized algorithms (like HNSW or IVF) so that searching through millions of documents takes milliseconds.

***
***
## Chunking
  - Fixed-Size Splitting: Define a number limit for tokens where splitter cuts exact at that limit
  - Overlap: To prevent a sentence from being cut in half and losing its meaning, we add a "buffer."
  *If Chunk A ends at character 500, Chunk B might start at character 400. This 100-character overlap ensures context is preserved across the "cut."*
  - Recursive Splitting: This is the most popular method. It tries to split by the largest logical separators first (like double line breaks for paragraphs), then by single line breaks, then by periods, and finally by individual characters if necessary.
Best Use Cases:
1. You have long, unstructured documents
2. The answers are localized
3. Context is easily lost
***
***
## Metadata filtering
**process of using structured attributes (like dates, categories, or authors) to narrow down the search space before or after performing a vector similarity search.**
### 1. The Setup (Indexing Time)
When you store data, you don't just store the text and its vector. 
*You attach a "JSON-like" object of metadata to each chunk.*
###  2. The Extraction (Retrieval Time)
When a user asks a question like *"What were the Q3 profits from 2023?"*, an LLM (often called a "Self-Querying Retriever") extracts two things:
  - The Semantic Query: "Q3 profits"
  - The Filter: year == 2023
### 3. Pre-Filtering vs. Post-Filtering
**Pre-filtering (Most common)**: The database first ignores every document that isn't from 2023. 
  -Then, it performs a vector search only on the remaining 2023 documents.
*This is faster and more accurate.*
**Post-filtering**: The database finds the top 100 most similar documents, then throws away any that aren't from 2023.
*This is riskier because if all top 100 results were from 2024, you'd end up with zero results.*
**Best Use Cases**:
1. You have massive datasets
2. Recency is critical
3. You have security/permission needs
4. You need "Exact Matches
***
***
## Hierarchical Indexing (Multi-level indexing)
**organizes information into a layered, "tree-like" structure.**
### Typical 3-tier hierarchy:
  - **Level 1 (The Forest)**: Summaries of entire documents or large collections.
  - **Level 2 (The Trees)**: Summaries of specific chapters or sections.
  - **Level 3 (The Leaves)**: The actual raw text chunks (paragraphs/sentences).
### The "Drill-Down" Retrieval Workflow
1. The system compares the query against the *Level 1* summaries to identify which documents are relevant.
2. It "drills down" into only the relevant documents to search their *Level 2* section summaries.
3. It finally retrieves the most relevant *Level 3* chunks from those specific sections to feed to the LLM.
**Best use cases**:
1. The answer requires "Big Picture" + "Small Detail"
2. You are doing Multi-Document RAG
3. You want to reduce "Noise"

***
***
***
## Over-retrieval
**situation where the system fetches too many document chunks—or chunks that are too long—and "stuffs" them into the LLM's prompt**
### Why it occurs?
1. High $k$ Values: 
  - If $k$ is set too high (e.g., $k=20$), the system might retrieve 3 relevant chunks and 17 "distractors" that happen to share similar keywords but are irrelevant to the user's specific question.
*In the retrieval step, the system asks for the "top $k$" most similar documents.*
2. Large or Inefficient Chunking
3. 3. Vague or Broad Queries
4. Poor Semantic Discrimination
  -Embedding models sometimes struggle with "false positives"—sentences that are mathematically close in vector space but logically different.
### Way to fix the issue
1. Re-ranking (The "Elite Judge")
2. Contextual Compression
  -This technique uses a "Compressor" model to scan the retrieved chunks and extract only the relevant sentences or phrases, discarding the surrounding "fluff."

***
***
## Semantic Drift
**a phenomenon where the original meaning or intent of information is gradually distorted, lost, or "shifted" as it travels through the various stages of the AI pipeline.**
### Why it occurs?
1. The "Dumb Chunking" Problem
  - If you use fixed-size chunking, you might slice a sentence exactly where the critical context lives.
*Ex: ... not allowed to park here after 5 PM. Cut into "not allowed to park" and "here at 5 PM".*
2. Embedding Model Limitations (Lossy Compression)
  - The model might prioritize general "themes" over specific "negations.
*Ex: If a user asks for "non-dairy options" and the model focuses on the word "dairy," it may retrieve "Whole Milk" documents because they are mathematically close in the "milk" vector space, even though the intent is the opposite.*
3. Query-Document Mismatch
  - The semantic gap between these two terms can cause the retriever to "drift" toward irrelevant but similarly phrased documents
4. Multilingual "Attractors" 
  - The LLM might "drift" its response into English because its training data is heavily English-centric, causing it to ignore the linguistic nuances of the retrieved context.
### Way to fix
1. Semantic Chunking (Meaning-Based Splits)
2. Knowledge Graph Integration (GraphRAG)
  - By linking text chunks to a Knowledge Graph, you ground the semantic search in hard relationships.
*If a chunk mentions "Nvidia," the graph knows it's a "Company" that makes "GPUs."*

  

