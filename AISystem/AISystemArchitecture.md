# AI Systems Architecture

## Data Architecture

### Streaming vs Batch
  - In AI and data architecture, the choice between Batch and Streaming is essentially a choice between "efficiency at scale" and "responsiveness in the moment"
#### Batch Processing (High-Volume Approach)
  - Collecting data over a period of time and processing it all at once in a large group (a "batch"). This is the traditional method for heavy-duty analytics and training AI models
##### Steps
  1. **Collection**: Data is gathered and stored in a landing zone
  2. **Triggering**: The process starts based on a schedule (e.g., every night at 2:00 AM) or a data size threshold
  3. **Transformation**: A processing engine (like Apache Spark) reads the entire dataset, cleans it, and applies transformations
  4. **Loading**: The final, processed data is moved to a Data Warehouse for reporting or used to update an AI model's weights
  - *Used for training a Large Language Model (LLM) on a massive static dataset*
  - Use Batch if you need high-volume accuracy and historical context where time is not of the essence
#### Streaming Processing (Real-Time Approach)
  - Handles data as it arrives, piece by piece. There is no waiting for a group; the system reacts to every individual event 
##### Steps:
  1. **Ingestion**: Data is produced as a continuous stream of events
  2. **Processing**: A stream processor (like Flink or Spark Streaming) analyzes the data instantly as it flows through the system
  3. **Immediate Action**: The system triggers an alert, updates a live dashboard, or feeds a real-time AI inference engine
  4. **Short-Term Storage**: Data is often kept in "windows" (e.g., the last 5 minutes of data) for immediate context
  - Use Streaming if your business depends on reacting to events the moment they happen
| Feature | Batch Processing | Streaming Processing |
| :--- | :--- | :--- |
| **Data Scope** | Processes large, static datasets (bounded). | Processes individual events or micro-batches (unbounded). |
| **Latency** | High: Minutes, hours, or days. | Low: Milliseconds to seconds. |
| **Complexity** | Generally simpler; easier to debug and re-run. | Higher; requires specialized tools for "state" and "windowing." |
| **Data Size** | Massive volumes handled in one go. | Continuous flow of smaller data packets. |
| **Use Case** | Deep historical analysis, training AI models, payroll. | Real-time fraud detection, IoT monitoring, live bidding. |
| **Efficiency** | High throughput, lower cost per record. | High responsiveness, higher infrastructure overhead. |

### Data Lineage
  - **The process of documenting the data's lifecycle. It answers three critical questions: Where did this data come from?, What happened to it?, and Who is using it now?**
  - **Upstream vs. Downstream**: "Upstream" refers to the source systems (databases, sensors); "Downstream" refers to the consumers (dashboards, AI models)
  - **Metadata**: Data about the data (timestamps, schema versions, ownership) that powers the lineage map
  - **Transformations**: Any logic applied to the data, such as x = (Price) X (Tax Rate), which must be recorded to understand how a final value was reached
  - **Impact Analysis**: The ability to see what will break "downstream" if you change a column "upstream."
#### Steps:
  1. **Extraction**: The system scans your data tools (SQL databases, Spark jobs, ETL pipelines) to extract logs and "recipes" (code) used to move data
  2. **Mapping (Linkage)**: The lineage tool connects the dots. It identifies that "Table A" in the Sales DB was joined with "Table B" in the Marketing DB to create "Table C" in the Data Warehouse
  3. **Visualization**: The metadata is converted into a Directed Acyclic Graph (DAG). This is a flow chart that shows the movement of data without any circular loops
  4. **Versioning & Tracking**: As the AI model is retrained, the lineage records which specific version of the data was used at that exact moment in time
#### Why Should You Use it?
  - **Data Lineage turns your data architecture from a "black box" into a transparent map. It ensures that your AI is built on a foundation of verifiable, high-quality information**

### Feedback Loops
  - Post-deployment process where the predictions made by an AI (the "Inferences") are compared against real-world outcomes (the "Ground Truth")
  - **Model Drift**: The phenomenon where an AI’s accuracy decays over time because the real world changes (e.g., a fashion AI not knowing about a new trend)
  - **Ground Truth**: The actual, verified reality of what happened (e.g., did the user actually click the recommendation?)
  - **Active Learning**: A specific type of feedback loop where the AI identifies data points it is "unsure" about and asks a human to label them
  - **Exploitation vs. Exploration**: Balancing using what the AI already knows (Exploitation) versus trying new things to see if they work better (Exploration)
#### Steps:
  1. **Prediction (Inference)**: The AI model processes live data and produces a result. For example, a music app predicts you will like a specific Jazz song
  2. **Observation (The Signal)**: The system monitors the user's reaction
    - Positive Signal vs Negative Signal (preference based on action)
  3. **Data Ingestion & Labeling**: The signal (the skip) is paired with the original input data.
    - Ex: User X + This Jazz Song = Dislike
  4. **Evaluation & Retraining**: Data scientists or automated MLOps (Machine Learning Operations) pipelines analyze these signals
    - If the "dislike" rate for Jazz is high across many users, the model is retrained using this new batch of data to adjust its weights
  5. **Deployment**: The updated model is pushed to production, and the loop begins again with better accuracy
#### Why use Feedback Loops?
  - **Without a feedback loop, an AI is a "frozen" snapshot of the past. With a feedback loop, the AI becomes a living system that grows smarter with every interaction**

## Model Orchestration

### Model Routing
  - **An orchestration layer that intercepts an incoming query and directs it to the most appropriate model (or ensemble of models) based on predefined rules or machine learning logic**
  - **Router**: A lightweight "gateway" (often a small, fast LLM or a set of IF/THEN rules) that sits in front of your expensive models
  - **Model Tiering**: Categorizing models by capability (e.g., Tier 1: GPT-4o for complex logic; Tier 2: GPT-4o-mini for simple summaries; Tier 3: Llama-3-8B for basic classification)
  - **Fallback Logic**: If the preferred model is down or overloaded, the router automatically sends the request to a "backup" model
  - **Cost-Awareness**: Routing simple queries to cheaper models to save money (e.g., don't use a $20/million token model to say "Hello")
#### Steps:
  1. **Request Ingestion**: The user sends a prompt
  2. **Intent Classification (Routing Decision)**: Router analyzes the requests. Look for: 
    - **Complexity**: Is this a math problem, a creative writing task, or a coding bug?
    - **Length**: Is the answer expected to be short or long?
    - **Metadata**: Does the user have a "Premium" subscription (gets the best model) or a "Free" one (gets the basic model)?
  3. **Dispatching**: The Router sends the prompt to the chosen model's API or inference server
  4. **Verification (Optional)**: The system checks the output. If the model returns an error or a low-confidence score, the router may "re-route" the same prompt to a more powerful model to try again
  5. **Response Delivery**: The final answer is sent back to the user, often with logs showing which model was used and how much it cost
#### Why Should You Use Model Routing?
  - Model Routing turns your AI architecture from a single "God Model" into a flexible team. It ensures you are using the right tool for the job, preventing you from "using a sledgehammer to crack a nut"

### Cascades
  - An orchestration pattern where an input is processed by a sequence of increasingly powerful (and costly) models. 
  - The process stops as soon as a model produces a response that meets a specific "confidence" or "quality" threshold
  - **While Model Routing (which we discussed earlier) tries to guess the right model before running it, a Cascade actually runs a model, checks if the result is good enough, and only escalates to a bigger, more expensive model if necessary**
  - **The Chain**: A series of models ordered from smallest/fastest to largest/smartest
  - **The Exit Criterion (Stop Judge)**: A logic gate that evaluates the output
  - **Confidence Scoring**: A way for a model (or a small supervisor model) to say, "I am 90% sure this answer is correct"
  - **Efficiency-Accuracy Trade-off**: The goal is to solve 80% of "easy" queries using 1% of the total budget, saving the heavy compute for the remaining 20%
#### Steps
  1. **Intial Attempts (Small Model)**: The system sends the user's request to a Small Language Model (SLM). This model is extremely fast and costs almost nothing to run
  2. **Quality Evaluation (Gatekeeper)**: An evaluation step (often a tiny "Classifier" model or a self-consistency check) looks at the result
    - Example: Did the model follow the formatting? Is the answer coherent? Does the model's own "log-probability" suggest high confidence?
  3. **Conditional Termination**
    - If answer is good enough, terminate immediately
    - If answer is low confidence, systems discard the result and go to step 4
  4. **Escalation**: The prompt is passed to the next, more powerful model in the chain. This model might also receive the "failed" attempt from the previous step as context
  5 **Final Output**: The process continues until either a model passes the quality gate or the system reaches the "Oracle" model (the most powerful one available), which provides the final answer

### Fallback Systems
  - A redundancy strategy where a "Plan B" is triggered immediately upon the failure of "Plan A"
  - If a primary AI model fails, times out, or returns an error, the architecture automatically "falls back" to a secondary system to ensure the user isn't left staring at a "404 Error" or an empty screen
  - **Contingency (The Fallback)**: A secondary model, often smaller or self-hosted (e.g., Llama-3 or a specialized BERT model), that is highly reliable
  - **Graceful Degradation**: Instead of the system crashing, it provides a slightly less "intelligent" but functional response
  - **Circuit Breaker**: A design pattern that stops sending requests to a failing model for a short time to allow it to recover, routing everything to the fallback instead
#### Steps
  1. **Request Initiation**: The user sends a prompt. The system attempts to call the Primary Model
  2. **Failure Detection**: The orchestration layer monitors the request. It looks for specific "Failure Triggers"
    - **HTTP Errors**: 500 (Internal Server Error) or 429 (Rate Limit Exceeded)
    - **Timeout**: The model takes longer than 5 seconds to respond
    - **Empty Response**: The model returns nothing or a "Safety Filter" refusal
  3. **Triggering the Fallback**: The system catches the error. Instead of returning an error message to the user, it instantly re-routes the exact same prompt to the Fallback Model
  4. **Execution and Logging**: The Fallback Model generates the response. Simultaneously, the system logs the failure of the Primary Model so engineers can investigate if there is an outage
  5. **Seamless Delivery**: The user receives the answer. To the user, it might seem like the response took an extra second, but they never saw a "System Error"
#### Why Should You Use Fallback Systems?
  - Fallbacks are about Reliability (starting big and going to a backup to prevent failure)
  - Cascades are about Optimization (starting small and going big to save money)

## Retrieval Architecture

### RAG
### Hybrid Retrieval

### Index Sharding
  - A data architecture technique where a single, massive vector index is split into smaller, manageable pieces called "Shards"
  - **Vector Index**: A specialized database (like Pinecone, Milvus, or Weaviate) that stores embeddings (lists of numbers representing the meaning of text/images)
  - **Horizontal Scaling**: Adding more machines to a system rather than just making one machine more powerful
  - **Partitioning Key**: The logic used to decide which data goes to which shard (e.g., by User ID, Document Type, or a Hash)
  - **Scatter-Gather**: The orchestration process of sending a query to all shards simultaneously and "gathering" the best results from each
#### Steps:
  1. **Data Partitioning**: When you ingest a new document, the system doesn't just "dump" it into one big pile. It applies a sharding logic
    - For example, if you have 4 shards, a "Modulo" operation on the Document ID might determine its home: `ID %4`
  2. **Distributed Storage**: The document’s vector is stored in the assigned shard on a specific server. Each shard maintains its own local index
  3. **Query "Scatter"**: When a user asks a question, the Retrieval Orchestrator (the "Coordinator") receives the query vector. It doesn't know which shard has the best answer, so it "scatters" the query to all shards at once
  4. **Parallel Search**: Every shard searches its own subset of data in parallel. Because the datasets are smaller, this search is incredibly fast. Each shard returns its "Top 10" most relevant results
  5. **"Gather" & Re-ranking**: The Coordinator gathers the results from all shards (e.g., 4 shards $\times$ 10 results = 40 total). it re-ranks them to find the true "Top 10" globally and sends them to the LLM
#### WHy should you use Index Sharding
  - Index Sharding is the difference between a small library (where one librarian looks through every shelf) and a massive university archive (where you have a different librarian for every floor, all looking for your book at the same time)

