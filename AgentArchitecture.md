# **AI Agent Architecture**
  - LLM: *Stateless engine that predicts the next word*
  - AI Agent: *Use that engine to reason, use tools, and pursue specific goals over time*
1. Brain (Reasoning & Planning)
  - This is the central LLM. It breaks down a complex goal into smaller steps
  Ex: Reflection/Self-Criticism: The agent looks at its own plan and corrects errors before executing
2. Memory (Short-term & Long-term)
  - An agent needs to remember what it just did and what it learned in the past
  - Short-term: The immediate conversation history (Context Window)
  - Long-term: Usually a Vector Database. The agent "retrieves" relevant documents or past experiences using RAG (Retrieval-Augmented Generation)
3. Toolset (Action/Capabilities)
  - The architecture includes APIs or scripts that allow the LLM to interact with the outside world
  Ex: Web Search, Code Interpreter, App Integration
Control Flow: 
  - Perceive -> Plan -> Act -> Observe

## ReAct (Reasoning + Acting)
  - *Before ReAct, AI agents usually did one of two things: they either "reasoned" in a vacuum (Chain of Thought) or they "acted" blindly by calling tools without explaining why*
  - ReAct Agent Architecture follows a repeatable cycle: **Thought -> Action -> Observation**
1. Thought: 
  -The agent generates a reasoning step. It identifies what it knows, what it's missing, and which tool it needs to use next
2. Action:
  - The agent interacts with an external source (e.g., a Google Search API, a Calculator, or a Database)
3. Observation: 
  - The agent reads the output of the action (the search result, the math answer) and integrates it into its memory
  ### **Mathematical Reasoning**
  - ReAct aims to improve the probability of a correct final answer y given an input x
  - *In a standard LLM, the model tries to predict P(y|x) directly*
  - In ReAct, we introduce latent reasoning traces (i) and action/observations (a,o), essentially explores a state space to find the most likely path to solution
  - **By forcing the model to generate the "Thought" (i) and "Action" (a) explicitly, we reduce the chance of the model taking a "greedy" but incorrect path**
  - It acts as a form of trace-based regularization, keeping the model's logic grounded in the actual "Observations" (o) it receives from the real world
  - (Step-by-Step agent) If the probability of success for each step is $p$, the probability of completing an $n$-step task successfully is p^n. 
  - *As n grows, the success rate drops exponentially*
  ### What is ReAct good at:
  - Reducing Hallucination: Because the agent must "Observe" a fact before moving to the next "Thought," it is less likely to make up facts
  - Complex Multi-step problems: It excels at tasks where the second step depends entirely on the results of the first
  - Error correction: If an "Observation" returns an error, the next "Thought" can acknowledge the failure and try a different "Action"
  ### What is limitation of ReAct
  - While powerful, ReAct can be expensive and slow

## Planner-Executor (Plan-and-Solve)
  - ReAct agent thinks and acts one step at a time (like someone navigating a maze by only looking at their feet)
  - *Planner-Executor agent maps out the entire route before taking the first step (like someone looking at a map before entering the maze)*
1. Planner: 
  - Its sole job is to take a complex user goal and decompose it into a directed acyclic graph (DAG) or a sequential list of sub-tasks
  - Input: User Query
  - Output: A structured plan
2. Executor:
  - It receives one specific sub-task at a time from the Planner
  - It doesn't worry about the "big picture". *Simply executes the command and returns the result*
3. Re-Planner (Optional)
  - After the Executor completes a few steps, a Re-Planner reviews the results
  - If the original plan is no longer valid (e.g., a search returned no results), it updates the remaining steps
  ### Mathematical Reasoning -- Search Space Reduction
  - Reducing the compounding error rate
  - Planner-Executor treats the task as MDP where the Planner attempts to find the optimal policy for the entire sequence at once
  - By defining the state space and goal upfront, the Planner minimizes the Heuristic Distance to the solution
  - Plan = argmax (a1,a2,...,an) P(a1,...,an|Goal, Context)
  - Creates a "top-down" constraint that prevents the agent from drifting off-topic during execution
  ### What are Planner-Executor good at?
  - Complex Multi-step project
  - Efficiency
  - Parallelization: Since the plan is laid out, an Executor can often run multiple independent steps at the same time
  ### What are the limitation of Planner-Executor?
  - Plan Rigidity: If the environment changes or the first step yields unexpected information, the initial plan may become obsolete immediately
  - The "Paper Plan" Problem: The Planner might suggest a step that sounds logical but is technically impossible for the Executor to perform (Ex: Access User's private file)
  - High Latency: There is a significant "wait time" at the beginning while the agent generates the full plan before the first action is taken
  - Error Propagation: If the Planner makes a mistake in Step 1, every subsequent step based on that error will fail, potentially wasting time and API credits on a doomed path
  ### Comparison with ReAct
1. ReAct agent is reactive (deciding the next move only after seeing the result of the current one)
  - In a ReAct loop, the model follows a "Local Search" strategy. 
  -At each step t, it predicts the next token based on a growing history. P(Action(t)|H(t), Goal)
2. Planner-Executor is proactive (mapping out a sequence of moves before starting)
  - Goal-Conditioned Planning
  - It generates a sequence of actions A={a1,a2,...,an} that maximizes the probability of reaching the goal G: Plan = argmax (a1,a2,...,an) P(a1,...,an|Goal, Context)

## Critic/Self-Reflection
  - **It treats its first attempt as a draft. It then looks at that draft, identifies flaws, and regenerates a better version**
  - Focus on refinement
1. **Generator(Actor)**: 
  - Takes the initial prompt and produces a "Draft 1." Its goal is to fulfill the user's intent
2. **Critic(Evaluator)**:
  - Examines the draft against specific rubrics (e.g., "Is this factually accurate?", "Is the code efficient?", "Does it follow the requested tone?"). It produces a Critique
3. **Refiner(Optimizer)**:
  - Takes the original draft + the critique and generates a "Draft 2." This role is often played by the Generator again, but with the critique provided as context
4. **The Stopping Criteria**: 
  - A logic gate that decides when to stop. This is usually based on a maximum number of iterations (e.g., 3 loops) or a "Pass/Fail" signal from the Critic
  ### Mathematical Reasoning
  - Verification is often easier than generation
  - **P vs NP distinction: Finding a solution to a problem might be hard, but checking if a given solution is correct is often computationally "cheaper"**
  - If an LLM has a probability P(y|x) of generating the correct answer $y$ in one shot, that probability might be low for complex tasks. However, the probability that the model can recognize an error in a candidate answer y' —denoted as P(text{correct} | y', x) —is typically much higher
  ### What is Critic/Self-Reflection good at?
  - Coding and Debugging
  - Creative Writing
  - Reasoning Accuracy
  ### potential Limitation of Critic/Self-Refinement
  - **Hallucinated Criticism**: Sometimes the Critic "invents" a mistake that wasn't actually there (the "Over-Correction" problem), leading the Actor to ruin a perfectly good answer
  - **Infinite Loop**: If Actor isn't smart enough to fix the error identified by the Critic, the two can get stuck in infinite loop
  - **Reinforcement of Errors**: If the Actor and Critic share the same underlying model biases, the Critic may "agree" with the Actor's hallucination because it shares the same flawed logic

## Hierarchical Agent
  - **By nesting agents within agents, the architecture abstracts complexity at each level**
  - Separates strategic planning from tactical execution
1. Manager (Top Layer)
  - **Role**: Interprets the user's high-level intent and breaks it into major milestones
  - **Key Function**: Task Decomposition. It defines the "What" and sets the "Success Criteria" for the layers below
  - **Memory**: Typically maintains the global context and the long-term history of the project
2. Supervisor (Middle Layer)
  - **This layer acts as a bridge between abstract goals and concrete actions**
  - **Role**: Receives a milestone from the Manager and converts it into a sequence of actionable steps
  - **Key Function**: Coordination & Validation. It assigns tasks to specific worker agents, manages the order of operations, and reviews their work before passing it back up
  - **Communication**: Handles both "Downstream" directives and "Upstream" reporting
3. The Workers (Bottom Layer)
  - **These are agents with very narrow, deep expertise**
  - **Role**: Executes specific tool calls (APIs, Python code, Web search) to complete one single task
  - **Key Function**: Tool Use. A worker doesn't need to know the "Big Picture"; it only needs to know how to use its specific tool perfectly
  - **State**: Often stateless or restricted to a very small "short-term" memory to prevent hallucinations
  ### Mathematical Reasoning
  - The mathematical strength of a hierarchy lies in reducing the state space complexity
  - Hierarchies apply Temporal Abstraction (often modeled using the Options Framework in Reinforcement Learning)
  - Manager selectts a high level Option, this option acts as a Subgoal(g) for the lower layer, Worker agent only needs to solve for the path to g 
  - P(S) = P(Manager selects g) * P(Worker achieves g)
  - By isolating the logic into smaller chunks, we significantly increase the probability of success for each individual sub-component
  ### What are hierarchical Agent good at?
  - **Long-Horizon Task**: Writing entire books, building complex apps, or conducting multi-week research projects
  - **Security & Governance**: You can place "Safety Supervisors" in the middle of the hierarchy to audit worker outputs before the user sees them
  - **Scalability**: You can add more specialists ("Legal Agent", "Finance Agent") without having to retrain or re-prompt the Top Manager
  ### Potential Limitation of Hierarchical Agent
  - **Information Bottleneck**: If a Worker finds a critical problem but the Supervisor "summarizes it away" to the Manager, the system makes decisions based on incomplete data
  - **Latency**: Every message must travel up and down the chain of command, making the system slower than a single-agent loop
  - **Managerial Hallucination**: If the Manager is too far removed from the tools, it might order a Worker to perform an impossible task

## Tree-of-Thought
  - **Definition: Tree-of-Thought is a decoding and reasoning framework that enables LLMs to solve complex tasks by considering multiple "thoughts" as intermediate steps**
  - It treats problem-solving as a search over a tree structure, where each node represents a partial solution and each branch represents a potential next step
  - **Thought Decomposition**
  - **Thought Generator**
  - **State Evaluator**: A "critic" (often the LLM itself) looks at the proposed thoughts and decides which ones are promising, which are mediocre, and which are failures
  - **Search Algorithms**: BFS or DFS
1. **Brainstorming (proposing)**: From the starting point, the AI generates k possible next steps
2. **Evaluating (pruning)**:  The agent looks at those $k$ options and assigns them a value (e.g., "Sure," "Maybe," or "Impossible")
3. **Searching (expanding)**: The agent moves forward only with the "Sure" or "Maybe" options, creating new branches from those points
4. **Backtracking (correcting)**: If all branches from a specific thought lead to "Impossible," the agent "climbs back down" the tree to the last successful node and tries a different branch
  - *You should opt for a Tree-of-Thought approach when the cost of being wrong is high or the problem requires "look-ahead" reasoning*

## Graph-of-Thought
  - **Definition: Graph-of-Thought is a framework that models the reasoning process of an AI as a directed graph**
  - Unlike a tree, where branches only move forward and never meet, GoT allows "thoughts" (nodes) to be combined, looped, or cross-referenced
  - **Node (thoughts)**: Each node is an intermediate solution or a piece of information
  - **Edges (Dependencies)**: These show how one thought leads to another
  - **Aggregation (Merging)**: The ability to take the best parts of Thought A and Thought B to create Thought C
  - **Transformation (Refining)**: Taking a "rough draft" thought and looping it through a feedback node until it reaches a specific quality threshold
  - **Graph Controller**: The "brain" that decides which nodes to connect and when a path has reached a dead end
1. **Generation**: The agent generates multiple independent starting thoughts
2. **Transformation/Expansion**: The agent expands these thoughts, but unlike a tree, it can use one thought to influence multiple others
3. **Evaluation & Scoring**: Each node is scored by a "Critic" LLM
4. **Merging (The Secret Sauce)**: The agent identifies that "Thought A" has a great structural idea and "Thought B" has the correct mathematical data. It creates a new node that combines them
5. **Looping**: If a result is "good but not perfect," the agent sends the thought back to an earlier stage for refinement (Self-Correction)
  - Use graph-of-thought if you need a graph to manage that "conversation" between different parts of the problem
```
class ThoughtNode:
    def __init__(self, content, parents=None):
        self.content = content
        self.parents = parents or []  # Can have multiple parents (Graph!)
        self.score = 0.0
        self.status = "active" # active, validated, or rejected

class GoT_Controller:
    def __init__(self, initial_prompt):
        self.graph = [ThoughtNode(initial_prompt)]
        self.best_solution = None

    def generate_thoughts(self, node, num_variants=3):
        """Expansion: Creates multiple new potential steps from a single node."""
        return [LLM.generate_next_step(node.content) for _ in range(num_variants)]

    def evaluate_node(self, node):
        """Scoring: The LLM acts as a critic to rank the thought."""
        node.score = LLM.evaluate(node.content) 
        if node.score < 0.3:
            node.status = "rejected"

    def merge_thoughts(self, node_a, node_b):
        """Aggregation: The unique 'Graph' feature where two paths become one."""
        merged_content = LLM.combine(node_a.content, node_b.content)
        return ThoughtNode(merged_content, parents=[node_a, node_b])

    def solve(self):
        # 1. Start with initial ideas
        initial_ideas = self.generate_thoughts(self.graph[0])
        for idea in initial_ideas:
            new_node = ThoughtNode(idea, parents=[self.graph[0]])
            self.graph.append(new_node)

        # 2. Evaluate and prune
        for node in self.graph[1:]:
            self.evaluate_node(node)

        # 3. Find two high-scoring but different thoughts and MERGE them
        high_scorers = [n for n in self.graph if n.status == "active"]
        if len(high_scorers) >= 2:
            optimal_node = self.merge_thoughts(high_scorers[0], high_scorers[1])
            self.graph.append(optimal_node)
            
        # 4. Final Refinement Loop
        self.best_solution = LLM.refine_to_final_answer(self.graph[-1].content)
        return self.best_solution

# Usage
controller = GoT_Controller("Design a carbon-neutral city layout for 1M people.")
final_plan = controller.solve()
```
### Tree-of-Thought vs Graph-of-Thought
| Feature | Tree-of-Thought (ToT) | Graph-of-Thought (GoT) |
| :--- | :--- | :--- |
| **Logic Structure** | Hierarchical (Parent → Child) | Network (Web-like / Mesh) |
| **Information Flow** | One-way (Down the branches) | Multi-way (Merging and Looping) |
| **Data Efficiency** | Can be redundant (Repeats work) | High (Reuses and combines nodes) |
| **Error Handling** | Backtracking to a previous node | Iterative refinement & self-correction |
| **Ideal Use Cases** | Puzzles, Chess, Linear Planning | Coding, Legal, Research, Architecture |

## Sandboxing
  - **Definition: The practice of running an agent's actions—such as executing code, calling APIs, or modifying files—inside a strictly isolated, low-privilege environment**
  - **Compute Isolation**: The agent runs on a separate "MicroVM" or container (like Docker) with its own mini-operating system. It cannot "see" your real files or hardware
  - **Network Gating**: You control exactly which websites or databases the agent can talk to. By default, most sandboxes block all outgoing internet traffic to prevent data theft
  - **Resource Caps**: You set limits on CPU, memory, and time. If an agent enters an infinite loop or tries to mine crypto, the sandbox simply kills the process after 30 seconds
  - **Ephemeral State**: Every time the agent finishes a task, the entire environment is deleted. It’s like a hotel room that is completely gutted and rebuilt after every guest
1. **Request**: The AI "decides" it needs to calculate a complex math formula or process a CSV file
2. **Provisioning**: The system spins up a fresh, empty sandbox (e.g., a Docker container) in milliseconds
3. **Injection**: Only the specific files and data needed for that specific task are copied into the sandbox
4. **Execution**: The AI runs its code inside the sandbox. If the code says os.remove('/'), it only "deletes" the empty sandbox, not your computer
5. **Extraction**: The system pulls the result (e.g., the answer to the math problem) back out and displays it to you
6. **Teardown**: The sandbox is instantly destroyed, wiping any temporary files or errors
```
# 1. Setup: Dockerfile
# Use a slim version of Python to keep it fast
FROM python:3.11-slim
# Create a non-privileged user for safety
RUN useradd -m sandboxuser
USER sandboxuser
# Set the working directory inside the sandbox
WORKDIR /home/sandboxuser/app
# The sandbox stays idle until we send it code
CMD ["python3"]
# 2. Controller (sandbox_manager.py): This Python script on your actual machine will spin up the sandbox, toss the AI's code inside, run it, and bring back the result
import docker
import os

def run_in_sandbox(ai_generated_code):
    client = docker.from_env()
    
    # 1. Build the image (only happens once)
    client.images.build(path=".", tag="ai-python-sandbox")

    try:
        # 2. Run the container with strict limits
        result = client.containers.run(
            image="ai-python-sandbox",
            command=f'python3 -c "{ai_generated_code}"',
            network_disabled=True,      # No internet access for the AI
            mem_limit="128m",           # Limit RAM to 128MB
            cpu_quota=50000,            # Limit to 50% of one CPU core
            remove=True,                # Auto-delete container after run
            stderr=True
        )
        return result.decode('utf-8')
    
    except Exception as e:
        return f"Sandbox Blocked Action or Failed: {str(e)}"

# --- Example Usage ---
dangerous_code = "import os; print('I am trying to see your files:', os.listdir('/'))"
output = run_in_sandbox(dangerous_code)
print(f"AI Output:\n{output}")
```
  - Why this works?
    1. **Network Disabled**: If the AI agent tries to send your data to an external server via a `POST` request, the sandbox will simply throw a network error
    2. **User Permissions**: By using sandboxuser, the AI cannot install new software or change system settings inside the container
    3. **Memory/CPU Limits**: This prevents "Denial of Service" attacks where an agent might accidentally (or intentionally) create an infinite loop that freezes your actual computer
    4. **Auto-Removal**: The `remove=True` flag ensures that as soon as the code finishes, the "disposable computer" vanishes, leaving no traces behind
    - You can potentially add a "Volume Mount" so the sandbox can safely read a specific folder of data without seeing the rest of your system