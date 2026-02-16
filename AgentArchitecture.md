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
