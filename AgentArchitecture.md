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
  - 
