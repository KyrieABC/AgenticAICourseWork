#Pydantic BaseModel&Field
"""
from pydantic import BaseModel, Field
from typing import Optional, List
#Basics:(more search online for usages)
class User(BaseModel):
    # Simple field - type hints only
    name: str
    # Field with validation
    age: int = Field(gt=0, lt=150)
    # Optional field with default
    email: Optional[str] = Field(default=None)
    # Field with description and example
    bio: str = Field(
        default="",
        description="User biography",
        examples=["Software engineer with 5 years experience"]
    )
"""
#For class StateName(BaseModel or TypeDict)
#Use Typedict when:
"""
You want simplicity: It behaves like a standard Python dictionary
You are using Reducers: LangGraph’s Annotated[list, operator.add] syntax feels very natural inside a TypedDict
Performance is key: TypedDict is a type-hinting construct; it has zero overhead at runtime
You don't need strict type enforcement: It won't stop you if you accidentally pass a string where an int should be; it just helps your IDE give you better autocomplete
"""
#Use BaseModel when:
"""
Data Validation: You want the graph to throw an immediate error if a node returns data in the wrong format (e.g., a string instead of a UUID)
Complex Defaults: You want to use Pydantic's Field(default_factory=...) to handle complex initializations
Serialization: You are sending this state over an API (like FastAPI). Since Pydantic models have built-in .json() methods, it makes integration seamless
Coercion: You want Pydantic to automatically turn the string "10" into the integer 10
"""

#Name: type - type hint,optional during run time but help with debugging.
#Name: Annotated[type,metadata]- Enhanced type hints, developer could see the constraint.

#Langgraph thinking pattern
#1.Map out your workflow as discrete steps
#-Each step will become a node (a function that does one specific thing). Then, sketch how these steps connect to each other.

#2.Identify what each step needs to do
"""
Types of Step:
  LLM steps- When a step needs to understand, analyze, generate text, or make reasoning decisions
  Data steps- When a step needs to retrieve information from external sources
  Action steps-When a step needs to perform an external action
  User input steps-When a step needs human intervention
"""

#3. Design your state
#State is the shared memory accessible to all nodes in your agent. Think of it as the notebook your agent uses to keep track of everything it learns.
"""
Include in state:
Does it need to persist across steps? If yes, it goes in state.
Don't store:
Can you derive it from other data? If yes, compute it when needed instead of storing it in state.
Keep State raw and format prompts on-demand 
  The state contains only raw data - no prompt templates, no formatted strings, no instructions.
This separation means:
  Different nodes can format the same data differently for their needs
  You can change prompt templates without modifying your state schema
  Debugging is clearer - you see exactly what data each node received
  Your agent can evolve without breaking existing state
"""

#Error handling:
"""
Transient errors(network issues, rate limits): Temporary failures that usually resolve on retry
  Ex:
  from langgraph.types import RetryPolicy

  workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
  )
"""
"""
LLM-recoverable errors(tool failures,parsing issues): LLM can see the error and adjust its approach
  Ex:
  from langgraph.types import Command
  def execute_tool(state: State) -> Command[Literal["agent", "execute_tool"]]:
    #Command[Literal["agent],"execute_tool"] means teh goto parameter can only be one of these two, agent or execute_tool
    try:
        result = run_tool(state['tool_call'])
        return Command(update={"tool_result": result}, goto="agent")
    except ToolError as e:
        # Let the LLM see what went wrong and try again
        return Command(
            update={"tool_result": f"Tool error: {str(e)}"},
            goto="agent"
        )
"""
"""
User-fixable errors(missing information,unclear instructions):Need user input to proceed
  Ex:
  from langgraph.types import Command
  def lookup_customer_history(state: State) -> Command[Literal["draft_response", "lookup_customer_history"]]:
    if not state.get('customer_id'):
        user_input = interrupt({
            "message": "Customer ID needed",
            "request": "Please provide the customer's account ID to look up their subscription history"
        })
        return Command(
            update={"customer_id": user_input['customer_id']},
            goto="lookup_customer_history"
        )
    # Now proceed with the lookup
    customer_data = fetch_customer_history(state['customer_id'])
    return Command(update={"customer_history": customer_data}, goto="draft_response")
"""
"""
Unexpected errors: Unknown issues that need debugging
  Ex:
  def send_reply(state: EmailAgentState):
    try:
        email_service.send(state["draft_response"])
    except Exception:
        raise  # Surface unexpected errors
"""

#Command class in LangGraph is the primary control mechanism for directing graph execution.
"""
Main parameters:
  1.goto(required):Specifies which node to execute next.
    Command(goto="next_node")
  2.update(optional): Updates the graph state with new data.
    Command(
      goto="next_node",
      update={
        "key1": "value1",
        "key2": ["list", "of", "items"]
      }
    )
  3.then(optional):Chains multiple commands for sequential execution. 
    Command(
      goto="node_a",
      then=[
        Command(goto="node_b"),
        Command(goto="node_c")
      ]
    )
  4.interrupt(optional):Pauses execution for human input.
    Command(
      interrupt={
        "message": "Need approval",
        "request": "Approve this action?",
        "options": ["yes", "no"]
      }
    )
  5.result(optional):Returns a final result (ends execution).
    Command(
      result={
        "status": "success",
        "data": processed_data
      }
    )
"""
#4. Build your Nodes


#Workflows:
# LLMs and Augmentation
"""
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )
# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)
# Invoke the augmented LLM
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
# Define a tool
def multiply(a: int, b: int) -> int:
    return a * b
# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])
# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("What is 2 times 3?")
# Get the tool call
msg.tool_calls
"""
#LLM WorkFlow
"""
LLM is embedded in predefined code paths
  1.Prompt-Chaining: when each LLM call processes the output of the previous call.
    Input->LLM calls->Execute->Output
  2.Parallelization:
    Input->LLM execution->output
         \>LLM execution->output
LLM directs control flow through predefined code paths
  3.Orchestrator-Worker
    Input->Orchestration->Workers->Synthesizer->output
                        \>Workers/>
  4.Evaluator-Optimizer
    Input->generator->Evaluator->Output
                    <-
"""
#Agent (With Autonomy)
"""
LLM directs its own actions based on environmental feedback
  Input->LLM call-action->(conditional edge)Tool->Output
                <Feedback- 
"""

"""
Each node is designed to read the current shared state, perform some work, and then return updates to that state.
"""
#Persistence:
#LangGraph has a built-in persistence layer, implemented through checkpointers.
# Thread is different for each conversation, checkpoint id represent each update of graph

#When you compile a graph with a checkpointer, the checkpointer saves a checkpoint of the graph state at every super-step.

#Thread: unique ID or thread identifier assigned to each checkpoint saved by a checkpointer.
#Contains the accumulated state of a sequence of runs. When a run is executed, the state of the underlying graph of the assistant will be persisted to the thread.
#When invoking a graph with a checkpointer, you must specify a thread_id as part of the configurable portion of the config:
#{"configurable": {"thread_id": "1"}}
#The checkpointer uses thread_id as the primary key for storing and retrieving checkpoints. 
#Two types of persistence components:
# - Persistence Component (LangGraph): Allow the agent to remember the state of a conversation across multiple interactions
"""
1.
from langgraph.checkpoint import InMemorySaver
memory = InMemorySaver()
# 2. Use it when compiling your graph
app = workflow.compile(checkpointer=memory)
# 3. Access a specific conversation using a thread_id
config = {"configurable": {"thread_id": "user_session_123"}}
app.invoke({"messages": ["Hi, I'm Gemini."]}, config)
#State is stored in RAM
#No persistence-data disappears when the process end
2. 
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
# 1. Setup a connection to a local database file
conn = sqlite3.connect("state.db", check_same_thread=False)
# 2. Initialize the saver with that connection
memory = SqliteSaver(conn)
# 3. Use it in your graph (same as above)
app = workflow.compile(checkpointer=memory)
# Now, even if you stop the script and run it tomorrow, 
# thread "user_session_123" will still be there!
  This stores data on a Disk (hard drive) or a database
"""
#Checkpoints: The state of a thread at a particular point in time
#Checkpoint is a snapshot of the graph state saved at each super-step
#Key properties:
"""
config: Config associated with this checkpoint
metadata: Metadata associated with this checkpoint
values: Values of the state channels at this point in time
next: A tuple of the node names to execute next in the graph
task: A tuple of PregelTask objects that contain information about next tasks to be executed
"""

#graph.get_state(config)
# - When interacting with the saved graph state, you must specify a thread identifier
# This will return a StateSnapshot object that corresponds to the latest checkpoint associated with the thread ID provided in the config or a checkpoint associated with a checkpoint ID for the thread, if provided

#graph.get_state_history(config)
# -This will return a list of StateSnapshot objects associated with the thread ID provided in the config

#If we invoke a graph with a thread_id and a checkpoint_id, then we will re-play the previously executed steps before a checkpoint that corresponds to the checkpoint_id, and only execute the steps after the checkpoint

#graph.update_state(config or values)
"""
config:
  - The config should contain thread_id specifying which thread to update
  - Optionally, if we include checkpoint_id field, then we fork that selected checkpoint
values:
  - These are the values that will be used to update the state
  - This means that update_state does NOT automatically overwrite the channel values for every channel, but only for the channels without reducers
Ex:
class State(TypedDict):
    foo: int
    #add: reducer specified for bar key so it appends the new update
    bar: Annotated[list[str], add]
"""

#as_node?


#Memories are namespaced by a tuple (Ex: <user_id>, "memories")
#Use the store.put method to save memories to our namespace in the store
"""
Ex:
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()
user_id = "1"
namespace_for_memory = (user_id, "memories")
memory_id = str(uuid.uuid4())
memory = {"food_preference" : "I like pizza"}
store.put(namespace_for_memory, memory_id, memory)
"""
#Read out memories in our namespace using the store.search method, return all memories for a given user as a list.
"""
memories = store.search(namespace_for_memory)
# Converts a specific search result into a standard python dictionary
#Each memory type is a Python class (Item) with certain attributes
memories[-1].dict()
{'value': {'food_preference': 'I like pizza'},
# value: The value (itself a dictionary) of this memory 
 'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
# key: A unique key for this memory in this namespace
 'namespace': ['1', 'memories'],
# namespace: A tuple of strings, the namespace of this memory type
 'created_at': '2024-10-02T17:22:31.590602+00:00',
 'updated_at': '2024-10-02T17:22:31.590605+00:00'}
"""
#Semantic Search:
"""
from langchain.embeddings import init_embeddings
store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),  # Embedding provider
        "dims": 1536,                              # Embedding dimensions
        "fields": ["food_preference", "$"]              # Fields to embed
    }
)
#Search:
# Find memories about food preferences
# (This can be done after putting memories into the store)
memories = store.search(
    namespace_for_memory,
    query="What does the user like to eat?",
    limit=3  # Return top 3 matches
)
You can control which parts of your memories get embedded by configuring the fields parameter or by specifying the index parameter when storing memories:
# Store with specific fields(In this Example: food_preference or $) to embed
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "food_preference": "I love Italian cuisine",
        "context": "Discussing dinner plans"
    },
    index=["food_preference"]  # Only embed "food_preferences" field
)

# Store without embedding (still retrievable, but not searchable)
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"system_info": "Last updated: 2024-01-01"},
    index=False
)
Retrievable: you can go directly to that "address" and grab the data.
Searchable (Semantic Search): This refers to the ability to find data using a natural language query (e.g., "What does the user like to eat?"). This requires Embeddings.
index=False:
  "Store this data, but do not send it to the OpenAI embedding model."
  If you run store.search(query="When was the system updated?"), the store calculates the distance between your query and the vectors it has.
  Since the system_info has no vector, it's invisible to the search algorithm.
"""
"""
Used in LangGraph:
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
@dataclass
class Context:
    user_id: str
# We need this because we want to enable threads (conversations)
checkpointer = InMemorySaver()
# ... Define the graph ...
# Compile the graph with the checkpointer and store
builder = StateGraph(MessagesState, context_schema=Context)
# ... add nodes and edges ...
graph = builder.compile(checkpointer=checkpointer, store=store)

# Invoke the graph
config = {"configurable": {"thread_id": "1"}}
# First let's just say hi to the AI
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi"}]},
    config,
    stream_mode="updates",
    context=Context(user_id="1"),
):
    print(update)

# You can access the store and the user_id in any node by using the Runtime object. 
# The Runtime is automatically injected by LangGraph when you add it as a parameter to your node function.
async def update_memory(state: MessagesState, runtime: Runtime[Context]):
    # Get the user id from the runtime context
    user_id = runtime.context.user_id
    # Namespace the memory
    namespace = (user_id, "memories")
    # ... Analyze conversation and create a new memory
    # Create a new memory ID
    memory_id = str(uuid.uuid4())
    # We create a new memory
    await runtime.store.aput(namespace, memory_id, {"memory": memory})

We can access the memories and use them in our model call
async def call_model(state: MessagesState, runtime: Runtime[Context]):
    # Get the user id from the runtime context
    user_id = runtime.context.user_id

    # Namespace the memory
    namespace = (user_id, "memories")

    # Search based on the most recent message
    memories = await runtime.store.asearch(
        namespace,
        query=state["messages"][-1].content,
        limit=3
    )
    info = "\n".join([d.value["memory"] for d in memories])

    # ... Use memories in the model call
"""

#Capabilities:
"""
Human-in-Loop:
  - First, checkpointers facilitate human-in-the-loop workflows by allowing humans to inspect, interrupt, and approve graph steps.
  - Human has to be able to view the state of a graph at any point in time
  - Graph has to be to resume execution after the human has made any updates to the state
"""
"""
Memory: 
  - Checkpointers allow for “memory” between interactions
"""
"""
Time Travel:
  - Allowing users to replay prior graph executions to review and / or debug specific graph steps
  - Make it possible to fork the graph state at arbitrary checkpoints to explore alternative trajectories
"""
"""
Fault-tolerance:
  - if one or more nodes fail at a given superstep, you can restart your graph from the last successful step.
  - when a graph node fails mid-execution at a given superstep, LangGraph stores pending checkpoint writes from any other nodes that completed successfully at that superstep
    - whenever we resume graph execution from that superstep we don’t re-run the successful nodes
"""