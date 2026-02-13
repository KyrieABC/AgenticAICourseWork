#Haven't installed agents module's requirement
from agents import Agent, Runner

#Multi-agent design pattern
#Agent as tool(manager):
#Delegation: User → Primary Agent → Tool Agent → Primary Agent → User
#Tool agent sees only the specific task
#Only primary agent talks to user
#Always returns to primary agent
"""
booking_agent = Agent(...)
refund_agent = Agent(...)
customer_facing_agent = Agent(
    name="Customer-facing agent",
    instructions=(
        "Handle all direct user communication. "
        "Call the relevant tools when specialized expertise is needed."
    ),
    tools=[
        booking_agent.as_tool(
            tool_name="booking_expert",
            tool_description="Handles booking questions and requests.",
        ),
        refund_agent.as_tool(
            tool_name="refund_expert",
            tool_description="Handles refund questions and requests.",
        )
    ],
)
"""

#Handoff:
#One-way transfer: User → Router → Specialist → User
#Specialist gets full conversation context
#Specialist talks directly to user
#No return to router (usually)
#User knows which agent they're talking to
"""
booking_agent = Agent(...)
refund_agent = Agent(...)
triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Help the user with their questions. "
        "If they ask about booking, hand off to the booking agent. "
        "If they ask about refunds, hand off to the refund agent."
    ),
    handoffs=[booking_agent, refund_agent],
)
"""



#Agent(key="value"), A LLM configured with instructions and tools
#Context: It's like giving the agent a "memory" or "profile" of the current user.
#Dependency-injection mechanism that acts as shared container for data and services during agent's execution lifecycle.
"""
class UserContext:
    name: str
    uid: str
    is_pro_user: bool
    async def fetch_purchases() -> list[Purchase]:
        return ...
agent = Agent[UserContext](
    ...,
)
This class represents runtime state + capabilities about a user.
It has:
Data (state)
  name: user’s name
  uid: user ID
  is_pro_user: feature/permission flag  
Behavior (methods)
  fetch_purchases() → async access to user-specific resources (e.g., DB, API)

So UserContext is not just a data container—it’s a handle to user-specific logic.
Dependency Injection (DI) is a design pattern where an object receives its dependencies (other objects it needs) from an external source instead of creating them itself
"""

#Parameters:
"""
key:
  name="string that identifies your agent"
  instructions="developer message ?or system prompt (The instruction should be the context)"
  model="which LLM to use"
  tools=[functions or (agent as tools)]
  output_type=""Any type that can be wrapped in a Pydantic(or a object of this type) TypeAdapter-dataclasses, lists, TypeDict
  model_settings=ModelSettings(tool_choice)
    {auto, which allows the LLM to decide whether or not to use a tool.
    required, which requires the LLM to use a tool (but it can intelligently decide which tool).
    none, which requires the LLM to not use a tool.
    Setting a specific string e.g. my_tool, which requires the LLM to use that specific tool.}
  tool_use_behavior="":
    {"run_llm_again": The default. Tools are run, and the LLM processes the results to produce a final response.
    "stop_on_first_tool": The output of the first tool call is used as the final response, without further LLM processing.
    StopAtTools(stop_at_tool_names=[...]): Stops if any specified tool is called, using its output as the final response.
    ToolsToFinalOutputFunction: A custom function that processes tool results and decides whether to stop or continue with the LLM.
    -Check OpenAI Agent SDK for application.
    }
"""

#Dynamic instructions
"""
The function will receive the agent and context, and must return the prompt(defined by instructions).
def dynamic_instructions(context: RunContextWrapper[UserContext], agent: Agent[UserContext]) -> str:
  return f"The user's name is {context.context.name}. Help them with their questions."
agent = Agent[UserContext](
  name="Triage agent",
  instructions=dynamic_instructions,
)
"""

##Running Agent(from agent import Runner)
#When you use the run method in Runner, you pass in a starting agent and input. 
#The runner then runs a loop:
"""
We call the LLM for the current agent, with the current input.
The LLM produces its output.
  If the LLM returns a 'final_output', the loop ends and we return the result.
  If the LLM does a handoff, we update the current agent and input, and re-run the loop.
  If the LLM produces tool calls, we run those tool calls, append the results, and re-run the loop.
"""
#Runner.run() or Runner.run_sync(), which runs async and returns a RunResult(from RunResultBase).
#Result=Runner.run(...)
#Runner.run_sync(), which is a sync method and just runs Runner.run() under the hood.
"""
Parameter:
run(
    input: Union[str, dict, List, Any],  # Required
    context: TContext | None = None,
    starting_agent: Agent[TContext],
    run_config: Optional[RunnableConfig] = None(Find usage online)
      {model: Allows setting a global LLM model to use, irrespective of what model each Agent has.
      input_guardrails, output_guardrails: A list of input or output guardrails to include on all runs.
      tracing_disabled: Allows you to disable tracing for the entire run.
      }
) -> Any
async def main(): #main() must be async when using await Runner.run() or await orchestrator_agent.ainvoke()
    result = await Runner.run(orchestrator_agent, input="Say 'Hello, how are you?' in Spanish.")
    print(result.final_output)
"""
"""
Runresult(dataclass):
Result.last_agent: The last agent that was run( Agent[Any])
Result.input : The original input items i.e. the items before run() was called. This may be a mutated version of the input, if there are handoff input filters that mutate the input(str | list[TResponseInputItem])
Result.raw_responses: The raw LLM responses generated by the model during the agent run(list[ModelResponse]).
Result.final_output: The output of the last agent.
Result.input_guardrail_results or output_guardrail_results: Guardrail results for the input/output messages.
"""

#Running agent - agents.exceptions:
"""
AgentsException: This is the base class for all exceptions raised within the SDK. It serves as a generic type from which all other specific exceptions are derived.
UserError: This exception is raised when you (the person writing code using the SDK) make an error while using the SDK. This typically results from incorrect code implementation, invalid configuration, or misuse of the SDK's API.
InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered: This exception is raised when the conditions of an input guardrail or output guardrail are met, respectively.
ModelBehaviorError: This exception occurs when the underlying model (LLM) produces unexpected or invalid outputs. This can include:
  -Malformed JSON: When the model provides a malformed JSON structure for tool calls or in its direct output, especially if a specific output_type is defined.
  -Unexpected tool-related failures: When the model fails to use tools in an expected manner
"""


# Tools(let agents take actions)
#3 Classes of Tools:
#1.Agent as tools: allows you to use an agent as a tool, allowing Agents to call other agents without handing off to them(check online for applications)
#2.Hosted tools:these run on LLM servers alongside the AI models.
"""
OpenAI offers a few built-in tools when using the OpenAIResponsesModel:
The WebSearchTool lets an agent search the web.(from agents import ...)
  The FileSearchTool allows retrieving information from your OpenAI Vector Stores.
  The ComputerTool allows automating computer use tasks.
  The CodeInterpreterTool lets the LLM execute code in a sandboxed environment.
  The HostedMCPTool exposes a remote MCP server's tools to the model.
  The ImageGenerationTool generates images from a prompt.
  The LocalShellTool runs shell commands on your machine.
"""
#3.Function tools
"""
Ex:
class Location(TypedDict):
    lat: float
    long: float
@function_tool  
async def fetch_weather(location: Location) -> str:  
    #Fetch the weather for a given location.
    #Args:
        #location: The location to fetch the weather for.  
    # In real life, we'd fetch the weather from a weather API
    return "sunny"
@function_tool(name_override="fetch_data")  
def read_file(ctx: RunContextWrapper[Any], path: str, directory: str | None = None) -> str:
    #Read the contents of a file.
    #Args:
        #path: The path to the file to read.
        #directory: The directory to read the file from.
    
    # In real life, we'd read the file from the file system
    return "<file contents>"
agent = Agent(
    name="Assistant",
    tools=[fetch_weather, read_file],  
)
"""


#Tracing:The Agents SDK includes built-in tracing, collecting a comprehensive record of events during an agent run: LLM generations, tool calls, handoffs, guardrails, and even custom events that occur.
#Tracing is enabled by default. There are two ways to disable tracing:
#--You can globally disable tracing by setting the env var OPENAI_AGENTS_DISABLE_TRACING=1
#--You can disable tracing for a single run by setting agents.run.RunConfig.tracing_disabled to True
"""
Traces represent a single end-to-end operation of a "workflow". They're composed of Spans. Traces have the following properties:
  workflow_name: This is the logical workflow or app. For example "Code generation" or "Customer service".
  trace_id: A unique ID for the trace. Automatically generated if you don't pass one. Must have the format trace_<32_alphanumeric>.
  disabled: If True, the trace will not be recorded.
  metadata: Optional metadata for the trace.
"""
"""
Spans represent operations that have a start and end time. Spans have:
  'started_at' and 'ended_at' timestamps.
  trace_id, to represent the trace they belong to
"""
"""
from agents import Agent, Runner, trace
async def main():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")
    #use the trace as a context manager, i.e. with trace(...) as my_trace. This will automatically start and end the trace at the right time.
    with trace("Joke workflow"): 
        first_result = await Runner.run(agent, "Tell me a joke")
        second_result = await Runner.run(agent, f"Rate this joke: {first_result.final_output}")
        print(f"Joke: {first_result.final_output}")
        print(f"Rating: {second_result.final_output}")
"""
