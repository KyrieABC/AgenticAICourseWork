#Every time that i run "crewai run", it creates its own python virtual environments.
#Remember to set your .env key and put it in gitignore

#Create: crewai create crew latest-ai-development(name)
#change directory:cd latest_ai_development      in terminal
"""
To Run: crewai run
from latest_ai_development.crew import LatestAiDevelopmentCrew
def run():
  Run the crew.
  inputs = {
    'topic': 'AI Agents'
  }
  LatestAiDevelopmentCrew().crew().kickoff(inputs=inputs)
"""

"""
The names you use in your YAML files (agents.yaml and tasks.yaml) should match the method names in your Python code. For example, you can reference the agent for specific tasks from tasks.yaml file.
@agent
def reporting_analyst(self) -> Agent:
  return Agent(
    config=self.agents_config['reporting_analyst'], # type: ignore[index]
    verbose=True
  )
@task
def research_task(self) -> Task:
  return Task(
    config=self.tasks_config['research_task'], # type: ignore[index]
  )
same method names with _config['Name']
"""
#Include context in tasks
"""
email_summarizer_task:
  description: >
    Summarize the email into a 5 bullet point summary
  expected_output: >
    A 5 bullet point summary of the email
  agent: email_summarizer
  context:
    - reporting_task
    - research_task
"""

#agents.yaml('>' for multi-line)
"""
Attributes:
  #Required
  role:str-Defines the agent’s function and expertise within the crew.
  goal:str-The individual objective that guides the agent’s decision-making.
  backstory:str-Provides context and personality to the agent, enriching interactions.
  #Optional
  verbose:bool-Enable detailed execution logs for debugging. Default is False.
  llm:Union[str, LLM, Any]-Language model that powers the agent. Defaults to the model specified in OPENAI_MODEL_NAME or “gpt-4”.
  cache:bool-Enable caching for tool usage. Default is True.
  system_template,prompt_template,response_template:Optional[str]-Custom system/prompt/response prompt template for the agent.
  reasoning:bool-Whether the agent should reflect and create a plan before executing a task. Default is False.
  tools:List[BaseTool]-Capabilities or functions available to the agent. Defaults to an empty list.
"""
#CrewAI includes sophisticated automatic context window management to handle situations where conversations exceed the language model’s token limits.
"""
When an agent’s conversation history grows too large for the LLM’s context window, CrewAI automatically detects this situation and can either:
  Automatically summarize content (when respect_context_window=True)
  Stop execution with an error (when respect_context_window=False)
"""
#Variable in yaml replaced like: crew.kickoff(inputs={'topic': 'AI Agents'})
#tasks.yaml
"""
Attributes:
  #Required
  description:str- A clear, concise statement of what the task entails.
  expected_output:str-A detailed description of what the task’s completion looks like.
  #Optional
  name:Optional[str]- A name identifier for the task.
  agent:Optional[BaseAgent]-The agent responsible for executing the task.
  tools:List[BaseTool]-The tools/resources the agent is limited to use for this task.
  context:Optional[List["Task"]]-Other tasks whose outputs will be used as context for this task.
  config:Optional[Dict[str, Any]]-Task-specific configuration parameters.
  output_file:Optional[str]-File path for storing the task output.
"""
#Task Output:Once a task has been executed, its output can be accessed through the output attribute of the Task object.
"""
# Example task(alternaively without yaml)
task = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool]
    output_pydantic=A structured output
)
# Execute the crew
crew = Crew(
    agents=[research_agent],
    tasks=[task],
    verbose=True
)
result = crew.kickoff()
# Accessing the task output
task_output = task.output
"""
"""
output attributes:
  description:str-Description of the task.
  summary:Optional[str]-Summary of the task, auto-generated from the first 10 words of the description.
  raw:str-The raw output of the task. This is the default format for the output.
  output_format:OutputFormat-The format of the task output, with options including RAW, JSON, and Pydantic. The default is RAW.
  messages:list[LLMMessage]-The messages from the last task execution.
"""


#crews:collaborative group of agents working together to achieve a set of tasks.
"""
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class YourCrewName:
    Description of your crew
    agents: List[BaseAgent]
    tasks: List[Task]
    # Paths to your YAML configuration files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    @before_kickoff
    def prepare_inputs(self, inputs):
        # Modify inputs before the crew starts
        inputs['additional_data'] = "Some extra information"
        return inputs
    @after_kickoff
    def process_output(self, output):
        # Modify output after the crew finishes
        output.raw += "\nProcessed after kickoff."
        return output
    @agent
    def agent_one(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_one'], # type: ignore[index]
            verbose=True
        )
    @agent
    def agent_two(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_two'], # type: ignore[index]
            verbose=True
        )
    @task
    def task_one(self) -> Task:
        return Task(
            config=self.tasks_config['task_one'] # type: ignore[index]
        )
    @task
    def task_two(self) -> Task:
        return Task(
            config=self.tasks_config['task_two'] # type: ignore[index]
        )
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  # Automatically collected by the @agent decorator
            tasks=self.tasks,    # Automatically collected by the @task decorator.
            process=Process.sequential,#or hierarchical(based on roles and expertise)
            verbose=True,
        )
"""
"""
Attributes:
  #Required
  tasks=A list of tasks assigned to the crew.
  agents=A list of agents that are part of the crew.
  #Optional
  process=The process flow (e.g., sequential, hierarchical) the crew follows. Default is sequential.
  verbose=The verbosity level for logging during execution. Defaults to False.
  config=Optional configuration settings for the crew, in Json or Dict[str, Any] format.
  manager_agent=manager sets a custom agent that will be used as a manager.
  memory=Utilized for storing execution memories (short-term, long-term, entity memory).
  -More on this(memory=True, False by default)
"""
"""
Decoraters:
  #Required:
  @CrewBase:Marks the class as a crew base class.
  @agent:Denotes a method that returns an 'Agent' object.
  @task:Denotes a method that returns a Task object.
  @crew:Denotes the method that returns the Crew object.
  #Optional
  @before_kickoff:Marks a method to be executed before the crew starts.
  @after_kickoff:Marks a method to be executed after the crew starts.
"""
#crew_output = crew.kickoff()
"""
Attribute:
  crew_output.raw:str-The raw output of the crew. This is the default format for the output.
  crew_output.pydantic:Optional[BaseModel]-A Pydantic model object representing the structured output of the crew.
  crew_output.json_dict:Optional[Dict[str, Any]]-A dictionary representing the JSON output of the crew.
"""



