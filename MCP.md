# MCP (Model Context Protocol)
  - **An open-source standard for connecting AI applications to external systems**

## MCP Architecture

### Scope
**MCP includes the following projects:**
  - **MCP Specification**: A specification of MCP that outlines the implementation requirements for clients and servers
  - **MCP SDKs**: SDKs for different programming languages that implement MCP
  - **MCP Development Tools**: Tools for developing MCP servers and clients, including the MCP Inspector
  - **MCP Reference Server Implementation**: Reference implementations of MCP servers

### Concepts of MCP
**Participants:**
  - **MCP Host**: The AI application that coordinates and manages one or multiple MCP clients
    - Ex: Claude Code/Desktop, IDE
  - **MCP Client**: A component that maintains a connection to an MCP server and obtains context from an MCP server for the MCP host to use
  - **MCP Server**: A program that provides context to MCP clients
    - MCP server refers to the program that serves context data, regardless of where it runs (locally/remotely)
  - *MCP Host creating one MCP client for each MCP server*
  - *Each MCP client maintains a dedicated connection with its corresponding MCP server*
  - *MCP client is created at runtime by a host application*
    - Allows the AI to dynamically discover and use server capabilities, such as available tools and resources, without needing these integrations to be hardcoded at build time
![MCP Architecture](https://themlarchitect.com/wp-content/uploads/2025/06/mcp.drawio-1.png)

### Layers
  - **Data Layer**: Defines the JSON-RPC based protocol for client-server communication
    - **Lifecycle Management**: Handles connection initialization, capability negotiation, and connection termination between clients and servers
    - **Server features**: Enables servers to provide core functionality including tools for AI actions, resources for context data, and prompts for interaction templates from and to the client
    - **Client Feature**: Enables servers to ask the client to sample from the host LLM, elicit input from the user, and log messages to the client
    - **Utility Features**: Supports additional capabilities like notifications for real-time updates and progress tracking for long-running operations
  - **Transport Layer**:  Defines the communication mechanisms and channels that enable data exchange between clients and servers
    - Transport-specific connection establishment
    - Message framing
    - Authorization
    - (Support Mechanism) **Stdio transport**:  Uses standard input/output streams for direct process communication between local processes on the same machine
    - (Support Mechanism) **Streamable HTTP transport**: Uses HTTP POST for client-to-server messages with optional Server-Sent Events for streaming capabilities
  
### Data Layer Protocol
  - *A core part of MCP is defining the schema and semantics between MCP clients and MCP servers*
#### Lifecycle Management (MCP is a stateful protocol)
  - The purpose of lifecycle management is to negotiate the capabilities that both client and server support
  - Client sends an (initialize) request to establish the connection and negotiate supported features
#### Primitives
  - Define what clients and servers can offer each other
  - Three core primitives that servers can expose to client:
    1. **Tools**: Executable functions that AI applications can invoke to perform actions
    2. **Resources**: Data sources that provide contextual information to AI applications
    3. **Prompts**: Reusable templates that help structure interactions with language models
  - Each primitive type has associated methods for discovery(*/list), retrieval(*/get), in some case execution(tools/call)
  - MCP clients use */list methods to discover available primitives
  - Ex: **An MCP server that provides context about a database, expose tools for querying the database, resource that contains the schema of the database... 
  - Primitives that clients can expose to Server:
    - 1. **Sampling**: Allows servers to request language model completions from the client’s AI application. Use (sampling/complete) method to request a language model completion from client's AI application (For model-independency and not include a SDK in server)
    - 2. **Elicitation**: Allows servers to request additional information from users. Use (elicitation/request) method to request additional information from user
    - 3. **Logging**: Enables servers to send log messages to clients for debugging and monitoring purposes
#### Notification
  - **The protocol supports real-time notifications to enable dynamic updates between servers and clients**

### Core Message Types:
#### Initialization Messages
  - **`initialize` Request**: Establishes connection and negotiates protocol version and capabilities
  - **initialize` Response**: Confirms supported features and server information
  - **`notification/initialized`**: Signals that initialization is complete and the session is ready
#### Discovery Messages
  - **`tools/list` Request**: Discovers available tools from the server
  - **`resources/list` Request**: Lists available resources (data sources)
  - **`prompts/list` Request**: Retrieves available prompt templates
#### Execution Messages
  - **`tools/call` Request**: Executes a specific tool with provided parameters
  - **`resources/read` Request**: Retrieves content from a specific resource
  - **`prompts/get` Request**: Fetches a prompt template with optional parameters
#### Client-side Messages
  - **`sampling/complete` Request**: Server requests LLM completion from the client
  - **`elicitation/request`**: Server requests user input through the client interface
  - **Logging Messages**: Server sends structured log messages to the client
#### **Notification Messages**
  - **`notifications/tools/list_changed`**: Server notifies client of tool changes
  - **`notifications/resources/list_changed`**: Server notifies client of resource changes  
  - **`notifications/prompts/list_changed`**: Server notifies client of prompt changes

### Data Layer Examples:
1. **Initialization (Lifecycle Management)**
  - Initialize Request (From Host to Server  to establish capabilities and constraints):
```
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "elicitation": {}
    },
    "clientInfo": {
      "name": "example-client",
      "version": "1.0.0"
    }
  }
}
```
  - Initialize Response (Server's handshake back to Host to confirm the connection):
```
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "tools": {
        "listChanged": true
      },
      "resources": {}
    },
    "serverInfo": {
      "name": "example-server",
      "version": "1.0.0"
    }
  }
}
```
  - **Protocol Version Negotiation**: The (protocolVersion) field ensures both client and server are using compatible protocol versions
  - **Capability Discovery**: (capabilities) object allows each party to declare what features they support, including which primitives they can handle and whether they support features like *notification*
  - **Identity Exchange**: (clientInfo) and (serverInfo) objectts provide identification and versioning information for debugging and compatibility purposes
  - Client Capabilities
    - (elicitation: {}): The client declares it can work with user interaction requests, receive (elicitation/create) method calls
  - Server Capabilities
    - **("tools": {"listChanged": true})**: The server supports the tools primitive AND can send (tools/list_changed) notification when tool lists changes
    - **("resources": {})**: The server also supports the resources primitive, can handle (resources/list) and (resources/read) method
  - After  successful initialization, the client sends a notification to indicate it’s ready:
```
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```
  - During initialization, the AI application’s MCP client manager establishes connections to configured servers and stores their capabilities for later use
  - The application uses this information to determine which servers can provide specific types of functionality (tools, resources, prompts) and whether they support real-time updates
2. **Tool Discovery (Primitives)**
  - Since connection is established, the client can discover available tools by sending a (tools/lists) request
  - Tool List Request:
```
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list"
}
```
  - Tool List Response:
```
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "calculator_arithmetic",
        "title": "Calculator",
        "description": "Perform mathematical calculations including basic arithmetic, trigonometric functions, and algebraic operations",
        "inputSchema": {
          "type": "object",
          "properties": {
            "expression": {
              "type": "string",
              "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sin(30)', 'sqrt(16)')"
            }
          },
          "required": ["expression"]
        }
      },
      {
        "name": "weather_current",
        "title": "Weather Information",
        "description": "Get current weather information for any location worldwide",
        "inputSchema": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name, address, or coordinates (latitude,longitude)"
            },
            "units": {
              "type": "string",
              "enum": ["metric", "imperial", "kelvin"],
              "description": "Temperature units to use in response",
              "default": "metric"
            }
          },
          "required": ["location"]
        }
      }
    ]
  }
}
```
  - (name): A unique identifier for the tool within the server’s namespace
  - (title): A human-readable display name for the tool that clients can show to users
  - (description):  Detailed explanation of what the tool does and when to use it
  - (inputSchema): A JSON Schema that defines the expected input parameters, enabling type validation and providing clear documentation about required and optional parameters
  - AI application fetches available tools from connected MCP servers and combines them into a unified tool registry that the language model can access
3. **Tool Execution (primitive)**
  - client can now execute tool using (tool/call) method
  - Tool Call request:
```
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "weather_current",
    "arguments": {
      "location": "San Francisco",
      "units": "imperial"
    }
  }
}
```
  - Tool Call response:
```
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Current weather in San Francisco: 68°F, partly cloudy with light winds from the west at 8 mph. Humidity: 65%"
      }
    ]
  }
}
```
  - (arguments: request): Contains input parameters defined by tool's (input Schema)
  - (content: response): An array of content objects
  - Structured Output: The response provides actionable informatoin that AI application can use as context
  - AI application intercepts the tool call, routes it to the appropriate MCP server, execute it, and returns back to LLM
4. **Real-time Updates (Notification)**
  - MCP supports real-time notifications that enable servers to inform clients about changes (such as available tools change) without being explicitly requested
  - Requests:
```
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed"
}
```
  - No response required
  - **Capability-Based**: This notification is only sent by servers that declared ("ListChanged": true) in their tools capability during initialization
  - **Event-Driven**: The server decides when to send notifications based on internal state changes, making MCP connections dynamic and responsive
  - Client Response to Notification:
```
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/list"
}
```
  - Upon receiving this notification, the client typically reacts by requesting the updated tool list (keep client's understanding of available tools up-to-date)
  - **When the AI application receives a notification about changed tools, it immediately refreshes its tool registry and updates the LLM’s available capabilities**

### Message Structure
  - **Request Message**: Include `id`, `method`, `params`
  - **Response Message**: Include `id`, either `result` or `error`
  - **Notification**: Include `method`, optional `params`

### MCP Server
#### Tools (schema-defined interfaces that LLMs can invoke)
  - `tools/list`: Discover available tools (return array of tool definition with schemas)
  - `tools/call`: Execute a specific tool (Tool execution result)
```
{
  name: "searchFlights",
  description: "Search for available flights",
  inputSchema: {
    type: "object",
    properties: {
      origin: { type: "string", description: "Departure city" },
      destination: { type: "string", description: "Arrival city" },
      date: { type: "string", format: "date", description: "Travel date" }
    },
    required: ["origin", "destination", "date"]
  }
}
```
  - Call: searchFlights(origin: "", destination: "", date: "")
  - MCP emphasizes human oversight through several mechanisms
    - Permission settings for pre-approving certain safe operations
    - Activity logs that show all tool executions with their results
#### Resources (expose data from files, APIs, databases, or any other source that an AI needs to understand context)
  - `resources/list`: List available direct resources (return array of resource descriptor)
  - `resources/templates/list`: Discover resource templates (return array of resource template definition)
  - `resources/read`: Retrieve resource content (return resource data with metadata)
  - `resources/subscribe`: Monitor resource changes (return subscription confirmation)
**Resource support two discovery patterns**
  1. Direct Resources: fixed URLs that point to specific data
  2. Resource Templates: dynamic URIs with parameters for flexible queries
    - Ex: `travel://activities/{city}/{category}` - returns activities by city and category
Ex:
```
{
  "uriTemplate": "weather://forecast/{city}/{date}",
  "name": "weather-forecast",
  "title": "Weather Forecast",
  "description": "Get weather forecast for any city and date",
  "mimeType": "application/json"
}

{
  "uriTemplate": "travel://flights/{origin}/{destination}",
  "name": "flight-search",
  "title": "Flight Search",
  "description": "Search available flights between cities",
  "mimeType": "application/json"
}
```
  - User Interaction Model :
    1. Search and filter interfaces for finding specific resources
    2. Manual or bulk selection interfaces for including single or multiple resources
#### Prompts
  - `prompts/list`: Discover available prompts (return array of prompt descriptions)
  - `prompts/get`: Retrieve prompt details (return full prompt definition with arguments)
Ex:
```
{
  "name": "plan-vacation",
  "title": "Plan a vacation",
  "description": "Guide through vacation planning process",
  "arguments": [
    { "name": "destination", "type": "string", "required": true },
    { "name": "duration", "type": "number", "description": "days" },
    { "name": "budget", "type": "number", "required": false },
    { "name": "interests", "type": "array", "items": { "type": "string" } }
  ]
}
```
  - Structured input: Barcelona, 7 days, $3000, [“beaches”, “architecture”, “food”] by user invoking (Not exactly): 
  ```
  {
  "prompt": "plan-vacation",
  "arguments": 
    {
    "destination": "Barcelona",
    "departure_date": "2024-06-15",
    "return_date": "2024-06-22",
    "budget": 3000,
    "travelers": 2
    }
  }
  ```
**Real power of MCP emerges when multiple servers work together, combining their specialized capabilities through a unified interface**

### MCP Client
#### Elicitation (provides a structured way for servers to gather necessary information on demand)
Ex:
```
{
  method: "elicitation/requestInput",
  params: {
    message: "Please confirm your Barcelona vacation booking details:",
    schema: {
      type: "object",
      properties: {
        confirmBooking: {
          type: "boolean",
          description: "Confirm the booking (Flights + Hotel = $3,000)"
        },
        seatPreference: {
          type: "string",
          enum: ["window", "aisle", "no preference"],
          description: "Preferred seat type for flights"
        },
        roomType: {
          type: "string",
          enum: ["sea view", "city view", "garden view"],
          description: "Preferred room type at hotel"
        },
        travelInsurance: {
          type: "boolean",
          default: false,
          description: "Add travel insurance ($150)"
        }
      },
      required: ["confirmBooking"]
    }
  }
}
```
  - Request: Clients display elicitation requests with clear context about which server is asking, why the information is needed, and how it will be used
  - Response: Users can provide the requested information through appropriate UI controls (text fields, dropdowns, checkboxes), decline to provide information with optional explanation, or cancel the entire operation
    - Clients validate responses against the provided schema before returning them to servers
  - **Elicitation never requests passwords or API keys**
#### Roots (a mechanism for clients to communicate filesystem access boundaries to servers)
  - **Consist of file URIs that indicate directories where servers can operate (Roots are exclusively filesystem paths and always use `file://` URI scheme)**
  - **Root list can be updated dynamically with servers receiving notification through `roots/list_changed` when boundaries change**
  - **Roots serve as a coordination mechanism between clients and servers, not a security boundary**
Structure Ex:
```
{
  "uri": "file:///Users/agent/travel-planning",
  "name": "Travel Planning Workspace"
}
```
  - *Well-behaved servers respect these boundaries—accessing templates, saving the new itinerary, and referencing client documents within the specified roots*
  - If the agent opens an archive folder like `file:///Users/agent/archive/2023-trips`, the client updates the roots list via `roots/list_changed`
#### Sampling (enables servers to perform AI-dependent tasks without directly integrating with or paying for AI models)
Workflow:
  1. Server initialize sampling request
  2. Human-in-the-loop review
    - client present request to user for approval (LLM access)
    - user review and approve/modify
  3. client forward the approved request to LLM (from user since server is model-independent)
  4. LLM return generation to client
  5. client present the response to user for approval
  6. user approve/modify
  7. client return approved response to server
Ex request:
```
{
  messages: [
    {
      role: "user",
      content: "Analyze these flight options and recommend the best choice:\n" +
               "[47 flights with prices, times, airlines, and layovers]\n" +
               "User preferences: morning departure, max 1 layover"
    }
  ],
  modelPreferences: {
    hints: [{
      name: "claude-sonnet-4-20250514"  // Suggested model
    }],
    costPriority: 0.3,      // Less concerned about API cost
    speedPriority: 0.2,     // Can wait for thorough analysis
    intelligencePriority: 0.9  // Need complex trade-off evaluation
  },
  systemPrompt: "You are a travel expert helping users find the best flights based on their preferences",
  maxTokens: 1500
}
```
  - Sampling requests require explicit user consent.
  - Users can set model preferences, configure auto-approval for trusted operations, or require approval for everything. Clients may provide options to redact sensitive information.
  
