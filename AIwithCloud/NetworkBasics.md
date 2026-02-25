# Networking Basics for AI Infra
**Modern AI systems are inherently distributed across multiple nodes and services, requiring robust networking to function effectively**
  - Enables massive training to move efficciently between sorage and compute resources
  - Controls who can access your sensitive models, datasets and inference endpoints
  - Low latency connections dramatically impact model training speed and inference responsiveness

## VPC (Virtual Private Cloud)
**Creates an isolated sectoin of cloud provider's network that you control as foundation of AI infra
  1. **IP Address Management**: Define custom CIDR blocks (IP ranges) to organize your ML infra
  2. **Network Topology Control**: Create subnets, route tables, and gateways to structure traffic flow
  3. **Security Boundary**: Isolate sensitive AI workloads from other systems and the public internet
  4. **Cloud-Provider Integration**: Connect to managed AI services while maintaining security controls
  - **Subsets**: Logical Divisions within your VPC
    - **Public Subsets**: Connected to internet gateway. Host public-facing ML APIs. Run inference endpoints. Host management components
    - **Private Subnets**: No direct internet access. Host training clusters. Store sensitive datasets. Run internal processing jobs
  ![VPC Diagram](https://aws.plainenglish.io/how-to-build-a-vpc-in-aws-f66a5fd63bb5)

## Firewalls
**Act as filters for all traffic entering and exiting your ML infra**
  - **Rule-Based Control**: Specify exactly which traffic is allowed/denied based on IP addresses, ports and protocols
  - **ML Infra Protection**: Prevent unauthorized access to training data, model weights, and inference endpoints
  - **Multi-Tenent Isolation**: Essential for shared GPU clusters serving multiple teams or customers
  - **Security Groups vs Network ACLs**
| Feature | Security Groups | Network ACLs |
| :--- | :--- | :--- |
| **Protection Level** | **Instance-level** firewall protection | **Subnet-level** firewall protection |
| **State** | **Stateful**: Return traffic is automatically allowed | **Stateless**: Return traffic requires explicit rules |
| **Rule Types** | Supports **allow rules only** | Supports **both allow and deny rules** |
| **Application** | Applied to **specific instances** | Applied to **all instances in a subnet** |
| **Evaluation** | Evaluated **collectively** before traffic reaches instance | Evaluated in **numbered order** (first match wins) |

## Load Balancers
  - **Distribute requests across multiple servers, enabling scalable AI systems that can handle varying workloads**
  1. **Horizontal Scaling**: Allow adding more inference servers as demand increase, rather than scaling vertically-(increase the capacity of a single existing resource) 
  2. **High Availability**: Automatically routes around failed nodes, maintaining ML service uptime
  3. **Traffic Management**: Routes requests based on serve health, capacity and response time
  4. **Inference Endpoint Scaling**: Enables auto-scaling groups to handle variable prediction workloads
  - **Types of Load Balancers for AI Workloads**:
### By Protocol Layer
| Layer 4 (Transport) | Layer 7 (Application) |
| :--- | :--- |
| Routes based on IP address and port | Routes based on HTTP/HTTPS headers |
| Handles TCP/UDP traffic | Can direct traffic by URL path |
| Lower overhead, higher throughput | Supports content-based routing |
| Great for raw traffic distribution | Ideal for complex API endpoints |
### By Network Position
| Internal Load Balancers | External Load Balancers |
| :--- | :--- |
| Distribute within private subnets | Public-facing endpoints |
| Not accessible from internet | Distribute internet traffic |
| For microservices communication | Protect from DDoS attacks |
| Used between processing stages | Front public AI APIs |

## Network Best Practice for AI Infra
1. Restrict Administrative Access
2. Protect Data Bases and Storage
3. Encrypt all Traffic 
4. Monitor Network Traffic
5. Infrastructure as Code (Automate network configurations to ensure consistency and reproducibiilty)