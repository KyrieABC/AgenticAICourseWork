# Docker
  - **Eliminates environment inconsistencies**
  - **Package complete environments**: Code, dependencies, configurations travel together as one portable unit
  - **Enables reproducible experiments**: Critical for scientific validity and debugging in complex ML systems

## Docker Image (foundation of containerization)
  - Blueprints for running application
  - Docker Image provides:
    1. A complete package containing OS libraries, dependencies, configuration and code
    2. Immutability once built, ensuring consistency across environments
    3. Shareable artifacts via registris

## What is a Docker Container
  1. Docker container is a running instance of Docker Image
  2. **Lifecycle Management**: Container can be started, stopped, paused, or deleted as needed without affecting the host system
  3. **Resource Efficiency**: Multiple containers can run simultaneously on one machine, each with its own isolated environment
### Analogy
  - Image as Recipe (instructions for creating something)
  - Container as Prepared Meal (the actual instance that is used)

## Docker Workflow
  1. **Build**: Create an image from Dockerfile by `docker build`
  2. **Run**: Start a container from an image using `docker run`
  3. **Share**: Distribute images via registries using `docker push/pull`
  4. **Deploy**: Orchestrate containers in production using Kubernetes

## Key Docker Command
**Command Structure**: `docker [command][options][arguments]`
  - Getting help: `docker [command] --help` (Access documentation for any command)
  - `docker images`: List all images on your system
  - `docker ps -a`: List all containers
  - `docker build -t myimage .`: Build an image from a Dockerfile
    - `docker build`: Tells the docker engine to start the process of creating a container
    - `-t myimage`: Assignment a name to your image
    - `.`: Tells Docker to look in the current directory for `Dockerfile` and any file needed for that build
    - Docker read the `Dockerfile` in current folder, runs each instruction one by one, each step creates an 'layer' in image, once finished the final imageis stored in local Docker library name `myimage`
  - `docker run -it myimage`: Run a container interactively
    - `docker run`: Docker create and start a new container from an image
    - `-t`: Keeps the standard input open even if not attached
    - `-t`: Allocates a pseudo-terminal, makes it look and feel like a real terminal session
    - `myimage`: name of the image you want to use to start container
    - Docker creates a writable container layer over read-only image, it assigns a local IP address and network interface to container, it executes the default startup command defined in the `Dockerfile`, (because of `-it`) you no longer typing commands on host terminal, instead you are typing them inside isolated container environments
  - `docker exec -it (container_id) bash`: Enter a running container
    - `docker exec`: connects you to existing container
    - `(container_id)`: unique ID or name of specific container you want to enter
    - `bash`: tell Docker to start a bash shell once inside
    - First run `docker ps` to see list of running containers and grab `(container_id)`, run the command
  - Ex: *`docker run --gpus all -it \pytorch/pytorch:2.10-cuda12.1-cudnn8 \bash*
    - Pulls the official PyTorch image from Docker Hub, maps host GPUs to the container, launches interactive bash shell

## Dockerfile - Build Your Own Image
  - Dockerfile is a text file containing instructions on build a custom Docker image
  - Common Instruction:
    1. `FROM`: Base image to build upon
    2. `RUN`: Execute commands in image
    3. `COPY`: Add files from host to image
    4. `WORKDIR`: Set working directory
    5. `ENV`: Set environment variables
    6. `EXPOSE`: Document network ports
    7. `CMD`: Default command to run

## Containers vs Virtual Environments
| Feature | Virtual Environments (e.g., venv, conda) | Docker Containers |
| :--- | :--- | :--- |
| **Isolation Level** | Python-only isolation | Complete system-level isolation |
| **Dependencies** | No isolation for non-Python dependencies | Packages OS libraries, CUDA drivers, etc. |
| **Portability** | Limited portability across systems | Highly portable across systems |
| **Consistency** | Uses same OS and system libraries as host | Works consistently in production |
| **Setup** | Lightweight and simple to set up | Shareable with exact reproducibility |

## Best practice for AI containers
  1. Use official Base Image
  2. Keep Images small
  3. Pin specific Versions
  4. Don't run as root (create a non-root user inside containers for better security)
  5. Automate Builds: Implement CI/CD pipelines to automatically build, test, and push images when code changes

## Why containerize AI App
1. **Portability**
2. **Deployment Simplicity**: Streamlines deployment processes and enables easy scaling
3. **Reproducibility**: Guarantees identical environments across development and production
4. **API foundation**: Creates the foundation for serving AI models as production-ready APIs

## Example AI App - Image Classifier
**Build containerized image classification service that**:
  - Accept image uploads from users
  - Processes them with pre-trained CNN models
  - Returns predicted labels
  - Runs in a FastAPI server
### Example container
```
ai-app/
  |--- app.py #FastAPI code
  |--- model.pt #saved PyTorch
 model
  |--- requirement.txt
  |--- Dockerfile
```
### Project Structure Design Principle
  - **Self-contained**: Everything the app needs is in one directory
  - **Versioned Dependencies**: Requirements specify exact package versions
  - **Container Blueprints**: Dockerfile defines the build process
### The requirements.txt
  - **Dependency Management**: Explicitly defines all Python packages needed to run the application
  - **Version Pinning**: Specifies exact versions to ensure consistent behavior across environments
  - **Reproducibility**: Guarantees that anyone building the container gets identical dependencies
### app.py
  - **API logic**:
    1. Load pre-trained PyTorch model
    2. Set up image transformation pipeline 
    3. Create endpoint for image prediction
  - **Key Features**:
    1. Asynchronous handler for better performance
    2. Simple JSON response format
    3. Minimal code for maximum clarity
### the Dockerfile
  - **Base Image**
  - **Dependencies**: Install requirements first (for better caching)
  - **Application Code**: Copy app code and model file
  - **Launch Command**: Start FastAPI server on container boot
### Build the Image
`docker build -t ai-app:latest .`
  - `ai-app:latest`: Assigns the name `ai-app` and version label `latest` to the image
  -`.`: tell Docker that the `Dockerfile` and all necessary file are in current directory
  - Verify with `docker images` too see your new AI app image
### Run the container
`docker run -it -p 8000:8000 ai-app:latest`
  - `-p 8000-8000`: maps container port to host port
  - `ai-app:latest`: specifies the image to run
  - Your app is now accessible at https://localhost:8000/docs
  - `/predic` endpoints: click to expand it
  - Check container logs in your terminal

## Networking and Volumes in Docker (Enable scalable, stateful AI application)
  - Containers are isolated by default, creating secure but disconnected environments
  - Networking connects containers to users, services, and each other
  - Volumes persist data beyond container lifecycle, essential for stateful application

### Docker Networking
1. **Private IPs**: Each container get its own private IP address
2. **Default Bridge**: Default network is bridge, enabling basic communication
3. **Network Access**: Container can communicate if on the same network
  - **Port Mapping**: linking a port on host machine (your computer or server) to a port inside the isolated container
    - Without mapping, services are only accessible internally
