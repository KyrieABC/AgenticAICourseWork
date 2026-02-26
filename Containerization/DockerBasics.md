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
  - Ex: `docker run --gpus all -it \pytorch/pytorch:2.10-cuda12.1-cudnn8 \bash`
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
  - `.`: tell Docker that the `Dockerfile` and all necessary file are in current directory
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
  - Essential for inference endpoints that need external requests
4. **Custom Docker Network**
  - Perfect for complex ML systems with databases, APIs and worker containers
  - Containers on same custom network can:
    1. Communicate using container names as hostnames 
    2. Maintain isolation from other containers
    3. Establish secure multi-service AI pipelines
  ```
  docker network create ai-net
  docker run -d --name db --network ai-net postgres
  docker run -d --name app --network ai-net ai-app
  ```

### Docker Volumes
  - **Containers are temporary by design. Without volumes, all data is lost when containers are deleted or restarted**
  - **Volumes keep datasets, trained models, and logs safe--critical for ML workflows**
- **Persist data on host system**
- **Mount with special syntax**
```
docker run -v /host:/container/data ai-app
```
  - Map host path to container path

#### Named Volumes
  - Named volumes are perfect for shared datasets or model checkpoints, preventing dulplicated data and improving portability
1. **Create Named Volume**: Creates a managed volume with Docker
  ```
  docker volume create model-store
  ```
2. **Use in Container**: Mounts volume at specified container path
  ```
  docker run -v model-store:/app/models ai-app
  ```
3. **Share Across Container**: Same volume used in multiple containers
  ```
  docker run -v model-store:/models inference-api
  ```

#### Bind Mounts
  - Map host directly to container
  - **Careful: Bind mounts can overwrite container files! Use carefully in production environments**
```
docker run -v $(pwd)/notebooks:/workspace jupyter
```
  - Syncs local code changes instantly
  - Edit with your favorite IDE
  - Container immediately sees changes
  - No rebuild needed for code tweaks

#### Use Volumes for AI models
  - This approach saves bandwidth, reduces costs, and boosts speed in production environments
1. **Store Large Models**
2. **Share Across Containers**: Multiple inference containers acecss same model files
3. **Optimize Performance**: Volume acts as "model cache" - no repeated downloads

### Combining Networks + Volumes
1. API container handles user requests via port mapping
2. Database container stores persistent application data
3. Model container performs inference using shared models
4. All components communicate securely on private network

## Docker Compose for Multi-Service AI Systems
  - **One YAML file**: Define entire application stack in a single configuration file
  - **One Command**: Spin up your complete development environment with a single terminal command
  - **AI Systems Need Multiple Services**:
    1. **Model Server**: model inference
    2. **API Gateway**: handling client requests
    3. **Database**: storing data and results
    4. **Worker**: Background processes for data preprocessing or async tasks

### Docker Compose
1. **Define services, networks, and volumes**: configure each component of your AI system
2. **Version control friendly**: Keep infrastructure as code alongside your ML code
3. **Reproduce ML pipelines**: Ensure consistent development environments across team

#### Common Commands
1. **Start all services**: `docker-compose up`
2. **Run in background**: `docker-compose up -d`
3. **Stop and remove**: `docker-compose down`
  - Ex:
    - **API Service**: Custom build from local directory
    - **Dependency**: API waits for model to start
    - **Model Service**: pre-built PyTorch image with volume mount
```
version: "3.9"
services:
  api: 
    build: ./api
    ports:
      - "8000:8000"
    depends_on:
      - model
  model:
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8
    volumes:
      - ./models:/models
```

### Network in Compose
1. **Default Network**: Compose automatically creates a dedicated network for your application
2. **Service Discovery**: Services can call each other using service names: `https://model:5000/predict`
3. **No Manual IP Management**: No need to track container IPs or set up complex networking manually

### Volumes in Compose
1. **Persistent Storage**: Maintain data across container restarts
2. **Shared Resources**: Store model weights, checkpoints, and datasets
3. **Training -> Inference**: Perfect for AI workflows that generate and then use models
```
volumes: 
  model-store:
  services:
    volumes:
      - model-store:/models
```

### Scaling Services
  1. Run multiple instances of the same service
  2. Spread inference requests across multiple containers
  3. Scale different components independently based on bottlenecks
  ```
  docker-compose up --scale api=3
  ```

### Advantage of Docker Compose
1. **Automation**: Eliminates tedious manual container setup and linking
2. **Reproducibility**: Ensure ML pipeline work identically across environments
3. **Collaboration**: Share complete environments with one YAML file
4. **Local Testing**: Test infrastructure locally before cloud deployment
5. **Dev -> Prod Bridge**: Creates consistency between development and production

### Good practice for Docker compose
1. Use `.env` files for secrets and configuration
2. Configure automatic restarts for failed containers
3. Version Pinning: Specify exact image versions for stability
4. Document compose workflows in your README
5. Split large systems into multiple compose files
6. Set memory and CPU constraints for stability

## Lightweight AI Containers
1. **Faster Builds & Deployments**: Smaller images dramatically reduce CI/CD pipeline time and deployment latency
2. **Cost Efficiency**: Lower storage and network transfer costs in cloud environments
3. **Enhanced Security**: Reduced attack surface with fewer unused packages and dependencies
4. **Efficient Scaling**: Faster cold-starts make scaling inference endpoints more responsive

### Common Problems with Heavy AI container
1. **Bloated Images**: 10+ GB containers with numerous unused packages and libraries
2. **Registry Bottlenecks**: Painfully slow push/pull operations to container registries, especially in distributed teams
3. **Performance Issues**: Extended cold-start times for inference services, creating user-visible latency
4. **Security Vulnerabilities**: Hidden security risk in unused layers and dependencies
5. **Resource Waste**: Inefficient GPU utilization due to container overhead
   - **Solutions**:
  - Choose Minimal Base Images (purpose-built images)
  Ex:
    - Slim Python: removes dev tools and extras while keeeping core functionality
    - Ultra Light Options: can reduce base layer by hundred of MBs
    - Official ML Images: Pre-optimized with proper GPU dependencies

### Multi-Stage Builds
  - **Most effective technique for reducing image size**
1. **Before: Single-stage build**
  - Includes all build tools, test packages, and build artifacts in final image
  ```
  FROM python:3.10
  COPY ./app
  WORKDIR /app
  RUN pip install -r requirements.txt
  RUN pip install pytest
  RUN pytest
  CMD ["python","app.py"]
  ```
2. **After: Multi-stage Build**
  - Final image contains only runtime dependencies
  ```
  FROM python:3.10 AS builder
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  FROM python:3.10-slim
  COPY --from=builder /usr/local/lib/python3.10/site-packages/usr/local/lib/python3.10/site-packages/
  COPY ./app /app
  WORKDIR /app
  CMD ["python","app.py"]
  ```

### Minimize Dependencies
1. Dependency OptimizationL
  - Pin exact versions of only needed libraries
  - Remove test/dev packages from production images
  - Use `pip install --no cache-dir` to avoid caching pip files
  - Avoid mixing conda + pip installation
  - Consider specialized inference-only libraries
2. Regular Maintainance
**Implement dependency auditing in workflow**
  - Run `pip list-outdated` periodically
  - Use dependency scanning tools
  - Create separate dev and prod requirement files
  - Document the purpose of each dependency

### Layer Caching for Faster Builds
  - Install dependencies before copying code to leverage Docker's layer caching
  - **Each layer in Dockerfile is cached. When a layer changes, all subsequent layers must rebuilt**
    - Proper ordering ensures dependencies are cached, dramatically speeding up iterative builds
   ```
   FROM python:3.10-slim
   #Dependency change less frequently
   COPY requirements.txt
   RUN pip install -r requirements.txt
   #Code changes more frequently
   COPY ..
   CMD ["python","app.py"]
   ```

### Security in lightweight Containers
1. **Remove Root Privileges**: 
  - Create and use non-root users in your containers:
  ```
  RUN useradd -m appuser
  USER appuser
  ```
  - Limites potential damage from container breakouts
2. **Regular Updates**
  - Keep base images updated with security patches:
  ```
  docker pull python:3.10-slim
  ```
  - Schedule automated rebuilds to incorporate lateset patches
3. **Vulnerability Scanning**
  - Integrate scanning into CI/CDL
  ```
  docker scan myimage:latest
  # or
  trivy image myimage:latest
  ```
  - Catches vulnerabilities before deployment

### Optimize for GPU Containers
1. **Use Nvidia Official image**: `nvidia/cuda:11.8.0-runtime-ubuntu22.04`
  - Pre-optimized with proper driver compatibility
2. **Avoid Full CUDA Toolkit**: The runtime version is much smaller than the development toolkit
3. **Match CUDA Versions**: Ensure CUDA/cuDNN versions match your framework requirements exactly
  - Use `--gpus all` flag when running
  - Test with `nvidia-smi` inside container
  - Monitor GPU memory usage to ensure efficiency
  - Consider TensorRT for inference optimization
