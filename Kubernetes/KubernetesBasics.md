# Kubernetes
**Modern AI application present unique infrastructure challenges that can't be solved with traditional deployment methods**:
1. **Distributed Computing**: AI apps rarely run on a single machine, requiring coordination across clusters of CPUs/GPUs
2. **Operational Complexity**: Manual deployment and updateso become unsustainable at scale
3. **Production Reliability**: AI systems need high availability and fault tolerance
  - **Kubernetes** has emerged as the industry standard orchestrator for these challenges

## What is Kubernetes (K8s)
  - Open-source system that automates the deployment, scaling, and management of containerized application
  - **Container-Agnostic**: Works with any container runtime
  - **Declarative Configuration**: Kubernetes handles the implementation details of desired state

### Core Benefits of Kubernetes
1. **Scalability**: Automatically scales applications up or down based on demand and resource utilization metrics
2. **Resilience**: Self-healing system that automatically restarts failed containers and reschedules pods when nodes fail
3. **Portability**: Run workloads consistently across any environment: public cloud, private cloud, or on-premises
4. **Efficiency**: Optimize hardware utilization by packing containers efficiently on nodes based on resource requirements
5. **Automation**: Automates deployment, updates, and rollbacks with minimal human intervention

### Kubernetes for AI workloads
1. **GPU resource management**
  - Kubernetes enables efficient allocation of expensive GPU resources across teams and workload
  1. Schedule and isolates GPU access between pods
  2. Supports fractional GPU allocation with plugins
2. **AI-specific capabilities**
  - Orchestrates distributed training jobs across multiple nods
  - Scales model inference APIs to handle variable traffic
  - Manages data pipelines & streaming applications
  - Enables hybrid deployments across cloud and edge devices

### Kubernetes Architecture
![Architecture](https://kubernetes.io/images/docs/kubernetes-cluster-architecture.svg)

1. Control Plane
  - **API server**: Frontend for the control plane
  - **Scheduler**: Assigns pods to nodes
  - **Controller Manager**: Maintains desired state
  - **etcd**: Distributed key-value store for all cluster data
2. Node Components
  - **Kubelet**: Ensures containers are running in pods
  - **Kube-proxy**: Maintains network rules
  - **Container Runtime**: Runs the container
  - **Pods**: Groups of one or more containers

### Kubernetes in AI pipelines
  - Kubernetes orchestrate the entire ML lifecycle, enabling continuous deployment of new models through integrated CI/CD pipelines
1. **Data ingestion**: Scalable data collection and preprocessing containers
2. **Training**: Distributed GPU workloads with fault tolerance
3. **Inference**: Auto-scaling model serving APIs
4. **Monitoring**: Performance tracking and model drift detection

### Why not just use Docker alone?
**Docker Limitation**
1. **Single Node Only**: operates on a single machine, but AI workloads ned to span multiple servers
2. **No Cluster Management**: Lacks built-in capabilities for managing nodes across a distributed environments
3. **Limited Automation**: No auto-scaling, self-healing, or advanced scheduling features
4. **Manual Operations**: Maintaining hundred of containers manually becomes unmanageable at scale
  - Kubernetes solves these challenges by providing comprehensive platform orchestration at scale

### Kubernetes + MLOps Tools
1. **Kuberflow**: End-to-end ML platform for building, training and serving models on Kubernetes
2. **MLflow**: Experiment tracking model registry and deployment with K8s integration
3. **NVIDIA Triton**: High-performance inference server optimized for kubernetes

## Pods, Nodes, and Clusters
  - Understanding the hierarchy of Kubernetes components is foundational for designing and operating scalable AI infrastructure
1. **Layer Architecture**: Kubernetes uses a modular, layered design that builds from pods to clusters
2. **Core Hierarchy**: Pod -> Node -> Cluster, forms the backbone of all deployments
3. **AI Infrastructure Foundation**: Essential knowledge for scaling ML training and serving systems effectively

### Pods
  - **Smallest Deployable Unit**
  - **Container Encapsulation**: Houses one or more containers that are always scheduled together on the same node
  - **Shared Resources**: All containers within a pod share the same network namespace, storage volumes, and configuration
  - **Ephemeral By Design**: Pods are disposable and can be recreated or restarted at any time - never assume persistence

#### Pods in AI workflow
  - AI workloads are typically divided into specialized pods that each handle a specific par tof ML pipeline
1. **Training Pod**: Runs distributed PyTorch or TensorFlow workers from model training
2. **Inference Pod**: Serves model predictions via RESTful or gRPC APIs
3. **Data Pods**: Handles ETL pipelines and preprocessing operations
4. **Monitoring Pod**: Trakcs GPU utilization, memory usage, and inference latency

### Nodes
1.  **Working Machine**: physical server or virtual machine that provides the compute resources for running pods.
  - Each node runs a kubelet agent that communicates with the control plane and manages pods
2. **Resource Provider**: Nodes contribute their CPU, memory, storage, and GPU resources to the cluster
  - Forms the execution layer where all AI workloads actually run

#### Nodes in AI workflow
  - Kubernetes scheduler places pods on appropriate nodes based on resource requirements and constraints
1. **GPU Nodes**
  - Dedicated to training large AI models and accelerating inference
2. **CPU Nodes**
  - General purpose compute nodes for preprocessing, orchestration, and lightweight inference workloads
3. **Edge Nodes**
  - Specialized nodes deployed closer to data sources for low-latency inference and real-time processing
  - Often resource-constrained but optimzed for specific workloads

### Cluster
  - Collection of nodes managed by a control plane that coordinates all activities within the cluster
1. Provides a unified pool of compute, storage, and networking resources
2. Ensures workloads run reliably acorss the distributed system
3. Abstracts infrastructure complexity from application developers
4. Handles scaling, failover, and resource allocation
5. Represents your entire AI infrastructure environment

#### Cluster in AI workflow
  - **Training Clusters**: Massive clusters of interconnected GPU nodes for training foundation models
  - **Hybrid Clusters**: Mix of CPU and GPU resources optimized for different stages of the ML lifecycle
  - **Scalable Deployment**: From small lab setups to hyperscale production infrastructure serving millions of requests
  - **Multi-Region Clusters**: Geographically distributed nodes for global model serving with low latency

### Pods, Nodes and Clusters Interaction
**Kubernetes Hierarchy**
1. **Pod**: Containerized ML workload (model training job, inference service)
2. **Node**: Machine that provides compute resources to run pods
3. **Cluster**: Collection of nodes managed as a single entity
  - Scheduler matches pod requirements to node capabilities across the cluster, optimizing resource utilization

## Deployments and Services for AI Application
  - **Pod Problem**: Pods are ephemeral by design and can fail at any time, making them unreliable for production AI workloads
  - **Deployment Solution**: Deployments manage pod lifecycles, ensurign your AI application maintain the desired state despite failures
  - **Service Access**: Services provide stable network endpoints, making your AI APIs consistenly accessible

### Deployment
  - Manages pod lifecycles automatically
  - Ensures the desired number of replicas are running
  - Handles updates & rollbacks with zero downtime
  - Abstracts away pod failures through self-healing
  - Provides declarative updates for your AI application
  - Example:
    1. Creates 3 identical replicas of the AI API pod
    2. Self-heals if any pod crashes or become unhealthy
    3. Uses labels to track and manage all related pods
    4. Exposes port 8000 for the application
  ```
  apiVersion: apps/v1
  kind: Deployment
  metadata: 
    name: ai-api
  spec:
    replicas: 3
    selector:
      matchLabels:
        app: ai-api
    template:
      metadata:
        labels:
          app: ai-api
      spec:
        containers:
          - name: api
          images: ai-app:latest
          ports:
            - containerPort: 8000
  ```

#### Deployments for AI workloads
  - **Training jobs**: Often managed via specialized custom controllers or operators like KubeFlow, rather than standard Deployments
  - **Inference APIs**: Run as Deployments to handle variable traffic loads and ensure high availability for model serving
  - **Auto-scaling**: Scale replicas up/down based on CPU, memory, or custom metrics like prediction requeset volume
  - **Model Updates**: Use rolling updates to gradually replace pods running older model versions with zero downtime

### Service
  - Provide a stable network endpoint for accessing pods
  - Load-balances traffic across all available replicas
  - Decouples frontend clients from backend pod implementations
  - Enables discovery through DNS within the cluster
  - Comes in multiple types for different access patterns
  - Ex:
    1. Maps external port 80 to container port 8000
    2. Automatically load balances requests across all replicas
    3. Provides a stable external IP address
```
apiVersion: v1
kind: Service
metadata: 
  name: ai-service
spec:
  selector:
    app: ai-api
  ports:
    - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Types of Services
1. **ClusterIP**: Default type that exposes the Service on an internal cluster IP
  - Only accessible within the cluster
  - Ideal for internal AI microservices
2. **NodePort**: Exposes the Service on each Node's IP at a staic port
  - Accessible externally via `NodeIP: NodePort`
  - Good for development and testing
3. **LoadBalancer**: Exposes the Service externally using a cloud provider's load balancer
  - Creates an external IP that routes to the Service
  - Recommended for production AI APIs
4. **Headless**: Special service (ClusterIP:: None) that returns individual Pod IPs
  - Direct Pod-to-Pod communication
  - Useful for stateful workloads
  
#### Services for AI workload
  - **API Exposure**: Expose model servers to user via stable, load-balanced endpoints that persist even as pods come and go
  - **Data Access**: Connect inference pods with databases, vector stores, and feature stores using internal services
  - **Security**: Secure AI services with TLS certificates, authentication, and ingress controllers for traffic management

### Deployments + Services
  - Deployment ensures the right number of pods are always running
  - Service provides a stable endpoint for accessing those pods
  - Together enable production-grade AI inference APIs with high availability

## ConfigMaps, Secrets and Volumes
  - **Configuration Complexity**: ML workloads require numerous hyperparameters, paths, and environments, path and environment setting that change between development and production
  - **Security Requirement**: AI systems often need access to sensitive data, APIs, and credentials that must be securely managed
  - **Data Persistence**: Large datasets, model weights, and training checkpoints must persist beyond container lifecycles

### ConfigMap
Kubernetes objects that store non-sensitive configuration data as key-value pairs:
  1. Decouple configuration from container images
  2. Inject settings as environment variables or files
  3. Update configurations without rebuilding containers
  4. Share common setting across multiple pods
  - For AI workloads, ConfigMaps typically store:
    - Batch sizes and learning rates
    - Model paths and feature toggles
    - Logging levels and monitoring settings
  - Creating and using ConfigMaps for an AI application
    - This pattern enables updating configuration without touching application code or rebuilding containers
  ```
  apiVersion: v1
  kinds: ConfigMap
  metadata: 
    name: ai-config
  data:
    BATCH_SIZE: "64"
    MODEL_PATH: "/models/resnet.pt"
    LOG_LEVEL: "INFO"
    FEATURE_GATE_ADVANCED_METRIC: "true"
  ```
  - Consuming the configMap in a Pod
  ```
  containers:
    - name: inference-service
    image: ai-company/inference:v1.2
    envFrom:
      - configMapRef:
        name: ai-config
  ```

### Secret
1. **Secure Storage**: Secrets are encrpte key-value stores for sensitive information liek passwords, tokens, and keys
2. **Security Features**: Base64 encoded, encrpted at rest (with proper configuration), and only distributed to nodes that need them
  - Base64 encoding is not encrption, Secrets are only encrpted at reset if you've configured etcd encryption
3. **Flexible Consumption**: Mount as environment variables or as files within volumes, keeping credentials out of container images
4. **AI-Specific Use Cases**: API keys for model repositories (Hugging Face, NVIDIA NGC), database credentials, cloud storage access tokens
  - Creating a Secret for an AI application
```
apiVersion: v1
kind: Secret
metadata:
  name: ai-secret
type: Opaque
data:
  DB_PASSWORD: # password base64
  HF_API_KEY: # hf_abc123 base64
  REDIC_AUTH: # redispass base64
```
  - Consuming the Secret in Pod
```
containers:
  - name: training-job
  image: ai-company/training:v2.1
  env:
    - name: HF_API_KEY
    valueFrom:
      secretKeyRef:
        name: ai-secret
        key: HF_API_KEY
```

### Volume
  - **Persistent Storage**: persist beyond pod lifecycle
  - **Storage Types**: Backed by cloud block storage, network filesystems, or specialized AI storage solutions
  - **Access Modes**: ReadWriteOnce(RWO), ReadOnlyMany(ROX), ReadWriteMany(RWX), depending on workload needs
  - Creating a PersistentVolumeClaim for model storage:
```
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-model
spec:
  accessModesL
    - ReadWriteMany
  resources:
    requests:
      storages: 100Gi
  storageClassName: ssd-storage
```
  - Using the volume in an AI pod:
```
apiVersions: v1
kind: Pod
metadata:
  name: ai-pod
spec:
  containers:
    - name: ai-app
    image: pytorch/pytorch:latest
    volumeMounts:
      - mountPath: /models
      name: model-store
  volumes:
    - name: model-store
    persistentVolumeClaim: 
      claimName: pvc-model
```
 
### ConfigMap + Volumne
  - These components enable reproducible, secure, and scalable multi-stage AI workflows on Kubernetes without sacrificing flexibility or introducing security risks
1. **Data Preparation**: Use Volumes to store datasets, ConfigMaps for preprocessing parameters, and Secrets for data source credentials
2. **Model Training**: ConfigMaps define hyperparameters, Volumes store checkpoints, and Secrets provide access to private model repositories
3. **Inference Serving**: Volumes hold optimized models, ConfigMaps control batch sizes and timeouts, while Secrets manage API authentication

### Best Practices (scaling AI workloads in production Kubernetes environments)
#### Security
  - never hardcode credentials in images or ConfigMaps
  - Rotate Secrets regularly, especially in production
  - Use RBAC to limit access to sensitive resources
  - Consider external secret management tools for production
#### Organization
  - Use namespaces to separte development/staging/production on configs
  - Label resources clearly for tracking and auditing
  - Document ConfigMap options for ML engineers
  - Version control your config (not secrets) with code
#### Performance
  - Use SSD-backed storage for model serving
  - Consider ReadWriteMany volumes for shared datasets
  - Be aware of volume mount performance impacts
  - Use cloud-managed storage for every large datasets