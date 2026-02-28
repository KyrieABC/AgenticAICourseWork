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
  - **Deployment Solution**: Deployments manage pod lifecycles, ensuring your AI application maintain the desired state despite failures
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
  # A yaml (.yml) file define kubernetes deployment:
  # Tells Kubernetes which version of the API to use to create this resource (apps/v1 is the standard for Deployments)
  apiVersion: apps/v1
  kind: Deployment
  # Contains data that helps uniquely identify the object
  metadata:
    # The name of this deployment is ai-api
    name: ai-api
  # Defines the desired behavior of the Deployment
  spec: # Desired state
    # Tells Kubernetes to always keep three copies (pods) of this application running. If one dies, Kubernetes starts a new one automatically
    replicas: 3
    # Defines how the Deployment finds which Pods to manage
    selector:
      # The Deployment manages any Pod that has the label app: ai-api
      matchLabels:
        app: ai-api
    # Defines the blueprint for the pods that will be created
    template:
      metadata:
        # Crucial line. This applies the label app: ai-api to the pods. The selector above uses this label to identify them
        labels:
          app: ai-api
      spec:
        # Defines the containers running inside the pod
        containers:
          # Gives the container within the pod the name api
          - name: api
          images: ai-app:latest
          # Tells Kubernetes that the application inside this container is listening for traffic on port 8000
          ports:
            - containerPort: 8000
  ```
  1. Ensure 3 Pods are running
  2. Label them with `app: ai-api`
  3. Run your `ai-app:latest` image inside them
  4. Configure networking so the container inside can receive traffic on port 8000

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
```
# YAML files define kubernetes services:
apiVersion: v1
kind: Service
metadata:
  # The name of this service is ai-service. Other services in your cluster can use this name to find it via DNS
  # DNS: a built-in service discovery mechanism that automatically assigns DNS names to services and pods, allowing them to communicate using human-readable names instead of IP addresses
  name: ai-service
spec:
  selector:
    # Crucial line. This is how the Service finds the Pods to send traffic to. It looks for any pod in the cluster with the label app: ai-api
    app: ai-api
  # Defines how network traffic is handled
  ports:
    - protocol: TCP
    # The Service Port. This is the port that the service itself listens on inside the cluster
    # Other applications in the cluster will talk to ai-service on port 80
    port: 80
    # The Container Port. This is the port the actual application inside the pods is listening on (matching the containerPort in your Deployment)
    # The Service forwards traffic from port 80 to port 8000
    targetPort: 8000
  # Tells Kubernetes to request a physical Load Balancer from your cloud provider (like AWS ELB)
  # This makes your service accessible from the internet via an external IP address
  type: LoadBalancer
```
  1. Creates a stable IP address for `ai-service`
  2. Monitors for pods labeled `app:ai-api`
  3. Load balances incoming traffic from the internet (port 80) to those pods (port 8000)

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
    # File path inside container where the application should look for the pre-trained model
    MODEL_PATH: "/models/resnet.pt"
    # Except for "DEBUG", shows every other information for log
    LOG_LEVEL: "INFO"
    # It tells the application to turn on a specific piece of functionality—likely more detailed monitoring or telemetry data
    FEATURE_GATE_ADVANCED_METRIC: "true"
  ```
  - Consuming the configMap in a Pod
  ```
  containers:
    - name: inference-service
    image: ai-company/inference:v1.2
    # Take everything inside a specific source and make it an environment variable
    envFrom:
      # Specifies that the source of these variables is a ConfigMap
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
  # The unique identifier you'll use when you want to "borrow" these credentials for your app
  name: ai-secret
# This is the default type for Secrets. It simply means it's a "blob" of arbitrary user-defined data (key-value pairs) with no specific format requirements
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
    # Don't look for a plain text value here; go fetch it from another resource
    valueFrom:
      # Specifically points to a Secret as the source
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
  # This asks for 100 Gibibytes of space. Kubernetes will look for a physical disk that has at least this much room (resources.requests.storage: 100Gi)
  resources:
    requests:
      storages: 100Gi
  # This tells Kubernetes what kind of hardware to use. In this case, it's requesting high-performance SSDs rather than slower standard hard drives—essential for loading large models quickly
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
    # This tells the container where to put the disk
    volumeMounts:
      # Inside your Linux container, a folder named /models will appear. Anything you save there is actually being written to that 100Gi SSD we defined earlier.
      - mountPath: /models
      name: model-store
  volumes:
    # This name must match the volumeMounts.name above. It acts as the "glue" between the physical storage and the container folder
    - name: model-store
    # Tells Kubernetes to go look for an existing claim
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

## Horizontal Pod Autoscaling for AI inference
  - **Variable Traffic**: Inference workloads experience unpredictable spikes and idle periods, creating unique scaling challenges
  - **Cost Optimization**: Overprovisioning GPUs and CPUs wastes thousands in cloud spend that could be allocated elsewhere
  - **Performance Impact**: Underprovisioning leads to increased latency, failed requests, and degraded user experience
  - Kubernetes Horizontal Pod Autoscalar (HPA) provides the dynamic scaling mechanism needed to balance these competing concerns

### What is Horizontal Pod Autoscaling (Kubernetes resource)
  - Automaticall adjusts replica count of pods based on observed metrics
  - Watch CPU, memory, or custom metrics to make scaling decisions
  - Maintains workload responsiveness under variable load
  - Runs continuously in the background without manual intervention
  - **Ideal for AI inference APIs where user traffic is unpredictable and resource demands fluctuate**

### How HPA works
1. **Metrics Collection**: Metrics server continuously collects resource usage data from all pods in cluster
2. **Threshold Comparison**: HPA controller compares actual metrics against target thresholds defined in the HPA specification
3. **Scale Up Decision**: If load increases above target, controller calculates new replica count and scales deployment up
4. **Scale Down Decision**: If load drops below target, controller waits for stabilization period, then reduces replicas
  - HPA controller implements a control loop that runs every 15 seconds by default, continuously balancing replicas to match actual demand
```
apiVersion: autoscaling/v2
kind: HorizontalPodAutoScaler
metadata:
  name: ai-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```
  1. Scales to keep CPU utilization at 70%
  2. Use the `autoscaling/v2` API (stable)

### HPA for AI Inference workloads
  - **Handling Inference Spikes**: Automatically scales API pods during peak traffic periods, such as business hours or marketing events, when requests to LLMs or vision models spike
  - **Latency Management**: Maintains consistent low latency by ensuring enough pods are available to process inference requests without queuing
  - **Cost Efficiency**": Reduces expensive GPU/accelerator costs during off-peak hours by scaling down pods when they're not needed

### Custom Metrics for AI
1. **GPU Utilization**: Scale based on actual GPU compute or memory usage
2. **Request Latency**: Scale when p95/p99 inference time exceed thresholds
3. **QPS (Queries Per Second)**: Scale directly based on traffic volume to the service
  - **Tools**:
    1. **Prometheus Adapter**: Exposes custom metrics to Kubernetes API
    2. **KEDA**: supports event-driven scaling
    3. **DCGM Exporter**: Exports NVIDIA GPU metrics

### Autoscaling in Cloud AI Services
  - Cloud-manged AI services implement autoscaling by default, with a scaling-first architecture that's optimized for variable inference loads

### Limitation of HPA
1. **Scaling Delay**: Container startup and model loading create cold-start latency 
2. **Not for Training**: Unsuitable for batch training jobs 
3. **Stateless Only**: Works best for stateless inference services
4. **GPU Complexity**: GPU fractional allocation and bin-packing are more challenging than CPU
5. **Tuning Required**: Requires careful min/max replica configuration based on workload

### Best Practice
1. **Redundancy**: ALways set minReplicas >> 1 to maintain high availability during scaling events or node failures
2. **Readiness Probes**: Implement proper readiness probes to ensure pods only receive traffic after models are fully loaded
3. **Monitoring**: Create dedicated Prometheus + Grafana dashboards to visualize scaling events and resource usage
4. **Multi-level Scaling**: Combine HPA with Cluster Autoscaler to ensure nods are available when pod count increases

## Helm Charts for Simplified AI Deployments
  - Raw Kubernetes YAML manifests are:
    1. Long and repetitive
    2. Difficult to manage and scale
    3. Prone to configuration errors
    - AI applications require multiple manifests for Deployment, Services, PVCs and ingress resources
  - Helm solves these problems as a Kubernetes package managerL
    1. Simplifies installation & updates
    2. Works like apt/pip for Kubernetes
    3. Package related resources together
    4. Enables templating and reuse

### What is Helm
1. **Chart Packaging**: Packages Kubernetes YAML manifests into reusable Charts that can be versioned and shared across teams and environments
2. **Templating**: Supports powerful templating for configuration flexibility and maintainable infrastructure definitions
3. **Versioning & Rollbakcs**: Provides built-in versioning and rollback capabilities for reliable deployments for ML models and infrastructure

### Helm vs Raw YAML
1. YAML:
  - Verbose and repetitive
  - Hard to maintain at scale
  - Environment-specific values embedded
  - No built-in versioning
2. Helm Advantages
  - Parameterized, DRY approach
  - Reuse configs across environments
  - Built-in rollbacks for failed releases
  - Makes AI infrastructure modular

### Anatomy of a Helm Chart
```
my-ai-chart/
  # metadata
  |--- Chart.yaml 
  # default configs
  |--- values.yaml
  # k8s YAML templates, containing templated Kubernetes manifests
  |--- templates/
    |--- deployment.yaml
    |--- service.yaml
    |--- ingress.yaml
```
  - value.yaml files provides single source of configuration that:
    1. Controls deployment scale, image, and service type
    2. Allows engineers to tune parameters without editing YAML
    3. Separates configuration from implementation
    4. Makes GPU/memory allocatoin explicit and stardardized
    5. Simplifies environment-specific configuration
  - Ex of values.yaml:
```
replicaCount: 3
image:
  repository: ai-app
  tag: latest
service:
  type: LoadBalancer
  port: 80
resources:
  limits:
    cpu:"2"
    memory"4Gi"
```
 
### Installing a Chart
1. **Install a custom AI Chart**: `helm install my-ai-app ./my-ai-chart`
  - Deploy your own AI application with one command
2. **Add repository for public charts**: `helm repo add bitnami https://charts.bitnami.com/bitnami`
  - Access libraries of pre-built charts for common ML tools
3. **Install from repository**: `helm install mlflow bitnami/mlflow`
  - Deploy complex ML infrastructure with a single command
    
### Helm for AI Pipelines
1. Common AI Tools Available as Charts:
  - MLFlow for experiment tracking
  - Kubeflow components for ML pipelines
  - Model serving platforms
  - Monitoring tools
2. Benefits for ML Teams
  - Package inference APIs with Helm charts
  - Reuse charts across teams & environments
  - Accelerate infrastructure deployments
  - Standardize ML platform components

### Rollbacks & Updates
1. **Update**: `helm upgrade my-ai-app ./chart`
  - Safely deploy new model versions or configuration changes
2. **Rollback**: `helm rollback my-ai-app 1`
  - Instantly revert to previous version if issues occur
3. **Canary Deploy**
  - Support for canary deployments when testing new ML models in production

### Best Practice 
1. **Version Control**: Keep charts version-controlled in GIt using GitOps principles
2. **Environment Values**: Use separate values.yaml file for dev, staging and production
3. **Chart Repository**: Store enterprise AI charts in private repositories for security and governance
4. **CI/CD Integration**: Combine with CI/CD for automated deployments of ML models
