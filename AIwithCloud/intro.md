# CLoud Infrastructure Basics
  - **On-Demand**: Access power compute, storage, networkig infrastructure

## AI workload in cloud
  - **Training** : Distribute GPU clusters
  - **Inference**: Scalable services for real-time prediction with optimized throughtout

## COmpute Optoins for AI in cloud
  - **VMs with CPU/GPU** : Standard instances with attached GPUs for general ML workloads
  - **Specialized Accelerators**: TPUs and other AI-optimized hardware for max performace
  - **Serverless Options**: Functions-as-a-Service for lightweight, on-demand inference with auto-scaling
  - *Utilize spot/preemptible instances for non-critical training jobs to reduce expenses by 70-90%*

## Storage Solutions for AI workload
  - **Object storage**: ideal for datasets, checkpoints and model artifacts
  - **Block Storage**: High-IOPS SSDs for data-intensive training workloads requiring low latency
  - **File Storage**: NFS-compatible systems for shared access across compute cluster
  - **Databases & Lakes**: Structured storage for features, metadata, and experiment tracking

## Networking for Cloud AI infra
  - **Virtual Private Clouds**: Isolated network environments with configurable security policies to protect sensitive data and workloads
  - **Load Balancing**: Distribute inference requests across multiple model servers to maintain low latency under varying loads
  - **High-Bandwidth Interconnects**: enabling efficient multi-GPU and multi-node training
  - **Edge Network**: Globally distributed points of presence for deploying AI services closer to users

### Advantage of Cloud AI
1. Deployment Speed
2. Geographic reach
3. 100% Resource Utilization
### Challenges of Cloud AI
1. Cost Management (large-scale training)
2. Vendor Lock-in
3. Network costs
4. Skills gap (require specialized knowledge of cloud infra)

## Cost Optimization Strategies
  - **Spot Instances**: User preemptible VMs for fault-tolerant training jobs to save 70-90% on compute costs
  - **Storage Tiering**: Move infrequently accessed datasets to cold storage, reducing storage cost
  - **Usage Monitoring**: Implement dashboards and alerts to identify idle resources and unnecessary spending
  - **Autoscaling**: Configure inference endpoints to scale baed on actual demand rather than peak capacity
  - **Workload Balancing**: Train preliminary models locally, using cloud only for large-scale training and deployment

## Cloud & Edge AI integration
### Edge Deployment Benefits
1. Ultra-low latency for real-time application
2. Works in disconnected environments
3. Reduces bandwidth costs for high-volumn inference
4. Preserves data privacy by processing locally
### Key applications
  - Autonomous vehicle
  - IoT sensor networks
  - Augmented reality (AR)
  - Retial Computer visoin
  - industrial quality control