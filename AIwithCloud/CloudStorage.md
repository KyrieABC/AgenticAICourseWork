# Cloud Storage for AI
**AI workloads demand enoumous datasets and model weights**

## Object Storage 
**Highly scalable storage for unstructured data**
  - Infinite capacity potential
  - Cost-effective for large datasets
  - API-driven access patterns
  - Consists of:
    1. Data (binary blob)
    2. Metadata (properties and attributes)
    3. Unique identifier (global namespace)
  - *Accessed via APIs rather than filesystem paths, making it ideal for datasets, log and model chekcpoints*
  - AI Training dataset storage
  - AI model weights preservation
  - Data Lake implementation
  - Backup & Archival 

## Block Storage 
**Fast disk volumes for compute instances**
  - Low-latency performance
  - Instance-attacked resources
  - Similar to physical SSDs/HDDs
  - Virtual hard drive attacked to your compute instances
  1. **Low Latency Access**: microsecond response times
  2. **High throughout**: hundreds of MB/s to GB/s
  3. **Instance binding**: Typically attached to specific VMs
  4. **Familiar interface**: Appears as a local disk volume
  - AI dataset preprocessing
  - AI training jobs
  - AI feature stores

## File Storage
**Shared access via filesystem protocols**
  - Multi-server mounting capability
  - Familiar directory structure
  - Team collaboration-friendly
  - Provides shared access via standard filsystem protocols
  1. Simultaneously mountable across multiple compute instances
  2. Hierarchical directory structure familiar to all developers
  3. Granular permission and access controls
  4. Traditional POSIX-compliant interface
  - AI team collaboration (Enable multiple ML engineers to access teh same dataset, code and models simultaneously)
  - AI Multi-Server Model Serving (Hosts pre-training models across multiple inference servers from a single, consistent location)
  - AI interactive development

| Attribute | Object Storage | Block Storage | File Storage |
| :--- | :--- | :--- | :--- |
| **Scalability** | Virtually unlimited | Limited by volume size | High but with limits |
| **Performance** | Moderate throughput, high latency | High throughput, low latency | Variable, network-dependent |
| **Cost** | Very low (¢ per GB) | Moderate (10¢-50¢ per GB) | Higher (20¢-$1+ per GB) |
| **Multi-instance**| Via API only | Single instance only* | Multiple instances |
| **Access Pattern** | API/HTTP requests | Direct disk I/O | Filesystem operations |
| **Ideal for** | Datasets, archives, checkpoints | High-speed training, preprocessing | Shared workspaces, collaboration |