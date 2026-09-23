# Kubernetes Troubleshooting Guide

This guide covers common issues and solutions when deploying CDS on Kubernetes (Tested for AWS EKS).

## General Troubleshooting Steps

### Checking Pod Status

```bash
# Check all pods
docker exec cds-deployment bash -c "kubectl get pods"

# Check pods in specific namespace
docker exec cds-deployment bash -c "kubectl get pods -n default"

# Watch pods in real-time
docker exec cds-deployment bash -c "kubectl get pods -w"
```

### Viewing Logs

```bash
# View logs from a specific pod
docker exec cds-deployment bash -c "kubectl logs <pod-name> --tail=100"

# Follow logs in real-time
docker exec cds-deployment bash -c "kubectl logs -f <pod-name>"

# View logs from all pods with a label
docker exec cds-deployment bash -c "kubectl logs -l app=visual-search --tail=50"
```

### Describing Resources

```bash
# Get detailed pod information
docker exec cds-deployment bash -c "kubectl describe pod <pod-name>"

# Describe a deployment
docker exec cds-deployment bash -c "kubectl describe deployment <deployment-name>"

# Check events (shows recent cluster events)
docker exec cds-deployment bash -c "kubectl get events --sort-by='.lastTimestamp'"
```

### Accessing Pod Shell

```bash
# Open interactive shell in a pod
docker exec -it cds-deployment bash -c "kubectl exec -it <pod-name> -- /bin/bash"
```

## Common Issues

### Pods Won't Start

#### ImagePullBackOff

**Symptom**: Pod stuck in `ImagePullBackOff` or `ErrImagePull` status

**Check**:
```bash
docker exec cds-deployment bash -c "kubectl describe pod <pod-name> | grep -A 10 Events"
```

**Common Causes**:
- NGC API key is incorrect or not set
- Image doesn't exist or access denied
- Network issues preventing image download

**Solution**:
```bash
# Verify NGC secrets exist
docker exec cds-deployment bash -c "kubectl get secret nvcr-io ngc-api ngc-secret"

# Check secret contents (base64 encoded)
docker exec cds-deployment bash -c "kubectl describe secret ngc-api"

# If secrets are wrong, recreate them
# Inside container:
cd /workspace/blueprint/bringup
./secrets.sh

# Restart the problematic pod
docker exec cds-deployment bash -c "kubectl delete pod <pod-name>"
```

#### CrashLoopBackOff

**Symptom**: Pod keeps restarting

**Check logs**:
```bash
docker exec cds-deployment bash -c "kubectl logs <pod-name> --previous"
```

**Common Causes**:
- Application configuration error
- Missing environment variables
- Out of memory
- Dependency service not available

**Solution**: Check logs for specific error messages and fix the root cause.

#### Pending Pods

**Symptom**: Pod stuck in `Pending` status

**Check**:
```bash
docker exec cds-deployment bash -c "kubectl describe pod <pod-name>"
```

**Common Causes**:
- Insufficient resources (CPU/memory/GPU)
- Volume cannot be mounted
- Node selector doesn't match any nodes
- Taints preventing scheduling

**Solutions**:
```bash
# Check node resources
docker exec cds-deployment bash -c "kubectl describe nodes | grep -A 10 'Allocated resources'"

# Check if nodes match selectors
docker exec cds-deployment bash -c "kubectl get nodes --show-labels | grep role=cvs-gpu"

# Check PVC status
docker exec cds-deployment bash -c "kubectl get pvc"
```

#### Insufficient Resources

**Symptom**: Pod shows insufficient CPU, memory, or GPU

**Solution**:
```bash
# Check node capacity
docker exec cds-deployment bash -c "kubectl describe node <node-name>"

# For GPU nodes specifically
docker exec cds-deployment bash -c "kubectl describe nodes -l role=cvs-gpu | grep -A 15 'Allocated resources'"

# Scale up node group if needed
docker exec cds-deployment bash -c "eksctl scale nodegroup --cluster=\$CLUSTER_NAME --name=cvs-gpu --nodes=3"
```

### GPU Issues

#### GPU Not Available to Pods

**Check if GPU nodes exist**:
```bash
docker exec cds-deployment bash -c "kubectl get nodes -l role=cvs-gpu"
```

**Check if NVIDIA device plugin is running**:
```bash
docker exec cds-deployment bash -c "kubectl get pods -n kube-system | grep nvidia"
```

**Solution**:
```bash
# Check NVIDIA device plugin logs
docker exec cds-deployment bash -c "kubectl logs -n kube-system -l name=nvidia-device-plugin-ds"

# Restart device plugin if needed
docker exec cds-deployment bash -c "kubectl delete pod -n kube-system -l name=nvidia-device-plugin-ds"
```

#### NVIDIA Device Plugin Issues

**Verify plugin is running on GPU nodes**:
```bash
docker exec cds-deployment bash -c "kubectl get pods -n kube-system -o wide | grep nvidia"
```

**Check GPU availability**:
```bash
# Exec into GPU node and check nvidia-smi
docker exec cds-deployment bash -c "kubectl debug node/<gpu-node-name> -it --image=nvidia/cuda:11.8.0-base-ubuntu22.04 -- nvidia-smi"
```

### Networking Issues

#### Service Not Reachable

**Check service exists**:
```bash
docker exec cds-deployment bash -c "kubectl get svc"
```

**Check endpoints**:
```bash
docker exec cds-deployment bash -c "kubectl get endpoints <service-name>"
```

**Solution**:
```bash
# Test service connectivity from another pod
docker exec cds-deployment bash -c "kubectl run -it --rm debug --image=busybox --restart=Never -- wget -O- http://<service-name>:<port>"
```

#### DNS Resolution Failures

**Test DNS from within cluster**:
```bash
docker exec cds-deployment bash -c "kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup <service-name>"
```

**Check CoreDNS pods**:
```bash
docker exec cds-deployment bash -c "kubectl get pods -n kube-system -l k8s-app=kube-dns"
```

#### Ingress Not Working

**Check ingress status**:
```bash
docker exec cds-deployment bash -c "kubectl get ingress simple-ingress"
```

**Check ingress controller**:
```bash
docker exec cds-deployment bash -c "kubectl get pods -n ingress-nginx"
```

**Check load balancer**:
```bash
docker exec cds-deployment bash -c "kubectl get svc -n ingress-nginx"
```

**Common Issues**:
- ALB not created: Check ingress controller logs
- 404 errors: Verify ingress paths match service endpoints
- Connection refused: Wait 2-3 minutes for ALB to initialize

#### LoadBalancer Pending

**Check service type**:
```bash
docker exec cds-deployment bash -c "kubectl get svc -n ingress-nginx"
```

**AWS Specific**: Verify cloud controller is working:
```bash
docker exec cds-deployment bash -c "kubectl logs -n kube-system -l app=aws-cloud-controller-manager"
```

### Storage Issues

#### PVC Pending

**Check PVC status**:
```bash
docker exec cds-deployment bash -c "kubectl get pvc"
docker exec cds-deployment bash -c "kubectl describe pvc <pvc-name>"
```

**Common Causes**:
- Storage class doesn't exist
- No available volumes
- Zone mismatch

**Solution**:
```bash
# Check storage classes
docker exec cds-deployment bash -c "kubectl get storageclass"

# For AWS EKS, verify EBS CSI driver is running
docker exec cds-deployment bash -c "kubectl get pods -n kube-system | grep ebs-csi"
```

#### Volume Mount Failures

**Check pod events**:
```bash
docker exec cds-deployment bash -c "kubectl describe pod <pod-name> | grep -A 20 Events"
```

**Common Causes**:
- PVC not bound
- Volume in use by another pod (RWO volumes)
- Permission issues

**Solution**: Delete pod and let it recreate, or delete PVC and recreate volume.

### Model Loading Issues

#### Models Not Found (Cosmos-embed)

**Symptom**: Cosmos-embed pod stays in ContainerCreating for long time

**Check**:
```bash
docker exec cds-deployment bash -c "kubectl logs -l app.kubernetes.io/name=cosmos-embed --tail=50"
```

**This is usually normal**: Cosmos-embed downloads ~20GB model on first start.

**Expected logs**:
```
INFO Downloaded filename: Cosmos-Embed1/model-00001-of-00005.safetensors
INFO Downloaded filename: Cosmos-Embed1/model-00002-of-00005.safetensors
```

**Duration**: 10-15 minutes on first download

#### Out of Memory When Loading Models

**Check pod resource limits**:
```bash
docker exec cds-deployment bash -c "kubectl describe pod -l app.kubernetes.io/name=cosmos-embed | grep -A 10 Limits"
```

**Check node memory**:
```bash
docker exec cds-deployment bash -c "kubectl top nodes"
```

**Solution**: Ensure cosmos-embed is scheduled on a node with sufficient GPU memory (16GB minimum).

### Database Issues

#### Milvus Connection Errors

**Check Milvus pods**:
```bash
docker exec cds-deployment bash -c "kubectl get pods | grep milvus"
```

**Check Milvus proxy logs**:
```bash
docker exec cds-deployment bash -c "kubectl logs deployment/milvus-proxy --tail=100"
```

**Test Milvus connectivity**:
```bash
docker exec cds-deployment bash -c "kubectl run -it --rm debug --image=busybox --restart=Never -- nc -zv milvus.default.svc.cluster.local 19530"
```

#### etcd Issues

**Check etcd pod**:
```bash
docker exec cds-deployment bash -c "kubectl get pods | grep etcd"
docker exec cds-deployment bash -c "kubectl logs milvus-etcd-0 --tail=100"
```

**Common Issues**:
- Disk full: Check PVC size
- Memory issues: Check resource limits

#### Data Persistence Issues

**Check PVCs for Milvus**:
```bash
docker exec cds-deployment bash -c "kubectl get pvc | grep milvus"
```

**Check if data is actually in S3**:
```bash
aws s3 ls s3://$S3_BUCKET_NAME/ --recursive | head -20
```

### Performance Issues

#### Slow Query Response

**Check Milvus query node**:
```bash
docker exec cds-deployment bash -c "kubectl logs deployment/milvus-querynode --tail=100"
docker exec cds-deployment bash -c "kubectl top pod -l component=querynode"
```

**Solution**:
- Ensure query node is on high-memory instance (r7i.4xlarge)
- Check if GPU CAGRA index is being used
- Verify mmap is enabled for large collections

#### High Memory Usage

**Check pod memory**:
```bash
docker exec cds-deployment bash -c "kubectl top pods"
```

**Check node memory**:
```bash
docker exec cds-deployment bash -c "kubectl top nodes"
```

**Solution**: Scale horizontally or increase node size if consistently hitting limits.

#### GPU Underutilization

**Check GPU usage**:
```bash
# Get GPU node name
GPU_NODE=$(docker exec cds-deployment bash -c "kubectl get nodes -l role=cvs-gpu -o jsonpath='{.items[0].metadata.name}'")

# Check GPU utilization
docker exec cds-deployment bash -c "kubectl debug node/$GPU_NODE -it --image=nvidia/cuda:11.8.0-base-ubuntu22.04 -- nvidia-smi"
```

## Service-Specific Issues

### Visual Search Service

**Check status**:
```bash
docker exec cds-deployment bash -c "kubectl get pods -l app=visual-search"
docker exec cds-deployment bash -c "kubectl logs deployment/visual-search --tail=100"
```

**Common Issues**:
- Can't connect to Milvus: Check Milvus is running
- Can't connect to Cosmos-embed: Check cosmos-embed pod is ready
- 500 errors: Check logs for specific error messages

### CE1 OSS service

**Check pod status**:
```bash
docker exec cds-deployment bash -c "kubectl get pods -l app.kubernetes.io/name=cosmos-embed"
```

**Debug script** available:
```bash
docker exec cds-deployment bash -c "cd /workspace/blueprint/bringup && ./debug_cosmos_embed.sh"
```

**Common Issues**:
- ImagePullBackOff: Verify NGC_API_KEY
- Pending: Check GPU nodes and resources
- Long startup time: Normal - downloading model

### Milvus Vector Database

**Check all Milvus components**:
```bash
docker exec cds-deployment bash -c "kubectl get pods | grep milvus"
```

**Expected pods** (15 total):
- milvus-proxy
- milvus-rootcoord, datacoord, querycoord, indexcoord
- milvus-querynode, datanode, indexnode
- milvus-etcd-0
- milvus-kafka-0, kafka-1, kafka-2
- milvus-zookeeper-0

**Check specific component**:
```bash
docker exec cds-deployment bash -c "kubectl logs deployment/milvus-<component> --tail=100"
```

### Object Storage (S3)

**Verify S3 bucket exists**:
```bash
aws s3 ls s3://$S3_BUCKET_NAME/
```

**Check service account has S3 permissions**:
```bash
docker exec cds-deployment bash -c "kubectl describe serviceaccount s3-access-sa"
```

Should show IAM role ARN annotation.

**Test S3 access from pod**:
```bash
docker exec cds-deployment bash -c "kubectl run -it --rm s3-test --image=amazon/aws-cli --serviceaccount=s3-access-sa --restart=Never -- s3 ls s3://$S3_BUCKET_NAME/"
```

## Debugging Techniques

### Enable Debug Logging

**For Visual Search**:
```bash
# Set log level via environment variable
docker exec cds-deployment bash -c "kubectl set env deployment/visual-search LOG_LEVEL=DEBUG"
```

### Port Forwarding for Local Access

```bash
# Forward Visual Search API
docker exec -it cds-deployment bash -c "kubectl port-forward svc/visual-search 8888:8888"

# Forward Milvus
docker exec -it cds-deployment bash -c "kubectl port-forward svc/milvus 19530:19530"

# Forward Cosmos-embed
docker exec -it cds-deployment bash -c "kubectl port-forward svc/cosmos-embed 8000:8000"
```

### Resource Monitoring

**Check resource usage**:
```bash
# Pod resources
docker exec cds-deployment bash -c "kubectl top pods"

# Node resources
docker exec cds-deployment bash -c "kubectl top nodes"

# Detailed node info
docker exec cds-deployment bash -c "kubectl describe nodes | grep -A 10 'Allocated resources'"
```

## AWS EKS-Specific Issues

### EKS Node Group Issues

**Check node group status**:
```bash
docker exec cds-deployment bash -c "eksctl get nodegroup --cluster=\$CLUSTER_NAME"
```

**Scale node group**:
```bash
docker exec cds-deployment bash -c "eksctl scale nodegroup --cluster=\$CLUSTER_NAME --name=cvs-gpu --nodes=2"
```

### IAM Role Problems

**Check service account IAM role**:
```bash
docker exec cds-deployment bash -c "kubectl describe serviceaccount s3-access-sa"
```

Should have annotation: `eks.amazonaws.com/role-arn`

**List IAM roles**:
```bash
docker exec cds-deployment bash -c "aws iam list-roles | grep \$CLUSTER_NAME"
```

### EBS CSI Driver Issues

**Check EBS CSI driver status**:
```bash
docker exec cds-deployment bash -c "eksctl get addon --cluster=\$CLUSTER_NAME --name=aws-ebs-csi-driver"
```

**Check driver pods**:
```bash
docker exec cds-deployment bash -c "kubectl get pods -n kube-system | grep ebs-csi"
```

## Data Issues

### Cannot Ingest Data

**Check**:
1. Collection exists: `cds collections list`
2. S3 bucket accessible: `aws s3 ls s3://$S3_BUCKET_NAME/`
3. Videos in S3: `aws s3 ls s3://$S3_BUCKET_NAME/videos/`
4. AWS profile configured (if using --s3-profile)

**Test ingestion with limit**:
```bash
cds ingest files s3://bucket/videos/ \
  --collection-id <id> \
  --extensions mp4 \
  --limit 1 \
  --s3-profile cds-s3-aws
```

### Search Returns No Results

**Check collection has documents**:
```bash
cds collections get <collection-id>
```

Look for `total_documents_count` in output.

**Check Cosmos-embed is working**:
```bash
docker exec cds-deployment bash -c "kubectl logs -l app.kubernetes.io/name=cosmos-embed --tail=50"
```

**Test Cosmos-embed endpoint**:
```bash
# Port forward and test
docker exec -it cds-deployment bash -c "kubectl port-forward svc/cosmos-embed 8000:8000" &
curl -k http://localhost:8000/v1/health/ready
```

## Recovery Procedures

### Restart Specific Pods

```bash
# Restart visual search
docker exec cds-deployment bash -c "kubectl rollout restart deployment/visual-search"

# Restart cosmos-embed
docker exec cds-deployment bash -c "kubectl rollout restart deployment/cosmos-embed"

# Restart Milvus component
docker exec cds-deployment bash -c "kubectl rollout restart deployment/milvus-proxy"
```

### Scale Down and Up

```bash
# Scale down deployment
docker exec cds-deployment bash -c "kubectl scale deployment/visual-search --replicas=0"

# Scale back up
docker exec cds-deployment bash -c "kubectl scale deployment/visual-search --replicas=1"
```

### Delete and Recreate Service

```bash
# Uninstall Helm release
docker exec cds-deployment bash -c "helm uninstall visual-search"

# Reinstall
docker exec cds-deployment bash -c "cd /workspace/blueprint/bringup && helm upgrade --install visual-search visual-search --values values.yaml"
```

## Getting Help

### Collecting Diagnostic Information

When reporting issues, collect:

```bash
# Pod status
docker exec cds-deployment bash -c "kubectl get pods -o wide"

# Recent events
docker exec cds-deployment bash -c "kubectl get events --sort-by='.lastTimestamp' | tail -50"

# Service logs
docker exec cds-deployment bash -c "kubectl logs deployment/visual-search --tail=200"
docker exec cds-deployment bash -c "kubectl logs -l app.kubernetes.io/name=cosmos-embed --tail=200"

# Node status
docker exec cds-deployment bash -c "kubectl get nodes"
docker exec cds-deployment bash -c "kubectl describe nodes | grep -A 10 'Allocated resources'"

# Helm releases
docker exec cds-deployment bash -c "helm list"
```

### Reporting Issues

Include:
- Pod status (`kubectl get pods`)
- Relevant logs (`kubectl logs`)
- Pod descriptions (`kubectl describe pod`)
- Cluster info (EKS version, node types)

## Additional Resources

- [AWS EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Milvus Troubleshooting](https://milvus.io/docs/troubleshooting.md)
- [CE1 OSS model files](https://huggingface.co/nvidia/Cosmos-Embed1-224p)
- [AWS EKS Deployment Guide](aws-eks-deployment.md)
- [CLI User Guide](cli-user-guide.md)
