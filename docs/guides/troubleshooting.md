# Troubleshooting

Here are some commonly occurring issues and ways to resolve them.

## General Issues

1. **Problem**:

   a. **Error signatures**: "The maximum number of addresses has been
      reached"

   b. **Cause**: The quota for Elastic IP Addresses has been
      reached.

   c. **Resolution**: Either delete old Elastic IP Addresses or increase the quota.
      increase the quota.

2. **Problem**:

   a. **Error signatures**:

      - "The maximum number of internet gateways has been reached"

      - "The maximum number of VPCs has been reached"

   b. **Cause**: The quota for Virtual Private Clusters has been
      reached.

   c. **Resolution**: Either delete old Elastic IP Addresses or
      increase the quota.

3. **Problem**:

   a. **Error signatures**:

      - "An error occurred (ExpiredTokenException) when calling the DescribeNodegroup operation:
        The security token included in the request is expired"

      - "WARN: failed to get session token, falling back to IMDSv1:
        404 Not Found: Not Found"

      - "ERROR session: fetching region failed:
        NoCredentialProviders: no valid providers in chain.
        Deprecated."

   b. **Cause**: The AWS access key has expired.

   c. **Resolution**: Set the new keys using the ENV variables:
      `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`.

4. **Problem**:

   a. **Error signatures**: "socket.gaierror: \[Errno -2\] Name or service
      not known" from command - "vius pipelines list".

   b. **Cause**: Pods are still initializing or failing.

   c. **Resolution**: Wait until the pods are initialized and try the
      command again. Contact the NVIDIA team if the issue
      persists. Use `kubectl get pods` to check the health of
      the pods.

5. **Problem**:

   a. **Error signatures**: "Max retries exceeded with url: /v1/pipelines"
      from command - "vius pipelines list"

   b. **Cause**: Pods are still initializing or failing.

   c. **Resolution**: Wait until the pods are initialized and try the
      command again. Contact the NVIDIA team if the issue
      persists. Use `kubectl get pods` to check the health of
      the pods.

6. **Problem**:

   a. **Error signatures**: "Got exception 502 Server Error: Bad Gateway
      for url: http://\<\>/api/v1/pipelines" from command - "vius
      pipelines list"

   b. **Cause**: The visual search API endpoint is not ready.

   c. **Resolution**: Wait for some time and try the command again.
      Contact the NVIDIA team if the issue persists.

7. **Problem**:

   a. **Error signatures**: API error:
      "Failed to search in collection` <Collection ID>`. Error
      details: "Something went wrong with the request: \<ClientError:
      An error occurred (AccessDenied) when calling the
      AssumeRoleWithWebIdentity operation: Not authorized to perform
      sts:AssumeRoleWithWebIdentity\>".

   b. **Cause**: The service is not able to access the assets (e.g.
      images/videos from the S3 bucket). The AWS configure step for the S3
      bucket does not work properly with the installation scripts at
      the moment.

   c. **Resolution**:

      1. Delete the AWS config file.

      2. Re-run `./enable_s3.sh`. This will reset the policies for the
         bucket.

      3. Uninstall visual search service using `helm uninstall visual-search`.

      4. Install the visual search service using `helm install visual-search visual-search --values=values.yaml`.

      5. Retry the API request and verify that it returns the assets.

8. **Problem**:

   a. **Error signature**: S3 bucket creation fails with the
      "IllegalLocationConstraintException" message.

   b. **Cause**: The region is not one of the valid regions,
      as mentioned in the [AWS S3 API guide](https://docs.aws.amazon.com/AmazonS3/latest/API/API_CreateBucketConfiguration.html).

   c. **Resolution**: Use one of the valid regions to create the S3 bucket
      or modify the script to comply with the AWS S3 API requirements.

9. **Problem**:

   a. **Error signature**: Problem installing `vius` pip client outside of installation
      docker container.

   b. **Resolution**: The `vius` pip client requires `pip<=25`.

## CE1 OSS Service Issues

10. **Problem**:

    a. **Error signatures**: 
       - "CE1 model manifest is incomplete"
       - "Model path is not readable"
       - "Model files are missing"

    b. **Cause**: The offline CE1 model snapshot is missing, incomplete, or inaccessible to the service.

    c. **Resolution**: 
       - Verify the model PVC is bound and mounted at `/models/cosmos-embed1`.
       - Validate the model files against `cosmos-embed1-224p.sha256`.
       - Confirm the files are readable by container UID 999.
       - Review `kubectl logs deployment/cosmos-embed` for the exact validation error.

11. **Problem**:

    a. **Error signatures**:
       - "CUDA out of memory"
       - "Pod killed due to memory limit"
       - "cosmos-embed pod in CrashLoopBackOff"

    b. **Cause**: Insufficient GPU memory for CE1 OSS service model.

    c. **Resolution**:
       - Increase GPU memory limits in `cosmos-embed/values.yaml`.
       - Reduce batch sizes in pipeline configuration.
       - Use GPU with more memory (A100/H100 recommended).
       - Enable model quantization options if available.

12. **Problem**:

    a. **Error signatures**:
       - "Model cache not found"
       - "Persistent volume claim failed"
       - "Storage class not found: high-perf-gp3"

    b. **Cause**: Storage configuration issues with the CE1 OSS service model cache

    c. **Resolution**:
       - Verify the `high-perf-gp3` storage class is properly configured.
       - Check PVC creation and binding status with the `kubectl get pvc` command.
       - Ensure a sufficient storage quota (50GB+) for the model cache.
       - Verify the EBS CSI driver is properly installed.

13. **Problem**:

    a. **Error signatures**:
       - "cosmos-embed service unavailable"
       - "Connection refused to cosmos-embed:8000"
       - "Timeout waiting for cosmos-embed to be ready"

    b. **Cause**: The CE1 OSS service has not properly started or health checks are failing.

    c. **Resolution**:
       - Check the pod status: `kubectl get pods -l app.kubernetes.io/name=cosmos-embed`
       - Review the pod logs: `kubectl logs -f <cosmos-embed-pod-name>`
       - Verify GPU node scheduling and tolerances.
       - Check startup probe timeout settings (30+ minutes may be needed for the first boot).
       - Use the debug script: `./debug_cosmos_embed.sh`

## REST API Troubleshooting

### API Not Responding

Check if the Visual Search API is accessible:

```bash
curl http://localhost:8888/health
```

If this fails, verify services are running:

```bash
docker compose -f deploy/standalone/docker-compose.build.yml ps
make test-integration-logs
```

### Invalid Request Errors (400)

**Problem**: API returns 400 Bad Request

**Resolution**:
- Verify JSON syntax is correct
- Check the request matches the expected schema in the interactive API docs
- Use Swagger UI at `http://localhost:8888/v1/docs` to see required fields and test requests
- Ensure all required fields are provided

### Collection Not Found (404)

**Problem**: API returns 404 for collection operations

**Resolution**:
- Verify the collection ID is correct (check for typos)
- List all collections to see available IDs: `curl http://localhost:8888/v1/collections`
- Ensure the collection wasn't deleted

### Video Ingestion Fails

**Problem**: Video ingestion returns errors or fails silently

**Resolution**:
- For LocalStack: Verify videos exist in bucket:
  ```bash
  # Using boto3
  python -c "import boto3; s3=boto3.client('s3',endpoint_url='http://localhost:4566',aws_access_key_id='test',aws_secret_access_key='test'); print(s3.list_objects_v2(Bucket='cosmos-test-bucket',Prefix='videos/'))"
  ```
- Check Visual Search service logs:
  ```bash
  docker compose -f deploy/standalone/docker-compose.build.yml logs visual-search --tail=50
  ```
- Verify CE1 OSS service is running and ready:
  ```bash
  curl http://localhost:9000/v1/health/ready
  ```
- Check presigned URLs are accessible (should return 200):
  ```bash
  # Generate a test presigned URL and verify
  python -c "import boto3,requests; s3=boto3.client('s3',endpoint_url='http://localhost:4566',aws_access_key_id='test',aws_secret_access_key='test'); url=s3.generate_presigned_url('get_object',Params={'Bucket':'cosmos-test-bucket','Key':'videos/video70.mp4'},ExpiresIn=3600); print(f'URL: {url}'); r=requests.head(url); print(f'Status: {r.status_code}')"
  ```
