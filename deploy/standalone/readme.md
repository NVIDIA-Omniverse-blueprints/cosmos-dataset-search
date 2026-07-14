# Visual Search Standalone Deployment

This directory contains the standalone deployment configuration and scripts for the Visual Search service. It is intended for debugging purposes.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [Accessing the Service](#accessing-the-service)
  - [Accessing the Service via react UI](#accessing-the-service-via-react-ui)
  - [Accessing the Service via Command Line Using CURL and REST API](#accessing-the-service-via-command-line-using-curl-and-rest-api)
  - [Accessing the Service via vius client](#accessing-the-service-via-vius-client)
    - [Install vius client](#install-vius-client)
- [Ingesting Data](#ingesting-data)
  - [Local Ingestion](#local-ingestion)
  - [Remote Ingestion from an AWS s3 Bucket](#remote-ingestion-from-an-aws-s3-bucket)
- [Stopping the Service](#stopping-the-service)
- [Additional Information](#additional-information)

## Overview

The standalone deployment package provides everything needed to run the Visual Search service as an independent application, without requiring additional infrastructure dependencies.

## Prerequisites

### Service

#### Hardware

- amd64 system with NVIDIA GPU (Compute capability 7.0+) with 12 GB+ VRAM.

#### Software

- Docker 27.0.1+ (able to run without admin privileges)
- Docker compose (https://docs.docker.com/compose/install/linux/#install-using-the-repository)
- Ubuntu 22.04

### Client

- Ubuntu 20.04 or 22.04
- NGC command line tool for Linux (for installation)
- Python 3.10 with pip3 23.0.1 or older (due to naming convention of the client wheel)
- (optional) Python virtualenv

## Configuration

1. Log in docker with your NGC credentials:
    ```bash
    docker login nvcr.io
    ```

2. Modify the `.env` file to set the environment variables.
    * Make sure you have the correct docker images (change environment variables to match the image you want to use).
    * Change `HOST_IP` variable to the host ip address of the machine where the visual-search service will be running. This must be an address that both host and client (browser) can reach, not localhost or a service name.
    * Change `NV_VIUS_NGC_KEY` to your NGC key.

3. If needed, modify the `docker-compose.yaml` file to mount other files from the host system's local storage in order to debug or develop. See example of mounting on the `visual-search-react-ui` service.

## Running the Service

1. Set the environment variables:
    ```bash
    source .env
    ```

2. Start the service:
    ```bash
    docker compose down && docker compose up
    ```

    The service is ready to use when you see the following message:
    ```bash
    visual-search           | [INFO] Application startup complete.
    ```

> **Note**: The `cosmos-embed` service mounts its temp directory as `- /tmp/ram:size=2g,mode=1777`; without `mode=1777` the NIM (running as non-root) cannot write temp video files and all video embedding requests fail with `Permission denied` / HTTP 500.

## Accessing the Service

You can access the visual search service three ways: via **vius client**, via **react UI**, via command line using **CURL** and REST API.

Note that to modify the collections and ingest data you must use the vius client.

### Accessing the Service via react UI
You can access the service UI using a web browser at `http://<host_ip>:8080/` . For collections containing the result of local ingestions, search results will not display thumbnails. Thumbnails and playback/display are only supported for ingestions made from an AWS s3 bucket.

You can access visual-search API at `http://<host_ip>:8888/v1/docs` .

### Accessing the Service via Command Line Using CURL and REST API

You can use command line CURL commands to send REST API requests to the service. The list of commands can be obtained from the visual-search API page at `http://<host_ip>:8888/v1/docs` . For example, to request the available pipelines, you can use this command:

```bash
curl -X 'GET' \
  'http://<host_ip>:8888/v1/pipelines' \
  -H 'accept: application/json'
```

### Accessing the Service via CDS client

The CDS client is a Python application that allows you to manage the collections and ingest data to the service.

Searching with the CDS client uses the `retrieval` API. The `search` API (as used by the react UI), is currently supported only through CURL commands. See [Accessing the Service via Command Line Using CURL and REST API](#accessing-the-service-via-command-line-using-curl-and-rest-api) for more information.

#### Install CDS client

The CDS client can be installed directly from the repository using the make target.

1. Install the client from the repository root:

```bash
# From the repository root
make install 
or
make install-cds-cli
```

This will install the `cds` command in your virtual environment.

2. Configure the client by creating the file `$HOME/.config/cds/config` with the following contents:

```bash
[default]
api_endpoint = http://<host_ip>:8888
```

Alternatively, you can run the interactive configuration:

```bash
cds config set
```

## Ingesting Data

You can use the CDS client to manage the collections and ingest data into the service.

### Local Ingestion

The following is an example to ingest local video files.

1. Create a new collection:
```bash
cds collections create --pipeline cosmos_video_search_milvus --name my-video-collection --config-yaml /path/to/cvfactory/src/visual_search/client/datasets/simple.yaml
```

2. Optional, obtain the ID of the new collection by listing all collections. Otherwise, the `create` command above should have printed this ID.

```bash
cds collections list
```

3. Ingest data to the collection:
```bash
cds ingest files "/path/to/video/dir/" --collection-id <collection_id> --extensions mp4 --batch-size 4 --num-workers 4
```

4. Optional, check the status of the ingestion:
```bash
cds collections get <collection_id>
```

This will list the collection and how many embeddings were created. It may take a few minutes before you see the collection populated.

5. Query the collection:
```bash
cds search --collection-ids <collection_id> --text-query "people dancing" --top-k 5
```

This should print out all the videos that match the query.

For more information use the CDS client help command:
```bash
cds --help
```

### Remote Ingestion from an AWS s3 Bucket

To ingest data located in an s3 bucket, you need to set up the secrets to allow the service and your vius client to access the s3 bucket.

You must have an s3 profile before ingestion with the AWS Access Key ID, AWS Secret Access key, and region.

#### Configure the client machine

1. Export these environment variables:

```bash
export VS_API_ENDPOINT="http://<host_ip>:8888"
export AWS_ACCESS_KEY_ID="<aws_key_id>"
export AWS_SECRET_ACCESS_KEY="<aws_secret_key>"
export AWS_REGION="<aws_region>"
export S3_STORAGE_LOCATION="s3://<bucket>/<directory>"
export S3_STORAGE_SECRET_NAME="s3-profile"
```

* Change the API end-point to the host machine IP address (same value as specified in your `.env` file).
* Change the `AWS_*` values to match your s3 bucket access configuration.
* Change the storage location to match the location of the s3 bucket.
* Optionally, change the storage secret name from the default value.

2. Create a yaml file to use when creating the collection that will hold the ingested data. The file contents should follow the template below:

```
{
  "tags": {
    "storage-template": "s3://<bucket>/<folder>/{{filename}}",
    "storage-secrets": "<storage secret name>"
  }
}
```

You can use this command to output the file `s3_ingest.yaml` using the environment:

```bash
echo "{
  \"tags\": {
    \"storage-template\": \"${S3_STORAGE_LOCATION}/{{filename}}\",
    \"storage-secrets\": \"${S3_STORAGE_SECRET_NAME}\"
  }
}" > /path/for/s3_ingest.yaml
```

Make sure that the storage template follows the pattern of your s3 bucket or URLs generated will not be valid.

3. Configure the AWS s3 profile:

```bash
aws configure set aws_access_key_id "${AWS_ACCESS_KEY_ID}" --profile "${S3_STORAGE_SECRET_NAME}" && aws configure set aws_secret_access_key "${AWS_SECRET_ACCESS_KEY}" --profile "${S3_STORAGE_SECRET_NAME}" && aws configure set region "${AWS_REGION}" --profile "${S3_STORAGE_SECRET_NAME}"
```

The profile name in `S3_STORAGE_SECRET_NAME` will be used throughout with the vius client to access the s3 secrets.

#### Transmit Secret to Service

Set the storage secret into the service using the command below:

```bash
curl -X 'POST' \
  "${VS_API_ENDPOINT}/v1/secrets/set" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
  \"name\": \"${S3_STORAGE_SECRET_NAME}\",
  \"data\": {
    \"aws_access_key_id\": \"${AWS_ACCESS_KEY_ID}\",
    \"aws_secret_access_key\": \"${AWS_SECRET_ACCESS_KEY}\",
    \"aws_region\": \"${AWS_REGION}\"
  }
}"
```

The below message is returned upon successfully setting the secret in the service.

```
{"message":"Successfully set secrets `s3-profile`"}
```

#### Ingest from the s3 Bucket

1. Create the collection using the yaml file created earlier:

```bash
cds collections create --pipeline cosmos_video_search_milvus --name my-video-collection-s3 --config-yaml /path/for/s3_ingest.yaml
```

2. Optional, obtain the ID of the new collection by listing all collections. Otherwise, the `create` command above should have printed this ID.

```bash
cds collections list
```

3. Ingest data to the collection:
```bash
cds ingest files "${S3_STORAGE_LOCATION}" --collection-id <collection_id> --s3-profile "${S3_STORAGE_SECRET_NAME}" --extensions mp4 --batch-size 4 --num-workers 4
```

4. Optional, check the status of the ingestion:
```bash
cds collections get <collection_id>
```

This will list the collection and how many embeddings were created. It may take a few minutes before you see the collection populated, but queries should work right after ingestion is completed.

5. Query the collection:
```bash
cds search --collection-ids <collection_id> --text-query "people dancing" --top-k 5
```

6. Check that you are able to query for the asset URL when searching (change `<collection_id>` to the ID of the collection to search):

```bash
curl -X 'POST' \
  "${VS_API_ENDPOINT}/v1/collections/<collection_id>/search" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": {
    "text": "people dancing"
  },
  "top_k": 5,
  "reconstruct": false,
  "search_params": {},
  "filters": {},
  "generate_asset_url": true,
  "clf": {}
}'
```

Check that the `"asset_url"` field in the results is not `null`, points to the s3 bucket, and that the URL returned is valid.

## Stopping the Service

To stop the service, press CTRL + C on the terminal where the service is running.

If service is running detached, then run this command instead:

```bash
docker compose down
```

## Additional Information

To remove the persistent volumes for Postgres and Milvus, and start from scratch, run the following commands:

```bash
docker compose down
docker volume ls # lists all the persistent volumes to be used by docker compose
docker volume rm visual_search_postgres_data # check this is the same name shown from previous command
docker volume rm visual_search_milvus_data # check this is the same name shown from previous command
```
This may be necessary if the database format changes or gets corrupted when debugging or developing.
