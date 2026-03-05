#!/bin/bash

QUERY="sunny city"

# Clear existing containers and environment. 
# From cosmos-dataset-search/ directory:
rm -rf .venv
docker stop $(docker ps -q)

# Setup environment 
make build-docker
make install
make install-cds-cli
make test-integration-down && make test-integration-up

# Download PAI dataset and ingest into CDS collection
# TODO: update to NIM 1.1 to avoid ziping files
make ingest-pai

# Query the generated CDS collection
cds collections list | jq .collections[0].id | xargs -I {} cds search --collection-ids {} --text-query "${QUERY}" --top-k 5 | jq -r '.retrievals[].metadata.video_id'