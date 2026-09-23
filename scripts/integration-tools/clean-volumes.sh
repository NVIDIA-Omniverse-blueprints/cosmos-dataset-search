#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/deploy/standalone/docker-compose.build.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}CVDS Volume Cleanup Script${NC}"
echo "=================================="

# Check if docker-compose file exists
if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo -e "${RED}ERROR: Docker compose file not found at $COMPOSE_FILE${NC}"
    exit 1
fi

# Function to check if services are running
check_services_running() {
    cd "$PROJECT_ROOT/deploy/standalone"
    if docker compose -f docker-compose.build.yml ps --services --filter "status=running" | grep -q .; then
        return 0
    else
        return 1
    fi
}

# Function to stop services
stop_services() {
    echo -e "${YELLOW}Stopping all CVDS services...${NC}"
    cd "$PROJECT_ROOT/deploy/standalone"
    docker compose -f docker-compose.build.yml down
    echo -e "${GREEN}Services stopped${NC}"
}

# Function to clean volumes
clean_volumes() {
    echo -e "${YELLOW}Removing persistent data volumes...${NC}"
    
    # List volumes that will be removed
    echo "The following volumes will be removed:"
    echo "  - standalone_milvus_data (Milvus vector database)"
    echo "  - standalone_etcd_data (etcd metadata store)"
    
    read -p "Are you sure you want to continue? This will delete all collections and indexed data. (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Operation cancelled.${NC}"
        exit 0
    fi
    
    # Remove the volumes
    docker volume rm standalone_milvus_data standalone_etcd_data 2>/dev/null || true
    echo -e "${GREEN}Volumes removed${NC}"
}

# Function to start services
start_services() {
    echo -e "${YELLOW}Starting CVDS services with fresh volumes...${NC}"
    cd "$PROJECT_ROOT/deploy/standalone"
    docker compose -f docker-compose.build.yml up -d
    echo -e "${GREEN}Services started${NC}"
}

# Function to wait for services
wait_for_services() {
    echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
    
    # Wait for backend health check
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s --max-time 5 http://localhost:8888/health >/dev/null 2>&1; then
            echo -e "${GREEN}Backend service is healthy${NC}"
            break
        fi
        
        echo "Attempt $attempt/$max_attempts - waiting for backend..."
        sleep 10
        ((attempt++))
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        echo -e "${RED}WARNING: Backend service did not become healthy within expected time${NC}"
        echo "You may need to check the logs: docker compose -f deploy/standalone/docker-compose.build.yml logs visual-search"
    fi
}

# Function to verify cleanup
verify_cleanup() {
    echo -e "${YELLOW}Verifying cleanup...${NC}"
    
    # Check collections endpoint
    if collections_response=$(curl -s --max-time 10 http://localhost:8888/v1/collections 2>/dev/null); then
        if echo "$collections_response" | grep -q '"collections":\[\]'; then
            echo -e "${GREEN}Collections endpoint is clean (no collections)${NC}"
        else
            echo -e "${YELLOW}⚠ Collections endpoint returned: $collections_response${NC}"
        fi
    else
        echo -e "${RED}✗ Could not verify collections endpoint${NC}"
    fi
    
    # Check pipelines endpoint
    if pipelines_response=$(curl -s --max-time 10 http://localhost:8888/v1/pipelines 2>/dev/null); then
        if echo "$pipelines_response" | grep -q '"pipelines":\['; then
            echo -e "${GREEN}Pipelines endpoint is working${NC}"
        else
            echo -e "${YELLOW}Pipelines endpoint returned: $pipelines_response${NC}"
        fi
    else
        echo -e "${RED}✗ Could not verify pipelines endpoint${NC}"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}This script will:${NC}"
    echo "1. Stop all CVDS services"
    echo "2. Remove persistent data volumes (milvus_data, etcd_data)"
    echo "3. Start services with fresh volumes"
    echo "4. Verify the cleanup was successful"
    echo ""
    echo -e "${RED}WARNING: This will delete all indexed collections and data!${NC}"
    echo ""
    
    # Check if services are running and stop them
    if check_services_running; then
        stop_services
    else
        echo -e "${GREEN}Services are already stopped${NC}"
    fi
    
    # Clean volumes
    clean_volumes
    
    # Start services
    start_services
    
    # Wait for services to be ready
    wait_for_services
    
    # Verify cleanup
    verify_cleanup
    
    echo ""
    echo -e "${GREEN}Volume cleanup completed successfully!${NC}"
    echo ""
}

# Run main function
main "$@"
