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

# Makefile for Cosmos Video Dataset Search

.PHONY: help install build test lint format clean docker run
.DEFAULT_GOAL := help

PYTHON := python3.10
UV := uv
DOCKER_BUILDKIT := 1
IMAGE_TAG ?= latest
PYTHON_BASE_IMAGE ?= python-base:latest
REGISTRY ?= nvcr.io/nvidia/blueprint
DATA_DIR ?= $(HOME)/data
K400_JSON := $(DATA_DIR)/kinetics400_test.jsonl
LIMIT ?=

RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m

help:
	@echo "$(BLUE)Cosmos Video Dataset Search - Build System$(NC)"
	@echo "$(BLUE)==========================================$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# ==============================================================================
# Environment Setup
# ==============================================================================

install:
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(MAKE) install-python
	@echo "$(BLUE)Installing CDS client CLI...$(NC)"
	$(MAKE) install-cds-cli
	@echo ""
	@echo "$(BLUE)----------------------------------------------------------$(NC)"
	@echo "Activate the environment with: source .venv/bin/activate"
	@echo "$(BLUE)----------------------------------------------------------$(NC)"

# Install only the optional benchmark extras (datasets, etc.)
.PHONY: install-benchmark
install-benchmark: install-python ## Install benchmarking/accuracy dependencies
	@echo "$(BLUE)Installing benchmark extras (datasets, pandas …)…$(NC)"
	UV_GIT_LFS=1 $(UV) sync --extra benchmark --index-strategy unsafe-best-match

install-python:
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	$(UV) venv .venv --python $(PYTHON)
	@echo "$(BLUE)Installing all dependencies from pyproject.toml...$(NC)"
	UV_GIT_LFS=1 $(UV) sync --all-extras --index-strategy unsafe-best-match

install-cds-cli: ## Install CDS client CLI as 'cds' command - Run make install-python first
	@echo "$(BLUE)Installing CDS client CLI ...$(NC)"
	$(UV) pip install --python .venv/bin/python -e ".[client]" --index-strategy unsafe-best-match
	@echo ""
	@echo "$(BLUE)Verifying installation...$(NC)"
	@. .venv/bin/activate && which cds > /dev/null 2>&1 && echo "$(GREEN)✓ CDS CLI installed successfully!$(NC)" || echo "$(RED)✗ Installation verification failed$(NC)"
	@echo ""
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(YELLOW)To use the CLI:$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "1. Activate the virtual environment:"
	@echo "   $(GREEN)source .venv/bin/activate$(NC)"
	@echo ""
	@echo "2. Verify installation:"
	@echo "   $(GREEN)cds --help$(NC)"
	@echo ""
	@echo "3. Configure your API endpoint:"
	@echo "   $(GREEN)cds config set --profile <profile_name>$(NC) "
	@echo "   $(GREEN)'local' for docker-compose based local deployment$(NC)"
	@echo "   $(GREEN)'default' as the default profile$(NC)"
	@echo ""
	@echo "   $(GREEN)or create ~/.config/cds/config with the following contents:$(NC)"
	@echo "   $(GREEN)[default]$(NC)"
	@echo "   $(GREEN)api_endpoint = http://<host_ip>:<port>$(NC)"
	@echo ""
	@echo "   $(GREEN)[local]$(NC)"
	@echo "   $(GREEN)api_endpoint = http://localhost:8888 (for docker-compose based local deployment)$(NC)"
	@echo ""
	@echo "4. Start using CDS:"
	@echo "   $(GREEN)cds pipelines list --profile <profile_name>$(NC)"
	@echo "   $(GREEN)cds collections list --profile <profile_name>$(NC)"
	@echo ""
	@echo "5. To start the CDS services locally, run:$(NC)"
	@echo "  $(GREEN) make build-docker$(NC)"
	@echo "  $(GREEN) make test-integration-up$(NC)"
	@echo ""
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

# ==============================================================================
# Docker
# ==============================================================================

download-ce1-model: ## Download the pinned CE1 OSS model snapshot for Compose
	@bash deploy/standalone/scripts/download_ce1_model.sh

build-docker: download-ce1-model
	$(MAKE) IMAGE_TAG=$(if $(filter file,$(origin IMAGE_TAG)),local,$(IMAGE_TAG)) \
		build-python-base build-cosmos-embed-oss-pytorch \
		build-visual-search
	@echo "$(BLUE)Building all Docker images...$(NC)"
	@echo "$(YELLOW)Note: Dependencies will be installed in containers using frozen lock files$(NC)"
	@echo "$(GREEN)Docker images built successfully.$(NC)"

build-python-base:
	@echo "$(BLUE)Building Python base image...$(NC)"
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -f docker/base/python-base.Dockerfile -t "$(PYTHON_BASE_IMAGE)" .

build-visual-search: build-python-base
	@echo "$(BLUE)Building visual search service image...$(NC)"
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build --build-arg IMAGE_TAG=$(IMAGE_TAG) --build-arg PYTHON_BASE_IMAGE="$(PYTHON_BASE_IMAGE)" -f docker/services/visual-search.Dockerfile -t visual-search:$(IMAGE_TAG) .

build-cosmos-embed-oss-pytorch: build-python-base ## Build PyTorch-capable CE1 OSS service image
	@echo "$(BLUE)Building Cosmos-Embed OSS PyTorch service image...$(NC)"
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build --build-arg PYTHON_BASE_IMAGE="$(PYTHON_BASE_IMAGE)" -f docker/services/cosmos-embed-oss-pytorch.Dockerfile -t cvds-cosmos-embed-oss-pytorch:$(IMAGE_TAG) .

build-host-setup:
	@echo "$(BLUE)Building host setup container...$(NC)"
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -f infra/blueprint/host-setup-docker/Dockerfile -t host-setup:$(IMAGE_TAG) .


# ==============================================================================
# Local Unit Testing
# ==============================================================================

test-unit-local: check-install
	@echo "$(BLUE)Running unit tests locally...$(NC)"
	$(MAKE) test-visual-search-local
	$(MAKE) test-haystack-local
	$(MAKE) test-models-local

test-visual-search-local: check-install
	@echo "$(BLUE)Running visual search unit tests...$(NC)"
	. .venv/bin/activate && pytest src/visual_search/common/tests/ -v
	. .venv/bin/activate && pytest src/visual_search/tests/test_app_instrumentation.py -v
	. .venv/bin/activate && pytest src/visual_search/tests/test_bulk_indexing.py -v
	. .venv/bin/activate && pytest src/visual_search/tests/test_curator_parquet_converter.py -v
	. .venv/bin/activate && pytest src/visual_search/tests/test_pipelines.py -v
	. .venv/bin/activate && pytest src/visual_search/tests/test_retrieval.py -v
	. .venv/bin/activate && pytest src/visual_search/v1/apis/utils/test_milvus_utils.py -v
	. .venv/bin/activate && pytest src/visual_search/tests/test_k8s_secrets.py -v
	. .venv/bin/activate && pytest src/visual_search/tests/test_nvcf_file_based_secrets_manager.py -v
	. .venv/bin/activate && pytest src/visual_search/tests/test_cosmos_document_indexing.py -v
	. .venv/bin/activate && pytest src/visual_search/tests/test_cosmos_pipeline_config.py -v
	. .venv/bin/activate && pytest src/visual_search/tests/test_collections.py -v
	. .venv/bin/activate && pytest src/visual_search/tests/test_document_deletion.py -v

test-haystack-local: check-install
	@echo "$(BLUE)Running haystack unit tests...$(NC)"
	. .venv/bin/activate && pytest src/haystack/tests/serializer_test.py -v
	. .venv/bin/activate && pytest src/haystack/components/tests/joiners_test.py -v
	. .venv/bin/activate && pytest src/haystack/components/milvus/tests/document_store_test.py -v
	. .venv/bin/activate && pytest src/haystack/components/milvus/tests/filter_utils_test.py -v
	. .venv/bin/activate && pytest src/haystack/components/milvus/tests/schema_utils_test.py -v
	. .venv/bin/activate && pytest src/haystack/tests/test_cosmos_all_components.py -v
	. .venv/bin/activate && pytest src/haystack/tests/test_cosmos_integration.py -v
	. .venv/bin/activate && pytest src/haystack/tests/test_cosmos_video_embedder.py -v

test-cosmos-embed-oss-local: check-install
	@echo "$(BLUE)Running CE1 OSS service tests...$(NC)"
	. .venv/bin/activate && pytest src/cosmos_embed_oss/tests/ -v

test-models-local: check-install
	@echo "$(BLUE)Running model unit tests...$(NC)"
	. .venv/bin/activate && pytest src/models/linear_classifier/tests/model_test.py -v

test-all-unit-local: check-install
	@echo "$(BLUE)Running all unit tests...$(NC)"
	. .venv/bin/activate && pytest src/visual_search/common/tests/ -v
	. .venv/bin/activate && pytest src/cosmos_embed_oss/tests/ -v
	. .venv/bin/activate && pytest src/visual_search/tests/ -v
	. .venv/bin/activate && pytest src/haystack/tests/ -v
	. .venv/bin/activate && pytest src/haystack/components/tests/ -v
	. .venv/bin/activate && pytest src/haystack/components/milvus/tests/ -v
	. .venv/bin/activate && pytest src/models/linear_classifier/tests/ -v
	. .venv/bin/activate && pytest src/visual_search/v1/apis/utils/test_milvus_utils.py -v

test-watch-local: check-install
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	. .venv/bin/activate && pytest-watch src/

test-coverage-local: check-install
	@echo "$(BLUE)Running tests with coverage report...$(NC)"
	. .venv/bin/activate && pytest src/ -v --cov=src --cov-report=html --cov-report=xml

test-specific-local: check-install ## Run specific test file - usage: make test-specific-local TEST=src/visual_search/tests/test_retrieval.py
	@echo "$(BLUE)Running specific test: $(TEST)...$(NC)"
	. .venv/bin/activate && pytest $(TEST) -v

test-module-local: check-install ## Run tests for specific module - usage: make test-module MODULE=visual_search
	@echo "$(BLUE)Running tests for module: $(MODULE)...$(NC)"
	. .venv/bin/activate && pytest src/$(MODULE)/tests/ -v

# ==============================================================================
# Local Integration Testing
# ==============================================================================

test-integration: ## Run integration tests against full service stack
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(MAKE) test-integration-up
	$(MAKE) test-integration-run
	$(MAKE) test-integration-down

test-integration-up: check-install ## Start integration test environment
	@echo "$(BLUE)Starting docker compose deployment...$(NC)"
	@echo "$(YELLOW)----------------------------------------$(NC)"
	@echo "$(BLUE)Validating environment configuration...$(NC)"
	cd deploy/standalone && docker compose -f docker-compose.build.yml run --rm validate-env
	@echo "$(YELLOW)----------------------------------------$(NC)"
	@echo ""
	@echo "$(BLUE)Starting services...$(NC)"
	cd deploy/standalone && docker compose -f docker-compose.build.yml up -d --remove-orphans
	@echo ""
	@echo "$(YELLOW)----------------------------------------$(NC)"
	@echo "$(BLUE)Waiting for services to be ready...$(NC)"
	@echo "$(YELLOW)This may take a few minutes for GPU services to initialize...$(NC)"
	python scripts/wait_for_services.py --services milvus cosmos-embed visual-search
	@echo ""
	@echo "$(YELLOW)----------------------------------------$(NC)"
	@echo "$(BLUE)Services started...$(NC)"
	@echo "For local profile, run the following commands:"
	@echo "$(GREEN)Run 'cds config set' for 'default' profile and set the API endpoint to http://<ip_address>:8888 to configure the CLI$(NC)"
	@echo "$(GREEN)Run 'cds pipelines list' to verify the pipelines are running$(NC)"
	@echo ""
	@echo "$(GREEN)for other profiles, run 'cds config set --profile <profile>' to configure the CLI$(NC)"
	@echo "$(GREEN)Run 'cds pipelines list --profile <profile>' to verify the pipelines are running$(NC)"
	@echo ""
	@echo "$(YELLOW)----------------------------------------$(NC)"

test-integration-run: ## Run integration test scripts
	@echo "$(BLUE)Running integration test script...$(NC)"
	@echo "$(GREEN)Run 'cds config set' for 'default' profile and set the API endpoint to http://<ip_address>:8888 to configure the CLI$(NC)"
	@echo "$(GREEN)Run 'cds pipelines list' to verify the services are up to run the tests$(NC)"
	@echo "$(YELLOW)----------------------------------------$(NC)"
	. .venv/bin/activate && python src/visual_search/scripts/run_integration_test_minimal.py
	@echo ""
	@echo "$(YELLOW)----------------------------------------$(NC)"
	@echo "$(BLUE)Running Cosmos video end-to-end test...$(NC)"
	@echo "$(YELLOW)----------------------------------------$(NC)"
	. .venv/bin/activate && python src/visual_search/scripts/run_cosmos_video_integration_test.py
	@echo "$(YELLOW)----------------------------------------$(NC)"

test-integration-down: ## Stop integration test environment and clean all state
	@echo "$(BLUE)Stopping integration test environment and cleaning all state...$(NC)"
	cd deploy/standalone && docker compose -f docker-compose.build.yml down -v --remove-orphans || true

test-integration-logs: ## Show logs from integration test services
	@echo "$(BLUE)Showing integration test logs...$(NC)"
	docker compose -f deploy/standalone/docker-compose.build.yml logs -f

test-visual-search: check-install ## Run visual search
	@echo "$(BLUE)Running visual search...$(NC)"
	. .venv/bin/activate && visual-search

test-integration-clean: ## Clean up integration test environment completely
	@echo "$(BLUE)Cleaning up integration test environment...$(NC)"
	docker compose -f deploy/standalone/docker-compose.build.yml down -v --remove-orphans || true
	docker compose -f deploy/standalone/docker-compose-no-vs.build.yml down -v --remove-orphans || true
	@echo "$(BLUE)Removing unused Docker images...$(NC)"
	docker image prune -f
	@echo "$(BLUE)Removing unused Docker volumes...$(NC)"
	docker volume prune -f
	@echo "$(BLUE)Cleaning Docker system cache...$(NC)"
	docker system prune -f
	@echo "$(GREEN)All integration test state cleaned!$(NC)"

# ==============================================================================
# Dataset Preparation
# ==============================================================================

# make prepare-dataset CONFIG=scripts/k400_test_500.yaml
prepare-dataset: check-install
	@if [ -z "$(CONFIG)" ]; then \
	  echo "$(RED)ERROR: CONFIG=<path_to_yaml> is required$(NC)"; exit 2; \
	fi
	@echo "$(BLUE)Preparing dataset using $(CONFIG)...$(NC)"
	. .venv/bin/activate && python scripts/prepare_dataset.py $(CONFIG)

# ================================================
# Collection Ingestion Targets
# ================================================

ingest: check-install ## Ingest dataset into collection (use INGEST_FLAGS for options)
	@echo "$(BLUE)Ingesting dataset into collection with flags: $(INGEST_FLAGS)$(NC)"
	. .venv/bin/activate && python scripts/evals/accuracy/ingest_only.py $(INGEST_FLAGS)

ingest-msrvtt: ## Ingest full MSRVTT dataset into collection
	@echo "$(BLUE)Ingesting MSRVTT dataset into collection...$(NC)"
	$(MAKE) ingest INGEST_FLAGS="--dataset friedrichor/MSR-VTT --collection-name 'MSR-VTT Collection' --s3-endpoint $(S3_ENDPOINT)"

ingest-msrvtt-small: ## Ingest 100 MSRVTT videos into collection (for quick testing)
	@echo "$(BLUE)Ingesting MSRVTT dataset (100 videos) into collection...$(NC)"
	$(MAKE) ingest INGEST_FLAGS="--dataset friedrichor/MSR-VTT --collection-name 'MSR-VTT Small Collection' --limit 100 --s3-endpoint $(S3_ENDPOINT)"

# ==============================================================================
# Accuracy / Benchmarking
# ==============================================================================

# make accuracy ACC_FLAGS="--dataset-file=/data/my.jsonl --video-dir=/data/videos --top-k=5"
ACC_FLAGS ?=
INGEST_FLAGS ?=
S3_ENDPOINT ?= http://localhost:4566
ACCURACY_FLAGS ?=

accuracy: check-install
	@echo "$(BLUE)Running accuracy evaluation with flags: $(ACC_FLAGS)$(NC)"
	. .venv/bin/activate && python scripts/evals/accuracy/evaluate.py $(ACC_FLAGS)

accuracy-msrvtt:
	@echo "$(BLUE)Running MSRVTT accuracy evaluation on full test split...$(NC)"
	$(MAKE) accuracy ACC_FLAGS="--dataset friedrichor/MSR-VTT --top-k=10 --relevance-level=instance"

accuracy-only: check-install ## Run accuracy evaluation on existing collection (use ACCURACY_FLAGS for options)
	@echo "$(BLUE)Running accuracy evaluation on existing collection with flags: $(ACCURACY_FLAGS)$(NC)"
	. .venv/bin/activate && python scripts/evals/accuracy/accuracy_only.py $(ACCURACY_FLAGS)

accuracy-only-msrvtt: check-install ## Run accuracy evaluation on existing MSR-VTT collection
	@echo "$(BLUE)Running accuracy evaluation on MSR-VTT collection...$(NC)"
	$(MAKE) accuracy-only ACCURACY_FLAGS="--collection-name 'MSR-VTT Accuracy Test' --dataset friedrichor/MSR-VTT --top-k=10 --relevance-level=instance"

accuracy-only-msrvtt-small: check-install ## Run accuracy evaluation on existing MSR-VTT collection (100 queries)
	@echo "$(BLUE)Running accuracy evaluation on MSR-VTT collection (100 queries)...$(NC)"
	$(MAKE) accuracy-only ACCURACY_FLAGS="--collection-name 'MSR-VTT Accuracy Test' --dataset friedrichor/MSR-VTT --top-k=10 --relevance-level=instance --limit=100"

# ================================================
# Volume Management Targets
# ================================================

clean-volumes: ## Clean Docker volumes to fix metadata corruption issues
	@echo "$(BLUE)Cleaning Docker volumes...$(NC)"
	./scripts/integration-tools/clean-volumes.sh

# ==============================================================================
# Push Targets
# ==============================================================================
# Usage: make push-visual-search IMAGE_TAG=debug-20250729
push-visual-search: build-visual-search ## Push visual-search image to registry
	@echo "$(BLUE)Pushing visual-search image to registry...$(NC)"
	@echo "$(YELLOW)Tagging visual-search:$(IMAGE_TAG) -> $(REGISTRY)/cosmos-dataset-search:$(IMAGE_TAG)$(NC)"
	docker tag visual-search:$(IMAGE_TAG) $(REGISTRY)/cosmos-dataset-search:$(IMAGE_TAG)
	@echo "$(YELLOW)Pushing $(REGISTRY)/cosmos-dataset-search:$(IMAGE_TAG)$(NC)"
	docker push $(REGISTRY)/cosmos-dataset-search:$(IMAGE_TAG)
	@echo "$(GREEN)Successfully pushed visual-search image!$(NC)"

# ==============================================================================
# Cleanup
# ==============================================================================

clean:
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf .venv/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf bin/
	rm -rf node_modules/

clean-docker:
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker system prune -f
	docker image prune -f

clean-models:
	@echo "$(BLUE)Cleaning model weights...$(NC)"
	rm -rf models/*/

# ==============================================================================
# Utilities
# ==============================================================================

shell:
	@echo "$(BLUE)Opening development shell...$(NC)"
	. .venv/bin/activate && $(SHELL)

check:
	@echo "$(BLUE)Running all checks...$(NC)"
	$(MAKE) lint
	$(MAKE) test-unit-local

check-install:
	@echo "$(BLUE)Checking for required dependencies...$(NC)"
	@if [ ! -f "uv.lock" ]; then \
		echo "$(RED)ERROR: uv.lock file not found.$(NC)"; \
		echo "$(YELLOW)Please run 'make install' to generate the lock file.$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f "pyproject.toml" ]; then \
		echo "$(RED)ERROR: pyproject.toml file not found.$(NC)"; \
		echo "$(YELLOW)Please ensure you're in the project root directory.$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f ".venv/bin/activate" ]; then \
		echo "$(RED)ERROR: .venv/bin/activate file not found.$(NC)"; \
		echo "$(YELLOW)Please run 'make install' to generate the activate file.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Project files ready for Docker builds$(NC)"
