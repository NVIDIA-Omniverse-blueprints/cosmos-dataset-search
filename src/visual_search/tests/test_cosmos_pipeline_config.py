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

"""Tests for cosmos-embed pipeline configuration and mustache template rendering."""

import os
import tempfile
import textwrap
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import chevron
import pytest
import yaml

# isort: off
import src.visual_search.common.pipelines as pipelines_mod
from haystack import Pipeline as HaystackPipeline
from src.visual_search.common.pipelines import (
    EnabledPipeline,
    IndexPipelineInputs,
    PipelineConfig,
    QueryPipelineInputs,
    load_pipelines,
)

# isort: on


class _DummyEnabledPipeline(pipelines_mod.EnabledPipeline):
    def __init__(self):
        super().__init__(
            id="dummy",
            enabled=True,
            missing=[],
            config={},
            index_pipeline_inputs=IndexPipelineInputs(index_name=[]),
            index_pipeline=HaystackPipeline(),
            query_pipeline_inputs=QueryPipelineInputs(query=[]),
            query_pipeline=HaystackPipeline(),
        )


# --------------------------------------------------------------------------- #
# Test Fixtures                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def cosmos_pipeline_template():
    """
    Return the raw text of the Cosmos video-search pipeline template
    """
    template_path = "src/visual_search/tests/cosmos_video_search_milvus_test.mustache"
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def test_environment():
    """Environment variables for pipeline template rendering."""
    return {
        "COSMOS_EMBED_URI": "http://cosmos-embed.default.svc.cluster.local:8000",
        "MILVUS_DOCUMENT_STORE_URI": "http://milvus.default.svc.cluster.local:19530",
        "MILVUS_DB": "default",
    }


# --------------------------------------------------------------------------- #
# Template Rendering Tests                                                    #
# --------------------------------------------------------------------------- #


def test_cosmos_pipeline_template_variable_substitution(
    cosmos_pipeline_template, test_environment
):
    """Test that mustache variables are properly substituted in cosmos pipeline."""

    # Render template with test environment
    rendered = chevron.render(cosmos_pipeline_template, test_environment)

    # Verify variable substitution occurred
    assert "{{{ COSMOS_EMBED_URI }}}" not in rendered
    assert "{{{ MILVUS_DOCUMENT_STORE_URI }}}" not in rendered
    assert "{{{ MILVUS_DB }}}" not in rendered

    # Verify correct values were substituted
    assert "http://cosmos-embed.default.svc.cluster.local:8000" in rendered
    assert "http://milvus.default.svc.cluster.local:19530" in rendered
    assert 'database: "default"' in rendered


def test_cosmos_pipeline_batch_size_configuration(
    cosmos_pipeline_template, test_environment
):
    """Test that batch_size parameter is correctly configured in cosmos pipeline."""

    rendered = chevron.render(cosmos_pipeline_template, test_environment)
    parsed = yaml.safe_load(rendered)

    # Check that batch_size is in embedder configuration
    embedder_config = parsed["index"]["pipeline"]["components"]["embedder"]
    assert "batch_size" in embedder_config["init_parameters"]
    assert embedder_config["init_parameters"]["batch_size"] == 64

    # Verify embedder type is correct
    assert (
        embedder_config["type"]
        == "src.haystack.components.video.cosmos_video_embedder.CosmosVideoDocumentEmbedder"
    )


def test_cosmos_pipeline_256_dimensional_embeddings(
    cosmos_pipeline_template, test_environment
):
    """Test that pipeline is configured for 256-dimensional embeddings (cosmos-embed output)."""

    rendered = chevron.render(cosmos_pipeline_template, test_environment)
    parsed = yaml.safe_load(rendered)

    # Check index pipeline Milvus configuration
    writer_config = parsed["index"]["pipeline"]["components"]["writer"]
    milvus_config = writer_config["init_parameters"]["document_store"][
        "init_parameters"
    ]
    assert milvus_config["embedding_dim"] == 256

    # Check query pipeline Milvus configuration
    retriever_config = parsed["query"]["pipeline"]["components"]["retriever"]
    milvus_retriever_config = retriever_config["init_parameters"]["document_store"][
        "init_parameters"
    ]
    assert milvus_retriever_config["embedding_dim"] == 256


def test_cosmos_pipeline_all_embedder_components_configured(
    cosmos_pipeline_template, test_environment
):
    """Test that all cosmos embedder components are properly configured in query pipeline."""

    rendered = chevron.render(cosmos_pipeline_template, test_environment)
    parsed = yaml.safe_load(rendered)

    query_components = parsed["query"]["pipeline"]["components"]
    expected_components = {
        "text_embedder": "src.haystack.components.video.cosmos_video_embedder.CosmosTextEmbedder",
        "video_embedder": "src.haystack.components.video.cosmos_video_embedder.CosmosVideoEmbedder",
        "session_segment_embedder": "src.haystack.components.video.cosmos_video_embedder.CosmosSessionSegmentEmbedder",
    }

    for component_name, expected_type in expected_components.items():
        assert component_name in query_components
        assert query_components[component_name]["type"] == expected_type

        # Verify all have cosmos-embed URL
        init_params = query_components[component_name]["init_parameters"]
        assert (
            init_params["url"] == "http://cosmos-embed.default.svc.cluster.local:8000"
        )


# --------------------------------------------------------------------------- #
# Pipeline Configuration Validation Tests                                     #
# --------------------------------------------------------------------------- #


def test_cosmos_pipeline_config_validation(cosmos_pipeline_template, test_environment):
    """Test that cosmos pipeline configuration passes validation."""

    rendered = chevron.render(cosmos_pipeline_template, test_environment)
    parsed = yaml.safe_load(rendered)

    # This should not raise an exception
    with (
        patch(
            "src.haystack.components.video.cosmos_video_embedder.CosmosVideoDocumentEmbedder"
        ),
        patch("src.haystack.components.milvus.document_store.MilvusDocumentStore"),
        patch("src.haystack.components.milvus.document_writer.MilvusDocumentWriter"),
        patch("haystack.core.pipeline.base._types_are_compatible", return_value=True),
        patch(
            "src.haystack.components.milvus.embedding_retriever.MilvusEmbeddingRetriever.__init__",
            return_value=None,
        ),
        patch("src.haystack.components.routers.IndexTypeRouter"),
        patch("src.haystack.components.joiners.Concatenate"),
    ):

        config = PipelineConfig(**parsed)
        assert config.index is not None
        assert config.query is not None


def test_cosmos_pipeline_input_configuration(
    cosmos_pipeline_template, test_environment
):
    """Test that pipeline inputs are correctly configured."""

    rendered = chevron.render(cosmos_pipeline_template, test_environment)
    parsed = yaml.safe_load(rendered)

    # Check index pipeline inputs
    index_inputs = parsed["index"]["inputs"]
    assert "index_name" in index_inputs
    assert index_inputs["index_name"] == ["writer.index_name"]

    # Check query pipeline inputs
    query_inputs = parsed["query"]["inputs"]
    expected_query_inputs = ["query", "top_k", "index_name"]
    for input_name in expected_query_inputs:
        assert input_name in query_inputs


def test_cosmos_pipeline_connections_configured(
    cosmos_pipeline_template, test_environment
):
    """Test that pipeline connections are properly configured."""

    rendered = chevron.render(cosmos_pipeline_template, test_environment)
    parsed = yaml.safe_load(rendered)

    # Check index pipeline connections
    index_connections = parsed["index"]["pipeline"]["connections"]
    expected_index_connections = [
        ("router.to_index", "embedder.documents"),
        ("embedder.documents", "document_batcher.input"),
        ("router.embedded", "document_batcher.input"),
        ("document_batcher.output", "writer.documents"),
    ]

    for sender, receiver in expected_index_connections:
        connection_found = any(
            conn["sender"] == sender and conn["receiver"] == receiver
            for conn in index_connections
        )
        assert connection_found, f"Connection {sender} -> {receiver} not found"

    # Check query pipeline connections
    query_connections = parsed["query"]["pipeline"]["connections"]
    expected_query_connections = [
        ("router.texts", "text_embedder.texts"),
        ("router.videos", "video_embedder.video_urls"),
        ("router.clips", "session_segment_embedder.clips"),
    ]

    for sender, receiver in expected_query_connections:
        connection_found = any(
            conn["sender"] == sender and conn["receiver"] == receiver
            for conn in query_connections
        )
        assert connection_found, f"Query connection {sender} -> {receiver} not found"


# --------------------------------------------------------------------------- #
# Environment Variable Handling Tests                                         #
# --------------------------------------------------------------------------- #


def test_missing_environment_variables_handled_gracefully():
    """Test that missing environment variables are handled properly."""

    template = textwrap.dedent("""
        embedder:
            init_parameters:
                url: "{{{ COSMOS_EMBED_URI }}}"
                batch_size: 64
        """)

    # Render with missing environment variable
    empty_env = {}
    rendered = chevron.render(template, empty_env, keep=True)

    # Variable should remain unsubstituted when missing
    assert "{{ COSMOS_EMBED_URI }}" in rendered


def test_environment_variable_override():
    """Test that environment variables can be overridden for different deployments."""

    template = textwrap.dedent("""
        embedder:
            init_parameters:
                url: "{{{ COSMOS_EMBED_URI }}}"
                batch_size: {{{ COSMOS_BATCH_SIZE }}}
        """)

    # Test with custom environment
    custom_env = {
        "COSMOS_EMBED_URI": "http://cosmos-embed-dev.namespace.svc.cluster.local:8000",
        "COSMOS_BATCH_SIZE": "32",  # Custom batch size
    }

    rendered = chevron.render(template, custom_env)

    # Verify custom values are used
    assert "cosmos-embed-dev.namespace.svc.cluster.local" in rendered
    assert "batch_size: 32" in rendered


# --------------------------------------------------------------------------- #
# Migration from Vitcat Tests                                                 #
# --------------------------------------------------------------------------- #


def test_cosmos_replaces_vitcat_in_pipeline():
    """Test that cosmos-embed components replace vitcat components."""

    cosmos_template = textwrap.dedent("""
        components:
            embedder:
                type: src.haystack.components.video.cosmos_video_embedder.CosmosVideoDocumentEmbedder
                init_parameters:
                    url: "{{{ COSMOS_EMBED_URI }}}"
                    batch_size: 64
        """)

    env = {"COSMOS_EMBED_URI": "http://cosmos:8000"}
    rendered = chevron.render(cosmos_template, env)
    parsed = yaml.safe_load(rendered)

    # Verify cosmos embedder is used (not vitcat)
    embedder_type = parsed["components"]["embedder"]["type"]
    assert "cosmos_video_embedder" in embedder_type
    assert "vitcat" not in embedder_type

    # Verify batch_size parameter exists (new optimization)
    init_params = parsed["components"]["embedder"]["init_parameters"]
    assert "batch_size" in init_params
    assert init_params["batch_size"] == 64


def test_cosmos_embedding_dimensions_match_milvus():
    """Test that cosmos-embed 256D embeddings match Milvus configuration."""

    template = textwrap.dedent("""
        embedder:
            type: src.haystack.components.video.cosmos_video_embedder.CosmosVideoDocumentEmbedder
        milvus_store:
            init_parameters:
                embedding_dim: 256
        milvus_retriever:
            init_parameters:
                document_store:
                    init_parameters:
                        embedding_dim: 256
        """)

    parsed = yaml.safe_load(template)

    # All components should use 256 dimensions (cosmos-embed output)
    assert parsed["milvus_store"]["init_parameters"]["embedding_dim"] == 256
    assert (
        parsed["milvus_retriever"]["init_parameters"]["document_store"][
            "init_parameters"
        ]["embedding_dim"]
        == 256
    )


# --------------------------------------------------------------------------- #
# Pipeline Loading Integration Tests                                          #
# --------------------------------------------------------------------------- #

# @pytest.mark.asyncio
# async def test_load_cosmos_pipeline_integration(cosmos_pipeline_template, test_environment):
#     """Test loading cosmos pipeline through the pipeline loading system."""

#     with tempfile.NamedTemporaryFile(mode='w', suffix='.mustache', delete=False) as f:
#         f.write(cosmos_pipeline_template)
#         temp_file = f.name

#     try:
#         with patch.dict(os.environ, test_environment), \
#             patch('src.visual_search.common.pipelines._validate_query_pipeline', lambda _: None), \
#             patch('src.visual_search.common.pipelines._validate_index_pipeline', lambda _: None), \
#             patch('src.visual_search.common.pipelines.get_all_pipelines',
#                     return_value={'cosmos_video_search_milvus': _DummyEnabledPipeline()}), \
#             patch('src.visual_search.tests.test_cosmos_pipeline_config.EnabledPipeline', _DummyEnabledPipeline), \
#             patch('src.haystack.components.video.cosmos_video_embedder.CosmosVideoDocumentEmbedder'), \
#             patch('src.haystack.components.milvus.document_store.MilvusDocumentStore'), \
#             patch('src.haystack.components.milvus.document_writer.MilvusDocumentWriter'), \
#             patch('haystack.core.pipeline.base._types_are_compatible', return_value=True), \
#             patch('src.haystack.components.milvus.embedding_retriever.MilvusEmbeddingRetriever.__init__', return_value=None), \
#             patch('src.haystack.components.routers.IndexTypeRouter'), \
#             patch('src.haystack.components.routers.QueryTypeRouter'), \
#             patch('src.haystack.components.joiners.Concatenate'):

#             # Verify pipeline is available
#             pipelines = pipelines_mod.get_all_pipelines()
#             assert "cosmos_video_search_milvus" in pipelines

#             pipeline = pipelines["cosmos_video_search_milvus"]
#             assert isinstance(pipeline, EnabledPipeline)
#             assert pipeline.enabled is True
#     finally:
#         os.unlink(temp_file)


@pytest.mark.asyncio
async def test_load_cosmos_pipeline_integration(
    cosmos_pipeline_template, test_environment
):
    """
    Write the Cosmos pipeline template to a temporary directory, ask the real
    `load_pipelines()` to load it, and verify that the resulting entry is an
    EnabledPipeline.  Heavy components are patched to keep the test fast and
    dependency-free.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # create a .mustache file inside the temp dir
        file_path = Path(tmp_dir) / "cosmos_video_search_milvus.mustache"
        file_path.write_text(cosmos_pipeline_template)
        pipeline_name = file_path.stem  # "cosmos_video_search_milvus"

        # environment required by the loader
        env = {
            **test_environment,
            "PIPELINES_DIR": tmp_dir,
            "ALLOWED_PIPELINES": pipeline_name,
        }

        # apply lightweight patches
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, env, clear=False))
            stack.enter_context(
                patch(
                    "src.haystack.components.video.cosmos_video_embedder."
                    "CosmosVideoDocumentEmbedder"
                )
            )
            stack.enter_context(
                patch(
                    "src.haystack.components.milvus.document_store.MilvusDocumentStore"
                )
            )
            stack.enter_context(
                patch(
                    "src.haystack.components.milvus.document_writer.MilvusDocumentWriter"
                )
            )
            stack.enter_context(
                patch(
                    "src.haystack.components.milvus.embedding_retriever."
                    "MilvusEmbeddingRetriever.__init__",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch("src.haystack.components.routers.IndexTypeRouter")
            )
            stack.enter_context(
                patch("src.haystack.components.routers.QueryTypeRouter")
            )
            stack.enter_context(patch("src.haystack.components.joiners.Concatenate"))
            stack.enter_context(
                patch(
                    "haystack.core.pipeline.base._types_are_compatible",
                    return_value=True,
                )
            )
            stack.enter_context(
                patch(
                    "src.visual_search.common.pipelines._validate_query_pipeline",
                    lambda _: None,
                )
            )
            stack.enter_context(
                patch(
                    "src.visual_search.common.pipelines._validate_index_pipeline",
                    lambda _: None,
                )
            )

            #  load pipelines and assert registration
            await load_pipelines()

            pipelines = pipelines_mod.get_all_pipelines()
            assert pipeline_name in pipelines

            pipeline = pipelines[pipeline_name]
            assert isinstance(pipeline, EnabledPipeline)
            assert pipeline.enabled is True


# --------------------------------------------------------------------------- #
# Performance Configuration Tests                                             #
# --------------------------------------------------------------------------- #


def test_batch_size_performance_configuration():
    """Test different batch_size configurations for performance tuning."""

    template = textwrap.dedent("""
        embedder:
            init_parameters:
                url: "http://cosmos:8000"
                batch_size: {{{ COSMOS_BATCH_SIZE }}}
    """)

    test_cases = [
        ("1", 1),  # Individual processing
        ("16", 16),  # Small batches
        ("32", 32),  # Medium batches
        ("64", 64),  # Maximum batches
    ]

    for batch_size_str, expected_batch_size in test_cases:
        env = {"COSMOS_BATCH_SIZE": batch_size_str}
        rendered = chevron.render(template, env)
        parsed = yaml.safe_load(rendered)

        actual_batch_size = parsed["embedder"]["init_parameters"]["batch_size"]
        assert actual_batch_size == expected_batch_size


def test_cosmos_service_configuration_flexibility():
    """Test that cosmos service configuration is flexible for different environments."""

    template = textwrap.dedent("""
        embedder:
            init_parameters:
                url: "{{{ COSMOS_EMBED_URI }}}"
                batch_size: {{{ COSMOS_BATCH_SIZE }}}
                timeout: {{{ COSMOS_TIMEOUT }}}
        """)

    environments = [
        {
            "env_name": "development",
            "COSMOS_EMBED_URI": "http://cosmos-embed-dev.local:8000",
            "COSMOS_BATCH_SIZE": "16",
            "COSMOS_TIMEOUT": "120",
        },
        {
            "env_name": "staging",
            "COSMOS_EMBED_URI": "http://cosmos-embed-staging.cluster.local:8000",
            "COSMOS_BATCH_SIZE": "32",
            "COSMOS_TIMEOUT": "180",
        },
        {
            "env_name": "production",
            "COSMOS_EMBED_URI": "http://cosmos-embed.default.svc.cluster.local:8000",
            "COSMOS_BATCH_SIZE": "64",
            "COSMOS_TIMEOUT": "300",
        },
    ]

    for env_config in environments:
        env_name = env_config.pop("env_name")
        rendered = chevron.render(template, env_config)
        parsed = yaml.safe_load(rendered)

        init_params = parsed["embedder"]["init_parameters"]

        if env_name == "development":
            assert "dev.local" in init_params["url"]
            assert init_params["batch_size"] == 16
        elif env_name == "staging":
            assert "staging.cluster.local" in init_params["url"]
            assert init_params["batch_size"] == 32
        elif env_name == "production":
            assert "default.svc.cluster.local" in init_params["url"]
            assert init_params["batch_size"] == 64
