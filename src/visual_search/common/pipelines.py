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

import asyncio
import logging
import os
import re
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Literal, Optional, Tuple, Union

import chevron
import networkx as nx
import yaml
from fastapi import HTTPException
from haystack import Document as HaystackDocument
from haystack import Pipeline as HaystackPipeline
from haystack.components.writers import DocumentWriter
from haystack.core.component import InputSocket
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.tracing import auto_enable_tracing
from pydantic import BaseModel, ConfigDict, Field, ValidationError, validator
from pymilvus.exceptions import MilvusException

from .models import Collection, Pipeline, QueryType

if os.getenv("TRACE_HAYSTACK", "true") == "true":
    auto_enable_tracing()


def parse_connect_string(input: str) -> Tuple[str, str]:
    split = input.split(".")
    if len(split) != 2:
        raise ValueError(
            "Unable to parse connection string. "
            f"Got {input}, but expected pattern [component_name].[socket_name]."
        )
    component_name, socket_name = split
    return component_name, socket_name


class QueryPipelineInputs(BaseModel):
    """The pipeline query inputs consist of query and
    top k elements.

    Contains information about the query pipeline's
    expected inputs.

    The strings contain both a component and param names.

    Separated by a dot.

    For example query=["embedder.text"]
    specifies that the component with name "embedder" is expected to receive
    the query string to it's "text" parameter in the component run method.
    """

    model_config = ConfigDict(extra="forbid")

    query: List[str]
    top_k: List[str] = Field(default=[])
    reconstruct: List[str] = Field(default=[])
    index_name: List[str] = Field(default=[])
    search_params: List[str] = Field(default=[])
    filters: List[str] = Field(default=[])
    labelled_embeddings: List[str] = Field(default=[])
    clf: List[str] = Field(default=[])
    regularization_strength: List[str] = Field(default=[])


class QueryPipelineConfig(BaseModel):
    inputs: QueryPipelineInputs
    pipeline: HaystackPipeline

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    @validator("pipeline", pre=True)
    def ensure_pipeline(cls, v):
        if isinstance(v, dict):
            query_pipeline_dict = deepcopy(v)
            query_pipeline = HaystackPipeline.from_dict(query_pipeline_dict)
            validate_pipeline(query_pipeline)
            return query_pipeline
        return v

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.validate_query_pipeline()

    def validate_query_pipeline(self):
        input_connections = (
            self.inputs.query + self.inputs.top_k + self.inputs.reconstruct
        )
        validate_pipeline_inputs(self.pipeline, input_connections)
        _validate_query_pipeline(self.pipeline)
        return self


class IndexPipelineInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_name: List[str]


class IndexPipelineConfig(BaseModel):
    description: Optional[str] = Field(default=None)
    inputs: Optional[IndexPipelineInputs]
    pipeline: HaystackPipeline

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    @validator("pipeline", pre=True)
    def ensure_pipeline(cls, v):
        if isinstance(v, dict):
            index_pipeline_dict = deepcopy(v)
            index_pipeline = HaystackPipeline.from_dict(index_pipeline_dict)
            validate_pipeline(index_pipeline)
            return index_pipeline
        return v

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.validate_index_pipeline()

    def validate_index_pipeline(self) -> "IndexPipelineConfig":
        haystack_pipeline = self.pipeline
        _validate_index_pipeline(haystack_pipeline)
        return self


def _validate_query_pipeline(query_pipeline: HaystackPipeline) -> None:
    document_outputs = []
    for component_name, component_sockets in query_pipeline.outputs().items():
        for socket_name, socket_info in component_sockets.items():
            if socket_info["type"] == List[HaystackDocument]:
                document_outputs.append(f"{component_name}.{socket_name}")

    if len(document_outputs) == 0:
        msg = "Found no output of type List[Document] in query pipeline."
        raise ValueError(msg)

    if len(document_outputs) > 1:
        msg = (
            "Found multiple outputs of type List[Document] "
            f"in query pipeline. "
            f"Document outputs: {document_outputs}. "
            "Pipeline must only contain one component that outputs the final list of documents. "
        )
        raise ValueError(msg)


def _validate_index_pipeline(index_pipeline: HaystackPipeline) -> None:
    document_inputs = []
    for component_name, component_sockets in index_pipeline.inputs().items():
        for socket_name, socket_info in component_sockets.items():
            if socket_info["type"] == List[HaystackDocument]:
                document_inputs.append(f"{component_name}.{socket_name}")

    if len(document_inputs) == 0:
        msg = "Found no input of type List[Document] in index pipeline."
        raise ValueError(msg)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: IndexPipelineConfig
    query: QueryPipelineConfig


class DisabledPipeline(Pipeline):
    enabled: bool = False


class EnabledPipeline(Pipeline):
    """A Haystack Pipeline along with information needed to use the pipeline."""

    enabled: bool = True
    missing: List[str] = []

    index_pipeline_inputs: IndexPipelineInputs
    index_pipeline: HaystackPipeline
    query_pipeline_inputs: QueryPipelineInputs
    query_pipeline: HaystackPipeline

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.validate_query_pipeline_inputs()

    def validate_query_pipeline_inputs(self) -> "EnabledPipeline":
        input_connections = (
            self.query_pipeline_inputs.query
            + self.query_pipeline_inputs.top_k
            + self.query_pipeline_inputs.reconstruct
        )
        validate_pipeline_inputs(self.query_pipeline, input_connections)
        _validate_query_pipeline(self.query_pipeline)
        _validate_index_pipeline(self.index_pipeline)
        return self


def get_pipeline_inputs(pipeline: HaystackPipeline) -> Dict[str, List[InputSocket]]:
    """Return Inputs to pipeline

    Parameters
    ----------
    pipeline : HaystackPipeline
        The Haystack Pipeline to inspect

    Returns
    -------
    Dict[str, Dict[str, dict]]
        Dictionary mapping compoonent name with socket name and info about the socket
    """
    pipeline_inputs = {
        name: [
            socket
            for socket in data.get("input_sockets", {}).values()
            if not socket.senders
        ]
        for name, data in pipeline.graph.nodes(data=True)
    }
    return pipeline_inputs


def validate_pipeline_inputs(
    pipeline: HaystackPipeline,
    pipeline_input_connections: List[str],
) -> None:
    """Validate that the haystack pipeline matches

    Parameters
    ----------
    pipeline : HaystackPipeline
        Haystack pipeline to validate
    pipeline_input_connections : List[str]
        List of inputs to the pipeline. This is specified in the form of a connection string

    Raises
    ------
    ValueError
        If the pipeline input connections do not match up with the required inputs to the pipeline
    """
    pipeline_component_sockets = get_pipeline_inputs(pipeline)

    # Build a set of all the required inputs in the pipeline
    required_input_connections = {
        f"{component_name}.{input_socket.name}"
        for component_name, component_sockets in pipeline_component_sockets.items()
        for input_socket in component_sockets
        if input_socket.is_mandatory
    }

    # Validate component and socket names
    seen_input_connections = set()
    for input_connection in pipeline_input_connections:
        # `input_socket` is something like "embedder.embedding" - defined in QueryPipelineInputs
        component_name, socket_name = parse_connect_string(input_connection)
        if component_name not in pipeline_component_sockets:
            msg = (
                f"Component '{component_name}' does not exist. "
                "Available components are: "
                + ", ".join(pipeline_component_sockets.keys())
            )
            raise ValueError(msg)

        component_sockets: List[InputSocket] = pipeline_component_sockets.get(
            component_name, []
        )
        socket_name = _validate_socket_name(
            component_name, socket_name, component_sockets
        )

        seen_input_connections.add(f"{component_name}.{socket_name}")

    missing_required_sockets = required_input_connections.difference(
        seen_input_connections
    )
    if missing_required_sockets:
        msg = (
            ""
            f"Missing required inputs for pipeline. "
            f"Required inputs: {list(missing_required_sockets)}"
        )
        raise ValueError(msg)


def _validate_socket_name(
    component_name: str,
    socket_name: Optional[str],
    component_sockets: List[InputSocket],
) -> str:
    component_socket_names = set(
        input_socket.name for input_socket in component_sockets
    )
    if socket_name is None:
        mandatory_socket_names = [
            input_socket.name
            for input_socket in component_sockets
            if input_socket.is_mandatory
        ]
        if not len(mandatory_socket_names) == 1:
            msg = (
                f"Invalid inputs. "
                f"expected component '{component_name}' to have only 1 mandatory input. "
                f"Found {len(mandatory_socket_names)}"
            )
            raise ValueError(msg)
        socket_name = mandatory_socket_names[0]
    elif socket_name not in component_socket_names:
        msg = (
            f"Invalid inputs specified. "
            f"Socket '{socket_name}' not found in component '{component_name}'. "
            f"Component '{component_name}' has the following sockets: "
            f"{list(component_socket_names)}. "
        )
        raise ValueError(msg)
    return socket_name


def run_index_pipeline(
    index_pipeline: HaystackPipeline,
    index_pipeline_inputs: IndexPipelineInputs,
    documents: List[HaystackDocument],
    index_name: str,
) -> Dict[str, dict]:
    """Run the index pipeline. Returning the pipeline output

    Parameters
    ----------
    index_pipeline : HaystackPipeline
        Index pipeline to run
    index_pipeline_inputs: IndexPipelineInputs
        Index pipeline input config
    documents : List[HaystackDocument]
        List of documents to index
    index_name: str
        The collection name to write to, only useful to MilvusDocumentStore

    Returns
    -------
    Dict[str, dict]
        Dictionary of pipeline output
    """
    pipeline_input: DefaultDict[str, Dict] = defaultdict(dict)
    for component_name, input_sockets in get_pipeline_inputs(index_pipeline).items():
        for input_socket in input_sockets:
            if input_socket.type == List[HaystackDocument]:
                pipeline_input[component_name][input_socket.name] = documents
    for connection in index_pipeline_inputs.index_name:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = index_name
    pipeline_output = index_pipeline.run(pipeline_input)
    return pipeline_output


def run_query_pipeline(
    query_pipeline: HaystackPipeline,
    query_pipeline_inputs: QueryPipelineInputs,
    *,
    query: QueryType,
    index_name: str,
    search_params: Dict[str, Any],
    filters: Union[str, Dict[str, Any]],
    top_k: int,
    reconstruct: bool,
    clf: Optional[Dict[str, Any]],
) -> List[HaystackDocument]:
    """Run the query pipeline. Returning retrieved documents.

    Parameters
    ----------
    query_pipeline : HaystackPipeline
        Query Pipeline to run
    query_pipeline_inputs : QueryPipelineInputs
        Query Pipeline Inputs. Specifying component sockets expecting input params.
    query : Either a `TextQuery` or `VideoQuery` or `EmbeddingQuery` or a batch of these.
        The query to run for this retrieval query pipeline
    index_name: The collection name to query
    search_params: Index specific params, for example nprobe
    filters: Extra filters, for example session_id
    top_k : int
        The top-k value indicating how many documents we'd like to retrieve.
    reconstruct : bool
        Instructs the pipeline to attempt to retrieve the embedding.
    clf: Optional[Dict[str, Any]]
        Optional linear classifier object to filter retrieved results.
    Returns
    -------
    List[HaystackDocument]
        List of haystack documents retrieved.
    """
    pipeline_input: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
    for connection in query_pipeline_inputs.query:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = query
    for connection in query_pipeline_inputs.top_k:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = top_k
    for connection in query_pipeline_inputs.reconstruct:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = reconstruct
    for connection in query_pipeline_inputs.index_name:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = index_name
    for connection in query_pipeline_inputs.search_params:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = search_params
    for connection in query_pipeline_inputs.filters:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = filters
    for connection in query_pipeline_inputs.clf:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = clf

    # we do not expect labelled embeddings and regularization strength to be provided in this mode
    # please use dedicated `run_linear_probe_pipeline` method
    for connection in query_pipeline_inputs.labelled_embeddings:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = []

    for connection in query_pipeline_inputs.regularization_strength:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = []

    pipeline_output = query_pipeline.run(pipeline_input)

    documents = []
    for component_name, component_sockets in query_pipeline.outputs().items():
        for socket_name, socket_info in component_sockets.items():
            if socket_info["type"] == List[HaystackDocument]:
                documents = pipeline_output[component_name][socket_name]

    return documents


def run_linear_probe_pipeline(
    query_pipeline: HaystackPipeline,
    query_pipeline_inputs: QueryPipelineInputs,
    *,
    query: QueryType,
    subgraph_output_names: List[str],
    labelled_embeddings: List[Tuple[List[float], bool]] = [],
    regularization_strength: float = 0.05,
) -> List[List[float]]:
    """Run the linear probe pipeline. Returns optimized queries.

    Parameters
    ----------
    query_pipeline : HaystackPipeline
        Query Pipeline to extract linear probe subgraph.
    query_pipeline_inputs : QueryPipelineInputs
        Query Pipeline Inputs. Specifying component sockets expecting input params.
    query : Either a `TextQuery` or `VideoQuery` or `EmbeddingQuery` or a batch of these.
        The query to run for this retrieval query pipeline
    subgraph_output_names: list of node names to start backtracking from to extract subgraph.
        This should include the linear probe component name.
    labelled_embeddings: list of list of floats
        Optional labelled embeddings to trigger query learner components in pipelines.
    regularization_strength: How close the learned probe embedding should be to the anchor.

    Returns
    -------
    List[List[float]]
        List of learnt queries.
    """

    linear_probe_pipeline = extract_subgraph(
        query_pipeline,
        target_node_names=subgraph_output_names,
    )
    pipeline_input: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
    for connection in query_pipeline_inputs.query:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = query
    for connection in query_pipeline_inputs.labelled_embeddings:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = labelled_embeddings

    for connection in query_pipeline_inputs.regularization_strength:
        component_name, socket_name = parse_connect_string(connection)
        pipeline_input[component_name][socket_name] = regularization_strength

    pipeline_output = linear_probe_pipeline.run(pipeline_input)

    learnt_queries = []
    for component_name, component_sockets in linear_probe_pipeline.outputs().items():
        for socket_name, socket_info in component_sockets.items():
            if socket_info["type"] == List[List[float]]:
                learnt_queries = pipeline_output[component_name][socket_name]

    return learnt_queries


def extract_subgraph(
    pipeline: HaystackPipeline,
    target_node_names: List[str],
) -> HaystackPipeline:
    """Extract a subgraph pipeline by back-tracking from target node."""

    networkx_graph = pipeline.graph
    reverse_graph = networkx_graph.reverse(copy=False)

    visited = set()
    for node_name in target_node_names:
        nodes = list(nx.dfs_preorder_nodes(reverse_graph, source=node_name))
        visited.update(nodes)
    ancestors = list(visited)
    subgraph = networkx_graph.subgraph(ancestors)

    new_pipeline = HaystackPipeline()
    for name in subgraph.nodes:
        component = pipeline.get_component(name)
        if hasattr(component, "to_dict"):
            component_dict = component.to_dict()
        else:
            component_dict = default_to_dict(component)
        if hasattr(component, "from_dict"):
            new_component = component.from_dict(component_dict)
        else:
            new_component = default_from_dict(type(component), component_dict)
        new_pipeline.add_component(name, new_component)

    for input_node, output_node, sockets in subgraph.edges:
        input_socket, output_socket = sockets.split("/")
        new_pipeline.connect(
            f"{input_node}.{input_socket}", f"{output_node}.{output_socket}"
        )
    return new_pipeline


enabled_pipelines: Dict[str, EnabledPipeline] = {}
disabled_pipelines: Dict[str, DisabledPipeline] = {}


def get_all_pipelines() -> Dict[str, Pipeline]:
    return {**enabled_pipelines, **disabled_pipelines}


def get_pipeline_by_collection(collection: Collection) -> EnabledPipeline:
    try:
        return enabled_pipelines[collection.pipeline]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline '{collection.pipeline}' not loaded or enabled.",
        )


def draw_pipeline(name: str, mode: Literal["index", "query"]) -> bytes:
    pipeline = enabled_pipelines[name]
    haystack_pipeline: HaystackPipeline
    if mode == "index":
        haystack_pipeline = pipeline.index_pipeline
    elif mode == "query":
        haystack_pipeline = pipeline.query_pipeline
    else:
        raise KeyError("Only `index` and `query` mode is permitted.")

    with tempfile.TemporaryDirectory(prefix="/tmp/") as fp:
        diagram_file = Path(fp) / "diagram.png"
        haystack_pipeline.draw(diagram_file.as_posix())
        with diagram_file.open("rb") as file:
            return file.read()


def get_document_stores(pipeline: HaystackPipeline) -> List[Any]:
    """Return List of document stores present in the provided pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        A Haystack Pipeline that may contain document
        stores as attributes of components

    Returns
    -------
    List[Any]
        List of Document Stores in the Pipeline
    """
    document_stores = set()
    for _, instance in pipeline.graph.nodes(data="instance"):
        if hasattr(instance, "document_store"):
            document_stores.add(instance.document_store)
        elif hasattr(instance, "_document_store"):
            document_stores.add(instance._document_store)
    return list(document_stores)


def validate_pipeline(pipeline: HaystackPipeline) -> None:
    """Check that the pipeline satisfies requirements expected by service."""
    for _, instance in pipeline.graph.nodes(data="instance"):
        if (
            isinstance(instance, DocumentWriter)
            and instance.policy != DuplicatePolicy.OVERWRITE
        ):
            msg = (
                "Expecting DocumentWriter.policy to be OVERWRITE. "
                f"Found policy: {instance.policy}"
            )
            raise ValueError(msg)
    for document_store in get_document_stores(pipeline):
        if isinstance(document_store, InMemoryDocumentStore):
            continue
        assert hasattr(document_store, "create_index"), (
            f"Document Store {type(document_store)} "
            "does not provide a `create_index` method. "
            "This method is required for deleting the index. "
            "Please add a `create_index` method to the document store. "
        )
        assert hasattr(document_store, "delete_index"), (
            f"Document Store {type(document_store)} "
            "does not provide a `delete_index` method. "
            "This method is required for deleting the index. "
            "Please add a `delete_index` method to the document store. "
        )


def is_mustache_template_balanced(template: str) -> bool:
    """Check if mustache template contains balanced braces.

    Parameters
    ----------
    template : str
        The string value of a mustache template

    Returns
    -------
    bool
        True if template contains balanced braces, otherwise False
    """
    for num_braces in [2, 3]:
        if num_braces == 2:
            left_pattern = r"[^{]\{\{[^{]"
            right_pattern = r"[^}]\}\}[^}]"
        elif num_braces == 3:
            left_pattern = r"[^{]\{\{\{[^{]"
            right_pattern = r"[^}]\}\}\}[^}]"
        if len(re.findall(left_pattern, template)) != len(
            re.findall(right_pattern, template)
        ):
            return False
    return True


def extract_mustache_variables(template: str, num_braces: int = 3) -> List[str]:
    """Extract variables enclosed with mustache curly brackets

    Parameters
    ----------
    template : str
        Template string
    num_braces : int, optional
        The number of braces to match on, by default 3

    Returns
    -------
    List[str]
        List of varaibles inside the mustache template

    Raises
    ------
    ValueError
        If num_braces is not 2 or 3
        or
        If unbalanced braces found in the template.
    """
    if num_braces == 2:
        pattern = r"[^{]\{\{\s*([^{}\s]+?)\s*\}\}[^}]"
    elif num_braces == 3:
        pattern = r"[^{]\{\{\{\s*([^{}\s]+?)\s*\}\}\}[^}]"
    else:
        error_msg = "Invalid number of braces. Use 2 or 3."
        raise ValueError(error_msg)

    matches = re.findall(pattern, template)
    variables = list(set(matches))
    return variables


async def load_pipelines() -> None:
    pipelines_dir = os.getenv("PIPELINES_DIR", "src/visual_search/pipelines")

    # todo: fail if we cannot load a pipeline for any reason and get rescheduled
    async def load_pipeline(name: str, file_path: str) -> None:
        start_time = asyncio.get_event_loop().time()
        while True:
            try:
                # the context for the mustache template are our environmental variables
                context = dict(os.environ)

                # Check if any required environmental variables are missing
                missing = []
                enabled = True

                with open(Path(file_path), "r") as f:
                    pipeline_file_str = f.read()
                    if not is_mustache_template_balanced(pipeline_file_str):
                        error_msg = (
                            "Found unbalanced mustache braces in template file for pipeline "
                            f"'{name}'"
                        )
                        raise ValueError(error_msg)
                    vars_with_two_braces = extract_mustache_variables(
                        pipeline_file_str, num_braces=2
                    )
                    if vars_with_two_braces:
                        error_msg = (
                            f"Found {len(vars_with_two_braces)} mustache variable(s) "
                            "enclosed with 2 braces. "
                            f"In template file for pipeline '{name}'. "
                            "Please use 3 braces to enclose your variables "
                            "in the pipeline template file."
                        )
                        raise ValueError(error_msg)
                    required_vars = extract_mustache_variables(
                        pipeline_file_str, num_braces=3
                    )

                for required_var in required_vars:
                    if required_var not in context:
                        # if not in env vars, then required variable wasn't rendered
                        context[required_var] = f"_MISSING_{required_var}_ENV_VAR_"
                        missing.append(required_var)
                        enabled = False

                with open(Path(file_path), "r") as f:
                    rendered_mustache = chevron.render(
                        f,
                        context,
                        keep=True,
                    )  # type: ignore

                    pipeline_dict = yaml.safe_load(rendered_mustache)

                if not enabled:
                    disabled_pipelines[name] = DisabledPipeline(
                        id=name,
                        missing=missing,
                        config=pipeline_dict,
                    )
                    break

                try:
                    pipeline_config = PipelineConfig(**pipeline_dict)
                except ValidationError as e:
                    msg = f"Failed to load pipeline from '{file_path}'"
                    raise ValueError(msg) from e

                enabled_pipelines[name] = EnabledPipeline(
                    id=name,
                    config=pipeline_dict,
                    index_pipeline=pipeline_config.index.pipeline,
                    index_pipeline_inputs=pipeline_config.index.inputs,
                    query_pipeline=pipeline_config.query.pipeline,
                    query_pipeline_inputs=pipeline_config.query.inputs,
                )
                break  # Break the loop if successful

            except MilvusException as e:
                logging.info("Waiting for pipeline services to come up...")
                current_time = asyncio.get_event_loop().time()
                if current_time - start_time > 300:  # 300 seconds = 5 minutes
                    logging.error(
                        "Unable to load pipelines after attempting for 300 seconds."
                    )
                    raise e
                await asyncio.sleep(10)  # Wait for 10 seconds before retrying

    # Pipelines are stored in pipelines/ as .mustache files. A pipeline is made up of
    # an index pipeline and a query pipeline.

    logging.info(f"Loading pipelines from {pipelines_dir}")

    allowed_pipelines = os.getenv("ALLOWED_PIPELINES", None)
    if allowed_pipelines:
        pipelines = set(allowed_pipelines.split(","))
    else:
        pipelines = set(
            p.name.split(".")[0] for p in Path(pipelines_dir).glob("*.mustache")
        )
    for name in pipelines:
        logging.info(f" pipeline: {name}")

        # Load the pipelines, download the relevant models, etc
        await load_pipeline(name, f"{pipelines_dir}/{name}.mustache")
