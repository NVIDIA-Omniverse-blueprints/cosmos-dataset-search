# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""CDS (Cosmos Dataset Search) Client CLI for visual indexing and search."""

import json
import logging
import os
from collections import defaultdict
from functools import lru_cache
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any, DefaultDict, Dict, Final, List, Optional, Union
from uuid import UUID

import fire
import pandas as pd
import ray
import requests
import urllib3
import yaml
from pydantic import BaseModel
from ray.util.queue import Queue
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.haystack.components.milvus.schema_utils import MetadataConfig
from src.visual_search.client.config import (
    DEFAULT,
    Profile,
    configure_credentials,
    load_config,
)
from src.visual_search.client.io_utils import (
    CSVLogger,
    DataFetcher,
    EmbeddingParquetProcessor,
    FileBatchProcessor,
    is_s3_path,
)

IMAGE_EXTENSIONS: Final = [".jpeg", ".jpg", ".png", ".JPEG"]
SUPPORTED_MIME_TYPES: Final = ["video/mp4", "image/jpeg", "image/png", "episode/zip"]
SUPPORTED_EXISTENCE_CHECKS: Final = ["must", "with_timeout", "skip"]

root = logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


urllib3.disable_warnings()


class PrettyDict(dict):
    """Wrapper for dict to print nicely."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return json.dumps(self, indent=2)


def request_token(token_endpoint: str, api_key: str) -> str:
    """Requests authentication token used for interaction with service."""
    response = requests.get(
        token_endpoint,
        headers={"Authorization": f"ApiKey {api_key}"},
        # Accept self-signed TLS
        verify=False,
    )
    response.raise_for_status()
    token = response.json()["token"]
    return token


@lru_cache
def load_profile(profile: str) -> Profile:
    """Load profile."""
    logging.info(f"Loading profile {profile}")
    config = load_config()
    return config.get_profile(profile)


@lru_cache
def get_token(profile: Profile) -> Optional[str]:
    """Loads token."""
    if profile.auth_endpoint is None or profile.auth_key is None:
        return None
    logging.info(f"Requesting auth token to {profile.auth_endpoint}")
    return request_token(profile.auth_endpoint, profile.auth_key)


def get_headers(profile: Profile) -> Dict[str, str]:
    """Gets headers with authentication, if needed."""

    token = get_token(profile)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class CollectionsConfig(BaseModel):
    """Collections configuration parameters."""

    index_config: Dict[str, Any] = {}
    collection_config: Dict[str, Any] = {}
    tags: Dict[str, Any] = {}
    metadata_config: MetadataConfig = MetadataConfig()


class Collections:
    """Collection commands for CDS (Cosmos Dataset Search)."""

    @staticmethod
    def create(
        pipeline: str,
        collection_id: Optional[UUID] = None,
        config_yaml: Optional[str] = None,
        name: str = "Collection created with CDS client",
        profile: str = DEFAULT,
        index_type: Optional[str] = None,
    ) -> PrettyDict:
        """Create collection (index of vectors and metadata).

        :param pipeline: Pipeline used to create collection.
        :param collection_id: Optional uuid of collection.
        :param config_yaml: JSON file with collection/index configuration.
            For example,
            ```
            partition_key: ""
            index_config:
                index_type: GPU_CAGRA
                params:
                    intermediate_graph_degree: 32
                    graph_degree: 64
                    build_algo: IVF_PQ
                    cache_dataset_on_device: "true"
                    adapt_for_cpu: "false"
            collection_config:
                properties:
                    mmap.enabled: false
            tags: {}
            metadata_config:
                allow_dynamic_schema: true
                fields: []
            ```
        :param name: Description of collection.
        :param profile: Credentials profile name for client.
        :param index_type: Override the index type (e.g., 'IVF_SQ8', 'GPU_CAGRA').
            Can also be set via CVDS_INDEX_TYPE environment variable.
        """

        logging.info(f"Loading profile...")
        cfg = load_profile(profile)
        logging.info(f"Profile loaded. Loading")
        headers = get_headers(cfg)
        
        # Load configuration from yaml file
        if config_yaml is not None:
            config_path = Path(config_yaml)
            if not config_path.exists():
                raise FileNotFoundError(f"Could not find config yaml {config_path}")
        else:
            # Use default.yaml if no config specified
            default_yaml_path = Path(__file__).parent / "default.yaml"
            config_path = default_yaml_path

        logging.info(f"Loading config file...")
        with config_path.open("r", encoding="utf-8") as fp:
            yaml_content = yaml.safe_load(fp)
        configuration = CollectionsConfig(**yaml_content)
        logging.info(f"Config loaded. Overriding index type...")

        # Override index_type if specified via parameter or environment variable
        override_index_type = index_type or os.getenv("CVDS_INDEX_TYPE")
        if override_index_type:
            configuration.index_config["index_type"] = override_index_type
            # Log the override for visibility
            logging.info("Overriding index_type to: %s", override_index_type)

        payload = {
            "pipeline": pipeline,
            "name": name,
            "collection_config": configuration.collection_config,
            "index_config": configuration.index_config,
            "tags": configuration.tags,
            "metadata_config": (
                configuration.metadata_config.dict()
                if configuration.metadata_config is not None
                else MetadataConfig()
            ),
        }
        logging.info(f"collection_id: {collection_id}")
        req = f"{cfg.api_endpoint}/v1/collections" + (f"?id={collection_id}" if collection_id else "")
        logging.info(f"Posting request: \n\t{req}")
        response = requests.post(
            req,
            json=payload,
            headers=headers,
            # Accept self-signed TLS
            verify=False,
        )
        try:
            response.raise_for_status()
        except Exception as e:
            logging.exception(
                f"Got exception {e}. Response content: {response.content}"
            )
            raise e
        return PrettyDict(response.json())

    @staticmethod
    def list(profile: str = DEFAULT) -> PrettyDict:
        """List collections."""

        cfg = load_profile(profile)
        headers = get_headers(cfg)
        response = requests.get(
            f"{cfg.api_endpoint}/v1/collections",
            headers=headers,
            # Accept self-signed TLS
            verify=False,
        )
        try:
            response.raise_for_status()
        except Exception as e:
            logging.exception(
                f"Got exception {e}. Response content: {response.content}"
            )
            raise e
        return PrettyDict(response.json())

    @staticmethod
    def get(collection_id: str, profile: str = DEFAULT) -> PrettyDict:
        """Get specific collection."""

        cfg = load_profile(profile)
        headers = get_headers(cfg)
        response = requests.get(
            f"{cfg.api_endpoint}/v1/collections/{collection_id}",
            headers=headers,
            # Accept self-signed TLS
            verify=False,
        )
        try:
            response.raise_for_status()
        except Exception as e:
            logging.exception(
                f"Got exception {e}. Response content: {response.content}"
            )
            raise e
        return PrettyDict(response.json())

    @staticmethod
    def delete(collection_id: str, profile: str = DEFAULT) -> PrettyDict:
        """Delete specific collection. N.B. this is irreversible!"""

        cfg = load_profile(profile)
        headers = get_headers(cfg)
        response = requests.delete(
            f"{cfg.api_endpoint}/v1/collections/{collection_id}",
            headers=headers,
            # Accept self-signed TLS
            verify=False,
        )
        response.raise_for_status()
        return PrettyDict(response.json())


def search(
    collection_ids: Union[List[str], str],
    text_query: Optional[str] = None,
    top_k: int = 10,
    profile: str = DEFAULT,
    generate_asset_url: bool = True,
) -> PrettyDict:
    """Search one or more collections for a text or file."""

    if not text_query:
        raise ValueError("Text query is empty!")

    collection_ids = (
        [collection_ids] if isinstance(collection_ids, str) else collection_ids
    )
    cfg = load_profile(profile)
    headers = get_headers(cfg)
    request_json = {
        "collections": collection_ids,
        "query": [{"text": text_query}],
        "params": {
            "nb_neighbors": top_k,
            "nb_probes": 1,
            "min_similarity": 0.0,
            "reconstruct": False,
        },
        "payload_keys": None,
        "generate_asset_url": generate_asset_url,
        "rerank": True,
    }
    response = requests.post(
        f"{cfg.api_endpoint}/v1/retrieval",
        json=request_json,
        headers=headers,
        # Accept self-signed TLS
        verify=False,
    )
    try:
        response.raise_for_status()
    except Exception as e:
        logging.exception(f"Got exception {e}. Response content: {response.content}")
        raise e
    return PrettyDict(response.json())


class Ingest:
    """Data ingestion commands for CDS (Cosmos Dataset Search)."""

    @staticmethod
    def embeddings(
        parquet_dataset: str,
        collection_id: Optional[str] = None,
        num_workers: int = 1,
        embeddings_col: str = "embeddings",
        id_cols: Optional[List[str]] = None,
        metadata_cols: Optional[List[str]] = None,
        fillna: bool = True,
        limit: Optional[int] = None,
        timeout: int = 60,
        profile: str = DEFAULT,
        s3_profile: Optional[str] = None,
        output_log: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Upload pre-computed embeddings with associated metadata stored in parquet.

        :param parquet_dataset: s3:// remote path to parquet dataset.
        :param collection_id: Identifier for the collection.
        :param num_workers: Number of parallel upload workers.
        :param embeddings_col: Dataframe column name for embeddings.
            Default is `embeddings`. We except the column to contain 1D numpy arrays.
        :param id_cols: Optional comma-separated list of columns in dataframe to be used to compute id.
            Default is None, which means IDs will be generated with a uuid4.
        :param metadata_cols: Optional comma-separated list of columns in dataframe for metadata.
            Default is None, which means all columns will be used, except for the embeddings column.
        :param fillna: Whether to fill NaN values in metadata columns. Default True.
        :param limit: Optional file limit.
        :param timeout: Timeout in seconds. (default: 60)
        :param profile: Credentials profile name for client.
        :param s3_profile: Optional s3 profile name in `~/.aws/config` if directory is remote.
        :param output_log: Optional file path for CSV writing results of post requests.
        """

        if collection_id is None:
            raise ValueError(
                "Please provide a collection UUID with `--collection-id <...>`!"
            )

        if not id_cols:
            raise ValueError(
                "Please provide ID columns for hashing with `--id-cols a,b,c`!"
            )

        # Disallow --mime-type for embeddings ingestion
        if mime_type is not None:
            raise ValueError("--mime-type is not supported for embeddings ingestion.")

        # Enforce S3-only ingestion for parquet datasets
        if not is_s3_path(parquet_dataset):
            raise ValueError(
                f"Embeddings ingestion only supports S3 parquet datasets (s3://...). Got: {parquet_dataset}"
            )

        cfg = load_profile(profile)
        token = get_token(cfg)

        # double check collection exists
        collection = Collections.get(collection_id=collection_id, profile=profile)
        if not collection["collection"]:
            raise KeyError(
                f"{collection_id} not found! "
                "Please create it with `cds collections create --help`."
            )

        # paginating through files, especially from remote key-value store,
        # can be time-consuming, so we spawn a separate actor that fetches files
        # and feeds the batches to Ray batch processor pool asynchronously.
        ray.init()
        job_queue = Queue()
        result_queue = Queue()
        data_fetcher = DataFetcher.remote(  # type: ignore
            queue=job_queue,
            directory_path=parquet_dataset,
            extensions=[".parquet"],
            s3_profile=s3_profile,
            batch_size=1,
            limit=limit,
            nb_consumer=num_workers,
        )
        fetcher_task = data_fetcher.fetch_data.remote()
        processors = [
            EmbeddingParquetProcessor.remote(  # type: ignore
                job_queue,
                result_queue,
                endpoint_url=cfg.api_endpoint,
                collection_id=collection_id,
                embeddings_col=embeddings_col,
                id_cols=id_cols,
                metadata_cols=metadata_cols,
                fillna=fillna,
                s3_profile=s3_profile,
                timeout=timeout,
                token=token,
            )
            for _ in range(num_workers)
        ]
        processor_tasks = [processor.process_data.remote() for processor in processors]  # type: ignore

        # set up monitoring console
        console = Console()
        console.log(f":brain: Spawned {len(processor_tasks)} parquet batch processors.")
        overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        pbar = overall_progress.add_task("[green]Processed files:", total=None)
        status_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        progress_table = Table.grid()
        progress_table.add_row(
            Panel.fit(
                overall_progress,
                title="[b]File ingestion",
                border_style="green",
                padding=(1, 1),
            ),
            Panel.fit(
                status_progress,
                title="[b]Responses",
                border_style="red",
                padding=(1, 1),
            ),
        )

        # start logging
        total = 0
        status_codes: DefaultDict[int, int] = defaultdict(lambda: 0)
        prog_tasks = dict()
        nb_processed = 0
        log_writer = CSVLogger(output_log) if output_log else None
        if log_writer:
            console.log(f":book: Logging responses in {output_log}")
        workers_done = 0

        # When a hard limit is provided we can set the total immediately so the
        # progress bar shows 0/<limit> instead of 0/? for the first batch.
        if limit:
            overall_progress.update(pbar, total=limit)

        with Live(progress_table, refresh_per_second=1, console=console):
            while True:
                result = result_queue.get(block=True, timeout=timeout)
                if result is None:
                    workers_done += 1
                    if workers_done == num_workers:
                        break
                    continue
                response, files = result
                status_code = response.status_code
                nb_processed += len(files)
                status_codes[status_code] += len(files)
                if status_code not in prog_tasks:
                    prog_tasks[status_code] = status_progress.add_task(
                        f"[yellow] Status code {status_code}:",
                        total=None,
                    )
                total = (
                    max([total, nb_processed, job_queue.size()])
                    if result is not None
                    else nb_processed
                )
                overall_progress.update(pbar, advance=len(files), total=total)
                status_progress.update(
                    prog_tasks[status_code], advance=len(files), total=total
                )
                if log_writer:
                    log_writer.log(files=files, status=status_code)

        # Wait for the DataFetcher to complete
        ray.get(fetcher_task)
        ray.shutdown()

        console.log(":rocket: Finished processing job queue!")
        for k, v in status_codes.items():
            if k == 200:
                console.log(f"[green]Processed {v} files successfully")
            else:
                console.log(f"[yellow]{v} files returned status code {k}")
        return status_codes

    @staticmethod
    def files(
        directory_path: str,
        collection_id: str,
        num_workers: int = 1,
        batch_size: int = 1,
        metadata_json: Optional[Path] = None,
        strip_directory_path: bool = True,
        extensions: List[str] = IMAGE_EXTENSIONS,
        limit: Optional[int] = None,
        timeout: int = 60,
        profile: str = DEFAULT,
        s3_profile: Optional[str] = None,
        output_log: Optional[str] = None,
        existence_check: str = "skip",
    ) -> Dict[int, int]:
        """
        Upload files from an S3-compatible storage path to a collection.

        :param directory_path: s3 path containing e.g. images (must start with s3://).
        :param collection_id: Identifier for the collection (required).
        :param num_workers: Number of parallel upload workers.
        :param batch_size: File batch size per JSON request. Relevant only if multipart is not used.
        :param metadata_json: Optional JSON file with key as the image file and value a dictionary of metadata.
        :param strip_directory_path: Whether to strip directory prefix to image path when saving to collection.
        :param extensions: Extensions to glob for. Defaults are common image extensions.
        :param limit: Optional file limit.
        :param timeout: Timeout in seconds.
        :param profile: Credentials profile name for client.
        :param s3_profile: Optional s3 profile name in `~/.aws/config`.
        :param output_log: Optional file path for CSV writing results of post requests.
        """

        if collection_id is None:
            raise ValueError(
                "Please provide a collection UUID with `--collection-id <...>`!"
            )

        # Enforce S3-only ingestion
        if not directory_path.startswith("s3://"):
            raise ValueError(
                "Ingestion is only supported from S3-compatible storage (s3://...)."
            )

        # s3_profile, metadata_json, and output_log remain optional depending on environment setup

        if batch_size < 1:
            raise ValueError(f"Batch size must be greater than 1. Got {batch_size}")

        if existence_check not in SUPPORTED_EXISTENCE_CHECKS:
            raise ValueError(
                f"{existence_check} not supported! Available are {SUPPORTED_EXISTENCE_CHECKS}"
            )

        logging.info(f"Loading profile...")
        cfg = load_profile(profile)
        logging.info(f"Profile loaded. Loading token...")
        token = get_token(cfg)
        logging.info(f"Token loaded. Checking collection exists...")
        # double check collection exists
        collection = Collections.get(collection_id=collection_id, profile=profile)
        if not collection["collection"]:
            raise KeyError(
                f"{collection_id} not found! "
                "Please create it with `cds collections create --help`."
            )
        logging.info(f"Collection exists. Loading metadata...")
        metadata_dict = dict()
        if metadata_json is not None:
            metadata_json = Path(metadata_json)
            if not metadata_json.exists():
                raise FileNotFoundError(f"{metadata_json} not found!")
            with metadata_json.open("r") as fp:
                metadata_dict = json.load(fp)
        logging.info(f"Metadata loaded. Loading base path...")
        base_path = directory_path if strip_directory_path else None
        logging.info(f"Base path loaded. Initializing Ray...")
        # paginating through files, especially from remote key-value store,
        # can be time-consuming, so we spawn a separate actor that fetches files
        # and feeds the batches to Ray batch processor pool asynchronously.
        ray.init()
        job_queue = Queue()
        result_queue = Queue()
        data_fetcher = DataFetcher.remote(  # type: ignore
            queue=job_queue,
            directory_path=directory_path,
            extensions=extensions,
            s3_profile=s3_profile,
            batch_size=batch_size,
            limit=limit,
            nb_consumer=num_workers,
        )
        logging.info(f"Fetcher task initialized. Initializing processors...")
        fetcher_task = data_fetcher.fetch_data.remote()
        logging.info(f"Fetcher task started. Initializing processors...")
        processors = [
            FileBatchProcessor.remote(  # type: ignore
                job_queue,
                result_queue,
                endpoint_url=cfg.api_endpoint,
                collection_id=collection_id,
                metadata=metadata_dict,
                base_path=base_path,
                s3_profile=s3_profile,
                timeout=timeout,
                token=token,
                existence_check=existence_check,
            )
            for _ in range(num_workers)
        ]
        processor_tasks = [processor.process_data.remote() for processor in processors]  # type: ignore
        logging.info(f"Processors initialized. Setting up monitoring console...")
        # set up monitoring console
        console = Console()
        console.log(f":brain: Spawned {len(processor_tasks)} file batch processors.")
        overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        pbar = overall_progress.add_task("[green]Processed files:", total=None)
        status_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        progress_table = Table.grid()
        progress_table.add_row(
            Panel.fit(
                overall_progress,
                title="[b]File ingestion",
                border_style="green",
                padding=(1, 1),
            ),
            Panel.fit(
                status_progress,
                title="[b]Responses",
                border_style="red",
                padding=(1, 1),
            ),
        )

        # start logging
        total = 0
        status_codes: DefaultDict[int, int] = defaultdict(lambda: 0)
        prog_tasks = dict()
        nb_processed = 0
        log_writer = CSVLogger(output_log) if output_log else None
        if log_writer:
            console.log(f":book: Logging responses in {output_log}")
        workers_done = 0
        
        if limit:
            overall_progress.update(pbar, total=limit)

        with Live(progress_table, refresh_per_second=1, console=console):
            while True:
                result = result_queue.get(block=True, timeout=timeout)
                if result is None:
                    workers_done += 1
                    if workers_done == num_workers:
                        break
                    continue
                response, files = result
                status_code = response.status_code
                nb_processed += len(files)
                status_codes[status_code] += len(files)
                if status_code not in prog_tasks:
                    prog_tasks[status_code] = status_progress.add_task(
                        f"[yellow] Status code {status_code}:",
                        total=None,
                    )
                total = (
                    max([total, nb_processed, batch_size * job_queue.size()])
                    if result is not None
                    else nb_processed
                )
                overall_progress.update(pbar, advance=len(files), total=total)
                status_progress.update(
                    prog_tasks[status_code], advance=len(files), total=total
                )
                if log_writer:
                    log_writer.log(files=files, status=status_code)

        # Wait for the DataFetcher to complete
        ray.get(fetcher_task)
        ray.shutdown()

        console.log(":rocket: Finished processing job queue!")
        for k, v in status_codes.items():
            if k == 200:
                console.log(f"[green]Processed {v} files successfully")
            else:
                console.log(f"[yellow]{v} files returned status code {k}")
        return status_codes


class Config:
    @staticmethod
    def set(profile: str = DEFAULT) -> None:
        """Configure credentials for a specific profile.
        
        :param profile: Profile name to configure (default: 'default')
        """
        configure_credentials(profile=profile)


class Pipelines:
    @staticmethod
    def list(verbose: bool = False, profile: str = DEFAULT) -> PrettyDict:
        """List available pipelines.

        :param verbose: If True will print out full configuration, default False.
        :param profile: Credentials profile name for client.
        """

        cfg = load_profile(profile)
        headers = get_headers(cfg)
        response = requests.get(
            f"{cfg.api_endpoint}/v1/pipelines",
            headers=headers,
            # Accept self-signed TLS
            verify=False,
        )
        try:
            response.raise_for_status()
        except Exception as e:
            logging.exception(
                f"Got exception {e}. Response content: {response.content}"
            )
            raise e
        json_dict = response.json()
        if not verbose:
            for pipeline in json_dict["pipelines"]:
                pipeline.pop("config")
        return PrettyDict(json_dict)


# Secrets API endpoints are not implemented
# Use Kubernetes secrets via kubectl instead:
# docker exec cds-deployment kubectl create secret generic <name> \
#   --from-literal=key1=value1 \
#   --from-literal=key2=value2


class Client:
    """Visual indexing and search microservice client.

    Commands:
      - config set: open guided credential configuration
      - pipelines list [--verbose] [--profile PROFILE]
      - collections create --pipeline PIPELINE [--collection-id UUID] [--config-yaml PATH] [--name NAME] [--profile PROFILE] [--index-type TYPE]
      - collections list [--profile PROFILE]
      - collections get COLLECTION_ID [--profile PROFILE]
      - collections delete COLLECTION_ID [--profile PROFILE]
      - ingest embeddings --parquet-dataset S3PATH --collection-id ID --id-cols COLS [other options]
      - ingest files --directory-path PATH --collection-id ID [other options]
      - search --collection-ids ID[,ID...] --text-query TEXT [--top-k K] [--profile PROFILE]

    Note: For secrets management, use kubectl to create Kubernetes secrets directly.

    Use --help for a concise overview printed by this script (enhanced),
    or run any subcommand with --help for parameter details via Python Fire.
    """

    def __init__(self) -> None:
        self.config = Config()
        self.pipelines = Pipelines()
        self.collections = Collections()
        self.ingest = Ingest()
        self.search = search

    @staticmethod
    def version() -> str:
        """Display the CDS CLI version."""
        try:
            version_str = get_version("cosmos-dataset-search")
            print(f"CDS CLI version {version_str}")
            return version_str
        except Exception:
            print("CDS CLI version: unknown")
            return "unknown"


def main() -> None:
    # Enhanced top-level help: if called with no args or explicit help flags, show a concise overview
    import sys
    
    # Handle --version flag
    if any(arg in ("-v", "--version") for arg in sys.argv[1:2]):
        try:
            version_str = get_version("cosmos-dataset-search")
            print(f"CDS CLI version {version_str}")
        except Exception:
            print("CDS CLI version: unknown")
        return
    
    if len(sys.argv) == 1 or any(arg in ("-h", "--help") for arg in sys.argv[1:2]):
        help_text = (
            "Cosmos Dataset Search Client CLI\n\n"
            "Usage:\n"
            "  CDS_CLI <section> <command> [options]\n"
            "  CDS_CLI <section> [--help]         Show help for a section/command\n"
            "  CDS_CLI --version                  Show version information\n\n"
            "About this help:\n"
            "  - Sections below group top-level commands by area (Config, Pipelines, Collections, Ingest, Search, Secrets).\n"
            "  - Each line shows a command synopsis. Run the line with --help to view parameters and defaults.\n"
            "  - Most commands accept --profile to select a credentials profile.\n\n"
            "Examples:\n"
            "  CDS_CLI pipelines list --verbose\n"
            "  CDS_CLI collections create --pipeline text --index-type GPU_CAGRA --help\n"
            "  CDS_CLI search --collection-ids <ID> --text-query \"cats playing\" --top-k 10\n\n"
            "----------------------------------------\n"
            "Commands\n"
            "----------------------------------------\n\n"
            "Global:\n"
            "  --profile PROFILE   Select credentials profile (default: default)\n\n"
            "Config:\n"
            "  config set [--profile PROFILE]\n"
            "      Guided credential configuration for the specified profile (default: default)\n\n"
            "Pipelines:\n"
            "  pipelines list [--verbose] [--profile PROFILE]\n\n"
            "Collections:\n"
            "  collections create --pipeline PIPELINE [--collection-id UUID] [--config-yaml PATH] [--name NAME] [--profile PROFILE] [--index-type TYPE]\n"
            "  collections list [--profile PROFILE]\n"
            "  collections get COLLECTION_ID [--profile PROFILE]\n"
            "  collections delete COLLECTION_ID [--profile PROFILE]\n\n"
            "Ingest:\n"
            "  ingest embeddings --parquet-dataset S3PATH --collection-id ID --id-cols COLS \\\n"
            "    [--num-workers N] [--embeddings-col NAME] [--metadata-cols COLS] [--limit N] [--timeout SEC] [--output-log PATH] [--profile PROFILE]\n"
            "  ingest files --directory-path PATH --collection-id ID \\\n"
            "    [--num-workers N] [--batch-size N] [--extensions .jpg,.png] [--limit N] [--timeout SEC] [--existence-check must|with_timeout|skip] [--output-log PATH] [--profile PROFILE]\n\n"
            "Search:\n"
            "  search --collection-ids ID[,ID...] --text-query TEXT [--top-k K] [--generate-asset-url true|false] [--profile PROFILE]\n\n"
            "Note: For secrets management, use kubectl to create Kubernetes secrets.\n"
            "  docker exec cds-deployment kubectl create secret generic <name> --from-literal=key=value\n\n"
            "Tip: run any command with --help to see parameter details.\n"
        )
        print(help_text)
        return

    fire.Fire(Client)


if __name__ == "__main__":
    main()
