# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utilities for distributed I/O client."""

import base64
import configparser
import csv
import logging
import mimetypes
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import boto3
import magic
import ray
import requests
import urllib3
from botocore.client import BaseClient
from botocore.config import Config
from ray.util.queue import Queue
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import configparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
urllib3.disable_warnings()

os.environ["AWS_ENDPOINT_URL"] = "http://localstack:4566"

def get_s3_client(s3_profile: Optional[str] = None) -> BaseClient:
    """Get s3 client.

    If s3_profile is `None`, will default to standard S3 environment variables or config.
    If s3_profile is set, will extract credentials and configurations from `~/.aws/config` and `~/.aws/credentials`.
    """
    if s3_profile is None:
        return boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", None),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", None),
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL", None),
            region_name=os.environ.get("AWS_DEFAULT_REGION", None),
        )
    else:
        session = boto3.Session(profile_name=s3_profile)
        # custom endpoints are not supported by `boto3.Session` object
        config = configparser.ConfigParser()
        config.read(os.path.expanduser("~/.aws/config"))
        section_name = f"profile {s3_profile}" if s3_profile != "default" else "default"
        endpoint_url = config.get(section_name, "endpoint_url", fallback=None)
        region = config.get(section_name, "region", fallback=None)

        config_to_fix_url_403 = Config(
            region_name=region,
            signature_version="s3v4",
            retries={
                "max_attempts": 10,
                # 'mode': 'standard'
            },
            s3={"addressing_style": "path"},
        )

        return session.client(
            "s3", config=config_to_fix_url_403, endpoint_url=endpoint_url
        )


def get_requests_session() -> requests.Session:
    """Get unique requests session."""
    session = requests.Session()
    retry = Retry(
        connect=3,
        backoff_factor=0.1,
        status_forcelist=[413, 429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Disable SSL certificate verification
    session.verify = False
    return session


def is_s3_path(uri: str) -> bool:
    return uri.startswith("s3://")


def extract_s3_bucket_and_key(url: str) -> Tuple[str, str]:
    """Extract bucket and key from a string like s3://bucket/some/prefix."""
    assert is_s3_path(url), "URL must start with 's3://'"
    s3_path = url[5:]
    bucket, key = s3_path.split("/", 1)
    return bucket, key


def list_files(
    uri: str,
    extensions: List[str],
    profile: Optional[str] = None,
    limit: Optional[int] = None,
) -> Generator[str, None, None]:
    """Find files with specific extensions either locally or on blobstore."""

    if not is_s3_path(uri):
        raise ValueError(
            f"Ingestion source must be an S3-compatible path (s3://...). Got: {uri}"
        )

    nb_total = 0
    s3_client = get_s3_client(profile)
    bucket_name, prefix = extract_s3_bucket_and_key(uri)
    continuation_token = None
    while True:
        list_kwargs = {"Bucket": bucket_name, "Prefix": prefix}
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**list_kwargs)
        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]
                if any(key.endswith(extension) for extension in extensions):
                    yield f"s3://{bucket_name}/{key}"
                    nb_total += 1
                    if limit and nb_total == limit:
                        return

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break
    return


def batch_iterable(iterable: Iterable, batch_size: int) -> Generator[List, None, None]:
    """
    Batches an iterable into smaller lists of a given batch size.

    Args:
        iterable (Iterable): The iterable to batch.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List, None, None]: A generator that yields batches of the iterable.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def get_presigned_url(uri: str, client: Optional[BaseClient] = None) -> str:
    assert is_s3_path(uri), f"URI {uri} is not a s3 path"
    if client is None:
        raise ValueError(f"S3 client is not provided, but path {uri} is on s3.")
    bucket_name, key = extract_s3_bucket_and_key(uri)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": key},
        ExpiresIn=3600,
    )


def encode_file_to_base64(file_bytes: bytes) -> str:
    """Encode file bytes to base64 encoded string for JSON.

    Returns:
        - base64 encoded string
    """
    return base64.b64encode(file_bytes).decode("utf-8")


@ray.remote
class DataFetcher:
    def __init__(
        self,
        queue: Queue,
        directory_path: str,
        extensions: List[str],
        s3_profile: str,
        batch_size: int,
        limit: Optional[int] = None,
        nb_consumer: int = 1,
    ) -> None:
        self.directory_path = directory_path
        self.extensions = extensions
        self.s3_profile = s3_profile
        self.batch_size = batch_size
        self.limit = limit
        self.queue = queue
        self.nb_consumer = nb_consumer

    def fetch_data(self) -> None:
        files_to_process = list_files(
            self.directory_path,
            extensions=self.extensions,
            profile=self.s3_profile,
            limit=self.limit,
        )
        for batch in batch_iterable(files_to_process, self.batch_size):
            self.queue.put(batch)
        for _ in range(self.nb_consumer):
            self.queue.put(None)  # Sentinels to indicate completion


@ray.remote
class FileBatchProcessor:
    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        endpoint_url: str,
        collection_id: str,
        metadata: Dict[str, Any] = {},
        base_path: Optional[str] = None,
        s3_profile: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
        existence_check: str = "must",
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.endpoint_url = endpoint_url
        self.collection_id = collection_id
        self.metadata = metadata
        self.base_path = base_path
        self.s3_profile = s3_profile
        self.timeout = timeout
        self.token = token
        self.existence_check = existence_check

        self.client = get_s3_client(s3_profile)
        self.session = get_requests_session()
        self.url = f"{endpoint_url}/v1/collections/{collection_id}/documents?existence_check_mode={existence_check}"
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.mime = magic.Magic(mime=True)

    def process_data(self) -> None:
        while True:
            batch = self.input_queue.get(block=True, timeout=self.timeout)
            if batch is None:
                self.output_queue.put(None)
                break
            to_process = self.process_batch(batch)
            self.output_queue.put(to_process)

    def process_batch(self, files: List[str]) -> Tuple[Any, List[str]]:
        """Process file batch."""
        batch = []
        for f in files:
            payload: Dict[str, Any] = {"id": sha256(f.encode("utf-8")).hexdigest()}
            if not is_s3_path(f):
                raise ValueError(
                    f"Ingestion only supports S3-compatible storage. Invalid path: {f}"
                )
            payload["url"] = get_presigned_url(f, self.client)
            mime_type, _ = mimetypes.guess_type(f)
            if not mime_type:
                logger.warning(
                    f"Could not infer mime type of {f} file name, using `application/octet-stream`"
                )
            payload["mime_type"] = mime_type or "application/octet-stream"

            key = f
            if self.base_path:
                if not f.startswith(self.base_path):
                    ValueError(f"{self.base_path} could not be extracted from file {f}")
                key = f[len(self.base_path) :]
                if key.startswith("/"):
                    key = key[1:]
            metadata = self.metadata.get(key, {})
            if not metadata and self.metadata:
                raise KeyError(
                    f"No metadata could be found for {key} in metadata dict."
                )
            payload["metadata"] = {"filename": key, **metadata}
            batch.append(payload)
        logging.info(f"{len(batch)=}, batch[0]: {batch[0]}")
        response = self.session.post(
            self.url, json=batch, headers=self.headers, timeout=self.timeout
        )
        if response.status_code != 200:
            logger.error(f"Got status code {response.status_code}, {response.json()}")
        return response, files


@ray.remote
class EmbeddingParquetProcessor:
    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        endpoint_url: str,
        collection_id: str,
        id_cols: Optional[List[str]] = None,
        embeddings_col: str = "embeddings",
        metadata_cols: Optional[List[str]] = None,
        fillna: bool = True,
        s3_profile: Optional[str] = None,
        timeout: Optional[int] = None,
        token: Optional[str] = None,
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.endpoint_url = endpoint_url
        self.collection_id = collection_id
        self.id_cols = id_cols
        self.embeddings_col = embeddings_col
        self.metadata_cols = metadata_cols
        self.fillna = fillna
        self.s3_profile = s3_profile
        self.timeout = timeout
        self.token = token

        self.client = get_s3_client(s3_profile)
        self.session = get_requests_session()
        self.url = f"{endpoint_url}/v1/insert-data"
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.mime = magic.Magic(mime=True)

    def process_data(self) -> None:
        while True:
            batch = self.input_queue.get(block=True, timeout=self.timeout)
            if batch is None:
                self.output_queue.put(None)
                break
            to_process = self.process_batch(batch)
            self.output_queue.put(to_process)

    def process_batch(self, files: List[str]) -> Tuple[Any, List[str]]:
        """Process parquet batch using bulk insert endpoint."""
        # Get AWS credentials from client config
        session = boto3.Session(profile_name=self.s3_profile) if self.s3_profile else boto3.Session()
        credentials = session.get_credentials()
        
        if not credentials:
            raise ValueError("AWS credentials not found. Please configure AWS credentials or use --s3-profile.")
        
        if not self.collection_id:
            raise ValueError("Collection ID is required for parquet ingestion.")

        # Get endpoint_url using the same logic as get_s3_client()
        endpoint_url = None
        if self.s3_profile:
            config = configparser.ConfigParser()
            config.read(os.path.expanduser("~/.aws/config"))
            section_name = f"profile {self.s3_profile}" if self.s3_profile != "default" else "default"
            endpoint_url = config.get(section_name, "endpoint_url", fallback=None)
        else:
            endpoint_url = os.environ.get("AWS_ENDPOINT_URL", None)
        
        payload: Dict[str, Any] = {
            "collection_name": self.collection_id,
            "parquet_paths": files,
            "access_key": credentials.access_key,
            "secret_key": credentials.secret_key,
            "endpoint_url": endpoint_url  # Will be None for default AWS S3, or custom for LocalStack/MinIO
        }
        logging.info(f"{self.collection_id=}")
        logging.info(f"{payload=}")
        logging.info(f"{credentials.access_key}")
        logging.info(f"{credentials.secret_key}")
        logging.info(f"{endpoint_url=}")
        response = self.session.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
            verify=False,  # Accept self-signed TLS
        )
        if response.status_code not in [200, 202]:
            logger.error(
                f"Got status code {response.status_code}, {response.json()}"
            )
        return response, files


class CSVLogger:
    """Utility to write response logs to CSV file."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.fieldnames = ["file", "status"]
        with open(file_path, mode="w", newline="\n") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, files: List[str], status: int) -> None:
        with open(self.file_path, mode="a", newline="\n") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            for f in files:
                writer.writerow({"file": f, "status": status})
