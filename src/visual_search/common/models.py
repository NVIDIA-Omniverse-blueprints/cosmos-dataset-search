# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import enum
import os
import re
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from typing import Annotated, Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID
from uuid import uuid4 as uuid_factory

from fastapi import HTTPException, Path
from pydantic import BaseModel, Field, HttpUrl, constr, root_validator, validator
from typing_extensions import TypeAlias

from src.haystack.components.milvus.schema_utils import MetadataConfig
from src.visual_search.exceptions import InputValidationError

# backward compatibility with PyDantic 1.x
JsonValue: TypeAlias = Union[
    int, float, str, bool, None, List[Dict[str, Any]], Dict[str, Any]
]

SUPPORTED_CAMERAS: List[str] = [
    "camera_front_wide_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_left_fisheye_200fov",
    "camera_right_fisheye_200fov",
    "camera_front_fisheye_200fov",
    "camera_rear_fisheye_200fov",
    "camera_rear_tele_30fov",
    "camera_front_tele_30fov",
    "episode_based",  # no camera needed for episode pipeline
]


class MaglevConfig:
    def __init__(self):
        env = os.getenv("ENV", "dev")
        self.host = "maglev.nvda.ai"
        self.env = env
        if env == "cn-prod":
            self.host = "maglev.cn.nvda.ai"

        self.authn_host = f"{self.host}/authn/v1"

    def get_maglev_workflow_link(self, workflow_name: str):
        return f"https://{self.host}/ide/workflows/workflow/{workflow_name}"


maglev_config = MaglevConfig()


# Function to create a Path with id restrictions for Collection and Document ids.
def make_path_with_id_restrictions(description: str) -> Any:
    return Path(
        pattern=r"^[a-zA-Z0-9_-]+$",
        max_length=100,
        description=description,
    )


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    detail: Optional[Union[dict, str]] = {}
    type: str


# Collection data model -
#  id - identifier for the collection
#  pipeline - name of the pipeline associated with the collection
#
# To achieve the desired API experience - only pipeline required on input,
# id always provided on output.
#
class CollectionBase(BaseModel):
    pipeline: str = Field(description="The pipeline of the collection.")
    name: str = Field(description="A name for the collection. Intended for people.")
    tags: Optional[dict] = Field(
        description="The key-value pair tags associated with the collection.",
        default=None,
    )
    init_params: Optional[dict] = Field(
        description="The init params for collection.",
        default=None,
    )

    cameras: Optional[str] = Field(
        default="camera_front_wide_120fov",
        description="Comma seperated list of camera sensors to index.",
    )

    @validator("cameras", pre=True, always=True)
    def validate_cameras(cls: Any, v: Any) -> Any:
        return validate_cameras_list(v)


# The Collection type used as input to the API
class CollectionCreate(CollectionBase):
    """
    Create a new collection with the given name and pipeline. for example

    {
      "pipeline": "patched_image_search_milvus",
      "name": "custom_param_collection",
      "collection_config": {
        "num_partitions": 10,
        "num_shards": 2,
        "properties": {
          "mmap.enabled": true
        }
      },
      "index_config": {
        "index_type": "IVF_SQ8",
        "params": {
          "params": {
            "nlist": 2048
          },
          "mmap.enabled": true
        }
      },
      "metadata_config": {
        "allow_dynamic_schema": True,
        "fields": [
            {
                "name": "session_id",
                "dtype": "VARCHAR",
                "max_length": 36,
                "is_partition_key": True,
            },
        ]
      }
    }
    """

    collection_config: Dict = {}
    index_config: Dict = {}
    metadata_config: MetadataConfig = MetadataConfig()


# The Collection type used as a confirmation of deletion
class CollectionDelete(CollectionBase):
    pass


# The Collection type returned from the API
class Collection(CollectionBase):
    """A collection of documents."""

    id: str = Field(description="The ID of the collection.")
    created_at: datetime = Field(
        description="The date and time the collection was created.",
    )

    @validator("id", pre=True)
    def coerce_uuid_to_str(cls: Any, value: Any) -> Any:
        if isinstance(value, UUID):
            return str(value)
        return value


# The Collection type used for name updates
class CollectionPatch(BaseModel):
    """A request to update the name of a collection."""

    name: Optional[str] = Field(
        description="A new name for the collection. Intended for people.", default=None
    )
    tags: Optional[dict] = Field(
        description="The key-value pair tags associated with the collection.",
        default=None,
    )

    @root_validator(skip_on_failure=True)
    def check_patch(cls: Any, v: Any) -> Any:
        name, tags = v.get("name"), v.get("tags")
        if name is None and tags is None:
            raise ValueError("At least one of name or tags must be provided")
        return v


# Envelope for GET /collections response
class CollectionListResponse(BaseModel):
    """A response containing a list of collections."""

    collections: list[Collection] = Field(description="The list of collections.")


# Envelope response for -
#  POST /collections
#  GET /collections/{collection_id}
#  PATCH /collections/{collection_id}
class CollectionResponse(BaseModel):
    """A response containing a collection."""

    collection: Collection = Field(description="The collection.")


class CollectionInfoResponse(CollectionResponse):
    """A response containing collection info."""

    total_documents_count: int = Field(
        description="The total number of documents in the collection."
    )


class CollectionInfoListResponse(BaseModel):
    """A response containing list of collections and its info."""

    collections: list[CollectionInfoResponse] = Field(
        description="List of Collections with info."
    )


#
# Document data model -
#  id - identifier for the document
#  content - the text content of the document
#  metadata - a dictionary of metadata about the document
#
# To achieve the desired API experience - only content and metadata required on input,
# id and indexed_at always provided on output.
#


class MimeType(Enum):
    """A list of acceptable format types for uploading files."""

    TEXT = "text/plain"
    PDF = "application/pdf"
    JPEG = "image/jpeg"
    PNG = "image/png"
    MP4 = "video/mp4"
    ZIP = "episode/zip"
    OTHER = "application/octet-stream"


class DocumentUploadBase(BaseModel):
    """A document to be indexed for retrieval."""

    id: Optional[str] = Field(
        description="The identifier to use for the document.", default=None
    )
    metadata: Optional[
        Dict[str, Annotated[str, constr(max_length=1024)] | int | float | bool]
    ] = Field(
        description="A dictionary of metadata about the document.",
        default=None,
    )

    @validator("metadata")
    @classmethod
    def check_metadata_length(cls: Any, value: Any) -> Any:
        max_entries = 42
        if value and len(value) > max_entries:
            raise InputValidationError(
                f"metadata may not have more than {max_entries} entries"
            )
        return value


class ExistenceCheckMode(Enum):
    # check existence and delete existing documents if exist, do not proceed if anything goes wrong
    MUST_CHECK = "must"
    # check existence and delete existing documents if exist with a timeout, best effort
    CHECK_WITH_TIMEOUT = "with_timeout"
    # check existence and ignore existing documents
    IGNORE_EXISTING = "ignore_existing"
    # do not check existence at all, just go ahead with insert,
    # this might result in duplicated records in the collection,
    # but might be preferred mode with large scale upload.
    SKIP = "skip"


class DocumentUploadJson(DocumentUploadBase):
    content: str = Field(
        description="The content of the document. If this is a file, "
        "it must be base64 encoded. For plain text, the maximum."
        "size is 5 MiB. For files, the maximum size is 50 MiB."
        "For larger files, it is recommended to use `DocumentUploadUrl`."
    )
    mime_type: MimeType = Field(description="The mime type of the content")

    @validator("content")
    def empty_content(cls, content: str, values: Dict[str, Any]) -> str:
        if not content:
            raise InputValidationError("Content is empty!")
        return content

    @validator("content")
    def check_content(cls, content: str, values: Dict[str, Any]) -> str:
        mime_type = values.get("mime_type")
        if mime_type == MimeType.TEXT and len(content) > 5 * 1024 * 1024:
            raise InputValidationError("Plain text content must be under 5 MiB.")
        elif len(content) > 50 * 1024 * 1024:
            raise InputValidationError("Content must be under 50 MiB.")
        return content


class DocumentUploadEmbedding(DocumentUploadBase):
    embedding: List[float] = Field(description="Precomputed embeddings for document")
    mime_type: Optional[MimeType] = MimeType.OTHER

    @validator("embedding")
    def empty_embedding(cls, embedding: str, values: Dict[str, Any]) -> str:
        if not embedding:
            raise InputValidationError("Embedding is empty!")
        return embedding


class DocumentUploadUrl(DocumentUploadBase):
    url: str = Field(description="URL to use for fetching binary content.")
    mime_type: MimeType = Field(description="The mime type of the content")

    @validator("url")
    def check_url(cls, url: str, values: Dict[str, Any]) -> str:
        if not url:
            raise InputValidationError("URL is empty!")
        url_pattern = r"^(http|https|ftp|ftps)://[^\s/$.?#].[^\s]*$"
        if not re.match(url_pattern, url):
            raise InputValidationError(f"URL {url} not a valid URL.")
        return url


class Document(DocumentUploadBase):
    id: str = Field(description="The ID of the document.")
    indexed_at: datetime = Field(
        description="The date and time the document was indexed."
    )
    content: str = Field(description="The content of the document.")
    mime_type: str = Field(description="Format of content.")
    metadata: Optional[
        Dict[str, Annotated[str, constr(max_length=1024)] | int | float | bool]
    ] = Field(description="A dictionary of metadata about the document.", default=None)


# Envelope response for POST /collections/{collection_id}/documents
class DocumentResponse(BaseModel):
    """A response containing a document."""

    documents: list[Document] = Field(description="List of created documents.")


class BulkEmbeddingsIngestResponse(BaseModel):
    """A response for bulk ingested embeddings."""

    ids: List[str] = Field(description="List of ingested uuids")


class BulkEmbeddingsIngestRequest(BaseModel):
    file_url: str = Field(
        description="The Parquet file containing embeddings and metadata"
    )
    embeddings_col: str = Field(
        description="Parquet column containing embedding (as 1D numpy array)."
    )
    id_cols: Optional[List[str]] = Field(
        default=None,
        description="Optional list of columns in parquet file to use for unique row hash. Default is None and will use auto-generate a uuid4.",
    )
    metadata_cols: Optional[List[str]] = Field(
        default=None,
        description="Optional list of columns in parquet file to use for metadata. If not specified, will use all columns except that for embeddings.",
    )
    fillna: bool = Field(
        default=True,
        description="Whether to fill NaN parquet values for metadata columns.",
    )


# Retrieval data model
# Multi modal queries from text, videos or embeddings can be used


class TextQuery(BaseModel):
    """Text input query."""

    text: str = Field(description="The actual text to be tokenised and embedded.")

    @validator("text")
    def text_not_empty(cls: "TextQuery", v: str) -> str:
        if not v:
            raise InputValidationError("`text` field is an empty string!")
        return v


class VideoQuery(BaseModel):
    """Video input query."""

    video: str = Field(
        description="Video input as either: 1) base64 data URI (data:video/<format>;base64,<data>), "
        "2) presigned URL data URI (data:video/<format>;presigned_url,<url>), or 3) raw presigned URL string."
    )

    @validator("video")
    def video_not_empty(cls: "VideoQuery", v: str) -> str:
        if not v:
            raise InputValidationError("`video` field is an empty string!")
        return v


class EpisodeQuery(BaseModel):
    """Episode input query."""

    episode: str = Field(
        description="Episode zip b64 bytes as a utf-8 string to be parsed and embedded."
    )

    @validator("episode")
    def episode_not_empty(cls: "EpisodeQuery", v: str) -> str:
        if not v:
            raise InputValidationError("`episode` field is an empty string!")
        return v


class SessionSegmentDetails(BaseModel):
    """Session Segment Details."""

    session_id: str = Field(description="Session id of the clip.")
    start_timestamp: str = Field(description="Starting timestamp of the clip.")
    end_timestamp: str = Field(description="Ending timestamp of the clip.")
    camera: str = Field(description="Camera sensor file to use.")

    @validator("session_id")
    def session_id_not_empty(cls: "SessionSegmentQuery", v: str) -> str:
        if not v:
            raise InputValidationError("`session_id` field is an empty string!")
        return v

    @validator("start_timestamp")
    def start_timestamp_not_empty(cls: "SessionSegmentQuery", v: str) -> str:
        if not v:
            raise InputValidationError("`start_timestamp` field is an empty string!")
        return v

    @validator("end_timestamp")
    def end_timestamp_not_empty(cls: "SessionSegmentQuery", v: str) -> str:
        if not v:
            raise InputValidationError("`end_timestamp` field is an empty string!")
        return v

    @validator("camera")
    def camera_not_empty(cls: "SessionSegmentQuery", v: str) -> str:
        if not v:
            raise InputValidationError("`camera` field is an empty string!")
        return v


class SessionSegmentQuery(BaseModel):
    """Clip/session segment input query."""

    session_segment: SessionSegmentDetails = Field(
        description="Session Segment details to search."
    )


class SessionFrameDetails(BaseModel):
    """Session Frame Details."""

    session_id: str = Field(description="Session id of the frame.")
    timestamp: str = Field(description="Timestamp of the frame.")
    camera: str = Field(description="Camera sensor file to use.")

    @validator("session_id")
    def session_id_not_empty(cls: "SessionFrameQuery", v: str) -> str:
        if not v:
            raise InputValidationError("`session_id` field is an empty string!")
        return v

    @validator("timestamp")
    def timestamp_not_empty(cls: "SessionFrameQuery", v: str) -> str:
        if not v:
            raise InputValidationError("`timestamp` field is an empty string!")
        return v

    @validator("camera")
    def camera_not_empty(cls: "SessionFrameQuery", v: str) -> str:
        if not v:
            raise InputValidationError("`camera` field is an empty string!")
        return v


class SessionFrameQuery(BaseModel):
    """Session Frame input query."""

    session_frame: SessionFrameDetails = Field(
        description="Session frame details to search."
    )


class EmbeddingQuery(BaseModel):
    """Embedding input query."""

    embedding: Tuple[float, ...] = Field(
        description="Vector to be used as directly. Will bypass embedding step on server-side."
    )

    @validator("embedding")
    def embedding_size(
        cls: "EmbeddingQuery", v: Tuple[float, ...]
    ) -> Tuple[float, ...]:
        if not v:
            raise InputValidationError("`embedding` field is an empty tuple!")
        return v


QueryType = Union[
    TextQuery,
    VideoQuery,
    EpisodeQuery,
    SessionSegmentQuery,
    SessionFrameQuery,
    EmbeddingQuery,
    List[
        Union[
            TextQuery,
            VideoQuery,
            EpisodeQuery,
            SessionSegmentQuery,
            SessionFrameQuery,
            EmbeddingQuery,
        ]
    ],
]


class SearchRequest(BaseModel):
    """A request to search for documents in a collection."""

    query: QueryType = Field(
        description="Either a `TextQuery` or `VideoQuery` or `EmbeddingQuery` or a batch of these."
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=16_000,
        description="The number of results to return. Max 16,000.",
    )
    reconstruct: bool = Field(
        default=False, description="Whether to return the document embedding."
    )
    search_params: Dict[str, Any] = Field(
        default={},
        description='Index specific search params, for example {"nprobe": 32} for IVF indices',
    )
    filters: Union[Dict[str, Any], str] = Field(
        default={},
        description="Extra filters, see https://milvus.io/docs/boolean.md for EBNF syntax examples",  # noqa
    )
    generate_asset_url: Optional[bool] = Field(
        default=True, description="Whether to generate asset URLs."
    )
    clf: Dict[str, Any] = Field(
        default={},
        description="Additional linear classifier to filter retrieved results.",
    )

    @validator("filters")
    def validate_filters(cls: Any, v: Any) -> Any:
        return validate_search_filter(v)


class TopKSearch(BaseModel):
    """Search for nearest neighbors to query."""

    nb_neighbors: int = Field(description="Number of nearest neighbors.")
    search_params: Dict[str, Any] = Field(
        default={},
        description='Index specific search params, for example {"nprobe": 32} for IVF indices',
    )
    filters: Union[Dict[str, Any], str] = Field(
        default={},
        description="Extra filters, see https://milvus.io/docs/boolean.md or https://docs.haystack.deepset.ai/docs/metadata-filtering for supported EBNF syntax examples",  # noqa
    )
    reconstruct: bool = Field(
        default=False, description="Whether to return the document embedding."
    )

    @validator("filters")
    def validate_filters(cls: Any, v: Any) -> Any:
        return validate_search_filter(v)

    class Config:
        frozen = True
        json_schema_extra = {
            "example": {
                "nb_neighbors": 8,
                "search_params": {"range_filter": 0.0},
                "filters": {},
                "reconstruct": False,
            }
        }


class RadiusSearch(BaseModel):
    """Retrieve data above minimum similarity metric to queries."""

    min_similarity: float = Field(
        description="Minimum similarity of nearest neighbors. N.B. cosine similarity is between -1 and +1, where +1 is most similar."
    )
    search_params: Dict[str, Any] = Field(
        default={},
        description='Index specific search params, for example {"nprobe": 32} for IVF indices',
    )
    filters: Dict[str, Any] = Field(
        default={},
        description="Extra filters, see https://milvus.io/docs/boolean.md or https://docs.haystack.deepset.ai/docs/metadata-filtering for supported EBNF syntax examples",  # noqa
    )
    reconstruct: bool = Field(
        default=False, description="Whether to return the document embedding."
    )

    @validator("filters")
    def validate_filters(cls: Any, v: Any) -> Any:
        return validate_search_filter(v)

    class Config:
        frozen = True


class RetrievalQuery(BaseModel):
    """Query data model for retrieval engine."""

    collections: Tuple[str, ...] = Field(
        description="Unique names of collections to be queried."
    )
    query: QueryType = Field(
        description="Either a `TextQuery` or `VideoQuery` or `EmbeddingQuery` or a batch of these."
    )
    params: Union[TopKSearch, RadiusSearch] = Field(
        description="Either `TopKSearch` or `RadiusSearch`."
    )
    payload_keys: Optional[Tuple[str, ...]] = Field(
        default=(), description="Additional payload data requested."
    )
    generate_asset_url: Optional[bool] = Field(
        default=True, description="Whether to generate asset URLs."
    )
    rerank: Optional[bool] = Field(
        default=True,
        description="Whether to rerank results based on the pipeline.",
    )

    class Config:
        frozen = True


class RetrievedDocument(DocumentUploadBase):
    collection_id: str = Field(description="The ID of the collection.")
    asset_url: Optional[str] = Field(
        description="The URL of the asset associated with the document."
    )
    score: Optional[float] = Field(
        default=None, description="A score for this retrieved document."
    )
    content: str = Field(description="The content of the retrieved document")
    id: str = Field(description="The ID of the source document.")
    mime_type: str = Field(description="The original format of the source document")
    metadata: Optional[
        Dict[str, Annotated[str, constr(max_length=1024)] | int | float | bool]
    ] = Field(description="The metadata of the source document.", default=None)
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Optional reconstructed embedding vector.",
    )

    def __lt__(self, other):
        if self.score is None:
            return True
        if other.score is None:
            return False
        return self.score < other.score


# Envelope for POST /search response
class SearchResponse(BaseModel):
    """A response containing a list of search results."""

    retrievals: list[RetrievedDocument] = Field(
        description="List of retrieved docs returned from the search."
    )


def validate_uuid4(value: str) -> str:
    try:
        UUID(value, version=4)
        return value
    except ValueError:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=(f"Invalid UUID4: {value}"),
        )


def validate_cameras_list(value: str) -> str:
    if value == "include_fe_if_rwd":
        return value
    cameras = value.split(",")
    for camera in cameras:
        if camera not in SUPPORTED_CAMERAS:
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid camera entry: {camera}, Supported cameras {SUPPORTED_CAMERAS}"
                ),
            )
    return value


def validate_search_filter(value: Any) -> Union[Dict[str, Any], str]:
    if not value:
        return value
    required_field = os.getenv("REQUIRED_FILTER_FIELD")
    if not required_field:
        return value
    if isinstance(value, str):
        # for string field, for example raw filter like `f_1 == "v_1" or f_2 == "v_2"`,
        # not easy to check them correctly, for now only checking if required_field is in the string,
        # which is not accurate.
        if "required_field" not in value:
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail=f"Invalid filter, must include field {required_field}: {value}",
            )
        return value
    else:
        if not is_field_included(value, required_field):
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail=f"Invalid filter, must include field {required_field}: {value}",
            )
        return value


def is_field_included(filters: Dict[str, Any], required_field: str) -> bool:
    """
    Checks if certain field is included, useful to validators trying to ensure that
    certain field has to be included for performance reasons.
    """
    if "conditions" in filters:
        operator = filters["operator"]
        conditions = filters["conditions"]
        if operator == "NOT":
            return False
        elif operator == "AND":
            return any(
                [
                    is_field_included(condition, required_field)
                    for condition in conditions
                ]
            )
        elif operator == "OR":
            return all(
                [
                    is_field_included(condition, required_field)
                    for condition in conditions
                ]
            )
        else:
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail=f"Unknown operator for condition clause: '{operator}'",
            )
    else:
        field_name = filters["field"].lower()
        if field_name.startswith("meta."):
            field_name = field_name[5:]
        return required_field.lower() == field_name and filters["operator"].lower() in [
            "==",
            "in",
        ]


# Define the Status Enum
class BackfillStatus(str, Enum):
    PENDING = "PENDING"
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    FINISHED = "FINISHED"
    DELETED = "DELETED"
    LAUNCH_FAILED = "LAUNCH_FAILED"


class BackfillQueueBase(BaseModel):
    id: str = Field(default_factory=uuid_factory)
    name: str = Field(description="Name of the backfill request.")
    user_email: str = Field(description="User Email who requested backfill.")
    collection_id: str = Field(description="Collection ID of the embedding.")
    pipeline: str = Field(description="The pipeline of the collection.")
    status: str = Field(default=BackfillStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class BackfillQueue(BackfillQueueBase):
    """Non-persistent model for backfill queue."""

    session_ids: List[str] = Field(
        description="Sessions List",
        default=None,
    )
    cameras: str = Field(description="Comma-separated Cameras.")
    workflow_name: str = Field(description="Maglev workflow name.")
    details: str = Field(description="Any additional details.", default="")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workflow_name = f"av-irs-backfill-{maglev_config.env}-{self.id}"


class BackfillQueueRequest(BaseModel):
    """A request to launch backfill indexing workflows."""

    name: str = Field(description="Name of the backfill request.")
    sessions: Set[str] = Field(description="List of sessions to backfill index for.")
    collection_id: str = Field(
        description="The id of the milvus collection to upload data to."
    )

    @validator("sessions")
    def validate_sessions_list(cls: Any, v: Any) -> Any:
        if not v:
            raise InputValidationError("Sessions list cannot be empty.")
        return v

    @validator("sessions", each_item=True)
    def validate_list_of_uuids(cls: Any, v: Any) -> Any:
        return validate_uuid4(v)

    @validator("collection_id", pre=True, always=True)
    def validate_individual_uuids(cls: Any, v: Any) -> Any:
        if v is not None:
            return validate_uuid4(v)
        return v


class BackfillQueueResponse(BackfillQueueBase):
    workflow_link: str = Field(description="Link to the maglev workflow.", default="")


# Envelope for GET /backfills/{id} response
# Should derive from BackfillQueue, but session_ids field misbehaves in that case.
class BackfillQueueDetailsResponse(BackfillQueueBase):
    session_ids: List[str] = Field(
        description="Sessions List",
        default=None,
    )
    cameras: str = Field(description="Comma-separated Cameras.")
    workflow_name: str = Field(description="Maglev workflow name.")
    details: str = Field(description="Any additional details.", default="")

    queue: int = Field(description="Number in the queue to launch workflow")
    workflow_link: str = Field(description="Link to the maglev workflow.", default="")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workflow_name = f"av-irs-backfill-{maglev_config.env}-{self.id}"


# Envelope for GET /backfills response
class BackfillQueueListResponse(BaseModel):
    backfills: list[BackfillQueueResponse] = Field(description="The list of backfills.")


class ItemList(BaseModel):
    items: list


class Pipeline(BaseModel):
    id: str = Field(description="The name of the pipeline.")
    enabled: bool = Field(description="Whether or not the pipeline is enabled.")
    missing: List[str] = Field(
        description="Missing environmental variables needed to enable the pipeline.",
    )

    # lowkey don't know if we need to show config to the user???
    # Is it giving the user too much unnecessary information?
    config: JsonValue = Field(
        description="The JSON components and configuration used to set up the pipeline"
    )


class PipelineMode(Enum):
    INDEX = "index"
    QUERY = "query"


class PipelinesResponse(BaseModel):
    """A list of all of the pipelines for the microservice."""

    pipelines: list[Pipeline] = Field(description="List of pipelines.")


class DeleteResponse(BaseModel):
    message: str = Field(default="Resource deleted successfully.")
    id: Optional[str] = Field(
        default=None, description="The ID of the deleted resource."
    )
    deleted_at: Optional[datetime] = Field(
        default=None, description="The timestamp when the resource was deleted."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Resource deleted successfully.",
                "id": "12345",
                "deleted_at": "2023-04-01T14:30:00Z",
            }
        }


class LabelledDocuments(BaseModel):
    """Labelled documents."""

    collection_name: str = Field(description="Name of collection")
    labelled_documents: Dict[str, bool] = Field(
        description="Labelled document IDs in collection, where "
        "`True` is a good retrieval, `False` is a bad retrieval"
    )

    class Config:
        frozen = True


class SearchRefinementMode(Enum):
    LINEAR_PROBE = "linear_probe"
    LINEAR_CLASSIFIER = "linear_classifier"


class SearchRefinementRequest(BaseModel):
    model_type: SearchRefinementMode = Field(
        description="""Choose a model to train for search refinement.""",
        default=SearchRefinementMode.LINEAR_PROBE,
    )
    grounding_queries: List[
        TextQuery | VideoQuery | EpisodeQuery | EmbeddingQuery
    ] = Field(
        description="Search refinement grounding queries.",
    )
    labels: List[LabelledDocuments] = Field(
        description="""List of selected labelled data."""
    )

    regularization_strength: float = Field(
        default=0.05,
        description="""How close the learned probe embedding should be to the anchor. Only valid for
                    linear probe.""",
    )

    @validator("labels")
    def empty_labels(
        cls: "SearchRefinementRequest", v: List[LabelledDocuments]
    ) -> List[LabelledDocuments]:
        if not v:
            raise InputValidationError("Empty labels!")
        return v

    @validator("regularization_strength")
    def negative_regularization_strength(
        cls: "SearchRefinementRequest", v: float
    ) -> float:
        if v < 0:
            raise InputValidationError("regularization_strength must be >= 0.")
        return v

    @validator("labels")
    def non_unique_collections(
        cls: "SearchRefinementRequest", v: List[LabelledDocuments]
    ) -> List[LabelledDocuments]:
        unique_names = set(labelled_docs.collection_name for labelled_docs in v)
        if len(unique_names) != len(v):
            raise InputValidationError("Non-unique collection names in `labels`!")
        return v


class LinearProbeRequest(BaseModel):
    grounding_queries: List[
        TextQuery | VideoQuery | EpisodeQuery | EmbeddingQuery
    ] = Field(
        description="Linear probe grounding queries.",
    )
    labels: List[LabelledDocuments] = Field(
        description="""List of selected labelled data."""
    )

    regularization_strength: float = Field(
        default=0.05,
        description="""How close the learned probe embedding should be to the anchor.""",
    )

    @validator("labels")
    def empty_labels(
        cls: "LinearProbeRequest", v: List[LabelledDocuments]
    ) -> List[LabelledDocuments]:
        if not v:
            raise InputValidationError("Empty labels!")
        return v

    @validator("regularization_strength")
    def negative_regularization_strength(cls: "LinearProbeRequest", v: float) -> float:
        if v < 0:
            raise InputValidationError("regularization_strength must be >= 0.")
        return v

    @validator("labels")
    def non_unique_collections(
        cls: "LinearProbeRequest", v: List[LabelledDocuments]
    ) -> List[LabelledDocuments]:
        unique_names = set(labelled_docs.collection_name for labelled_docs in v)
        if len(unique_names) != len(v):
            raise InputValidationError("Non-unique collection names in `labels`!")
        return v


class LinearProbeResponse(BaseModel):
    queries: List[EmbeddingQuery] = Field(
        description="Learnt embedding queries from linear probe training",
    )


class LinearClassifierBase(BaseModel):
    coef: List[List[float]] = Field(
        description="Coefs of a learned linear classifier",
    )
    intercept: List[float] = Field(
        description="Intercept/bias term of a learned linear classifier.",
    )


class LinearClassifierResponse(BaseModel):
    weights: LinearClassifierBase = Field(description="Coefficients and intercept")
    model: str = Field(
        default="",
        description="Deprecated. Classifier filtering uses weights instead of serialized models."
    )


class IngestRequest(BaseModel):
    """Ingest request for bulk ingestion."""

    collection_id: str = Field(description="The ID of the collection to ingest into.")
    filename: str = Field(description="The name of the file to ingest.")


class IngestStatusEnum(enum.Enum):
    UPLOADING = "UPLOADING"
    UPLOADED = "UPLOADED"
    INGESTING = "INGESTING"
    PROCESSING = "PROCESSING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
    UNKNOWN = "UNKNOWN"


class IngestProcessBase(BaseModel):
    pass


class StoredIngestProcess(IngestProcessBase):
    """Non-persistent ingest process model."""

    id: str = Field(default_factory=lambda: str(uuid_factory()))
    collection_id: str = Field(
        description="The collection the file is ingested into.",
    )
    filename: str = Field(description="The name of the file to ingest.")
    status: IngestStatusEnum = Field(description="The status of the ingest process.")
    job_id: str = Field(
        description="The job ID of the ingest process.",
        default=None,
    )


class EpisodeLookup(BaseModel):
    """Non-persistent embedding lookup model."""

    id: str = Field(default_factory=lambda: str(uuid_factory()))
    embedding: bytes = Field(description="Serialized numpy embedding of an episode.")
    content: str = Field(default="", description="Optional base64 encoded episode.")
    session_id: str = Field(description="Session UUID.")
    collection_id: str = Field(description="Collection ID of the embedding.")
    keystone_timestamp: int = Field(description="Episode's keystone timestamp.")


class EpisodeLookupAddRequest(BaseModel):
    """Episode lookup add request body."""

    id: Optional[str] = Field(description="Unique identifier.")
    embedding: Optional[List[float]] = Field(description="Embedding of an episode.")
    content: Optional[str] = Field(
        default="", description="Optional base64 encoded episode."
    )
    session_id: str = Field(description="Session UUID.")
    collection_id: str = Field(description="Collection ID of the embedding.")
    keystone_timestamp: int = Field(description="Episode's keystone timestamp.")


class EpisodeLookupAddResponse(BaseModel):
    message: str = Field(description="Response message.")
    inserted_rows: int = Field(description="Number of inserted rows.")
    skipped_rows: int = Field(description="Number of skipped rows.")


class EpisodeLookupGetResponse(BaseModel):
    embedding_query: EmbeddingQuery = Field(
        description="Embedding query of the closest match."
    )
    matched_timestamp: int = Field(
        description="Keystone timestamp of the closest match."
    )


# Bulk insert (v1).
class InsertDataRequest(BaseModel):
    """Request model for initiating a bulk insert job."""

    collection_name: str = Field(
        ..., description="Target Milvus collection (ID or name)."
    )
    parquet_paths: List[str] = Field(
        ...,
        description='Parquet files to import (e.g. ["s3://bucket/file.parquet"]).',
    )
    access_key: Optional[str] = Field(
        None, description="Access key for the file storage (e.g., S3 access key ID)."
    )
    secret_key: Optional[str] = Field(
        None,
        description="Secret key for the file storage (e.g., S3 secret access key).",
    )
    endpoint_url: Optional[HttpUrl] = Field(
        None,
        description="Endpoint URL for S3-compatible storage (e.g., MinIO).",
    )


class InsertDataResponse(BaseModel):
    """Response model after successfully submitting a bulk insert job."""

    status: str = Field(
        ..., example="accepted", description="Status of the job submission."
    )
    message: str = Field(
        ...,
        example="Bulk data insertion job accepted and started.",
        description="User-friendly message.",
    )
    job_id: str = Field(
        ...,
        example="12345",
        description="The unique ID assigned to the bulk insert job by Milvus.",
    )


# Insert JobStatusResponse definition before JobDetail
class JobStatusResponse(BaseModel):
    """Response model for querying the status of a specific job."""

    job_id: str = Field(
        ..., example="12345", description="The unique ID of the bulk insert job."
    )
    status: str = Field(
        ..., example="completed", description="Current status of the job."
    )
    details: str = Field(
        ..., description="Additional details or failure reason, if any."
    )
    progress: Optional[int] = Field(
        None, example=100, description="Percentage progress of the job, if available."
    )
    collection_name: Optional[str] = Field(
        None, description="The target collection name."
    )


class JobDetail(JobStatusResponse):
    """Detailed information about a single bulk insert task retrieved from Milvus."""

    row_count: Optional[int] = Field(None, description="Number of rows imported.")
    create_time_utc: Optional[int] = Field(
        None, description="Task creation timestamp (UTC milliseconds)."
    )
    last_update_time_utc: Optional[int] = Field(
        None, description="Task last update timestamp (UTC milliseconds)."
    )
    parquet_paths: List[str] = Field(
        [], description="List of parquet files associated with the task."
    )
