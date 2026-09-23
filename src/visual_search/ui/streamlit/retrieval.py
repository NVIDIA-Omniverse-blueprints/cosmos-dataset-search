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

"""Simple streamlit app for video retrieval using visual search endpoints."""

import base64
import io
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers

from src.visual_search.common.models import (
    Collection,
    CollectionInfoResponse,
    CollectionListResponse,
    EmbeddingQuery,
    LabelledDocuments,
    LinearProbeRequest,
    LinearProbeResponse,
    SearchRequest,
    SearchResponse,
    TextQuery,
    VideoQuery,
)

T = TypeVar("T")

STORAGE_SECRETS_TAG: Final = "storage-secrets"
STORAGE_PREFIX_TAG: Final = "storage-prefix"
STORAGE_TEMPLATE_TAG: Final = "storage-template"

# for visualizing files in tar archives
TAR_BYTE_OFFSET_KEY: Final = "byte_offset"
TAR_BYTE_SIZE_KEY: Final = "byte_size"


def get_this_session_info() -> str:
    """Get session info identifier.

    It should remain constant as long as one interacts with the application.
    If one refreshes or closes the tab, it will generate a new id.
    """
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid4().hex
    return st.session_state["session_id"]


def get_auth_api_key() -> Optional[str]:
    """Gets API key."""
    return os.getenv("HTTP_AUTH_API_KEY")


def get_auth_token_endpoint() -> Optional[str]:
    """Gets auth token endpoint."""
    return os.getenv("HTTP_AUTH_TOKEN_ENDPOINT")


def get_media_cdn() -> Optional[str]:
    """Gets s3 cdn endpoint."""
    return os.getenv("VIUS_CDN_ENDPOINT")


@lru_cache(maxsize=32)
def get_auth_token(session_id: str) -> Optional[str]:
    """Gets auth token. Cached by session ID."""
    token_endpoint = get_auth_token_endpoint()
    api_key = get_auth_api_key()
    if not token_endpoint or not api_key:
        return None
    response = requests.get(
        token_endpoint, headers={"Authorization": f"ApiKey {api_key}"}
    )
    jsonified = response.json()
    if "token" not in jsonified:
        st.write(jsonified)
        return None
    return jsonified["token"]


def is_url(maybe_url: str) -> bool:
    """Check if string is a valid URL."""
    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(regex, maybe_url) is not None


@st.cache_data(max_entries=int(os.getenv("MAX_CACHED_ENTRIES", 1000)))
def read_s3_object(
    s3_url: str,
    storage_secrets: Optional[str] = None,
    byte_offset: Optional[int] = None,
    byte_size: Optional[int] = None,
) -> io.BytesIO:
    """Read s3 object stream."""

    parsed_url = urlparse(s3_url)
    if parsed_url.scheme != "s3":
        raise ValueError("URL must be an S3 URL (s3://bucket/key)")

    url = f"{get_media_cdn()}/bytes?s3_url={s3_url}"

    if storage_secrets is not None:
        url += f"&storage_secrets={storage_secrets}"

    if byte_offset is not None and byte_size is not None:
        url += f"&start_bytes={byte_offset}&end_bytes={byte_offset + byte_size}"

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        byte_data = io.BytesIO()
        for chunk in response.iter_content(chunk_size=2048):
            if chunk:
                byte_data.write(chunk)
        byte_data.seek(0)
    return byte_data


@st.cache_data(max_entries=int(os.getenv("MAX_CACHED_ENTRIES", 1000)))
def read_url(url: str, session_id: str) -> io.BytesIO:
    """Download URL content."""
    auth_token = get_auth_token(session_id)
    headers = (
        {"Authorization": "Bearer " + auth_token} if auth_token is not None else {}
    )
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = io.BytesIO(response.content)
    return data


@st.cache_data
def get_indexes(endpoint: str, session_id: str, cookies: dict) -> List[Collection]:
    """Get index list from microservice."""
    response = requests.get(f"{endpoint}/collections", cookies=cookies)
    response.raise_for_status()
    return CollectionListResponse(**json.loads(response.text)).collections


@st.cache_data
def get_index_info(
    endpoint: str, index_names: List[str], session_id: str, cookies: dict
) -> Dict[str, CollectionInfoResponse]:
    """Get index info from microservice."""
    info = {}
    for name in index_names:
        response = requests.get(f"{endpoint}/collections/{name}", cookies=cookies)
        response.raise_for_status()
        info[name] = CollectionInfoResponse(**json.loads(response.text))
    return info


@st.cache_data
def get_retrievals(
    endpoint: str, index_name: str, query: Dict[str, Any], cookies: dict
) -> SearchResponse:
    """Gets retrievals from microservice."""
    response = requests.post(
        f"{endpoint}/collections/{index_name}/search", json=query, cookies=cookies
    )
    response.raise_for_status()
    retrievals = SearchResponse(**json.loads(response.text))
    return retrievals


@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> Any:
    """Converts pandas dataframe into CSV text file."""
    return df.to_csv(index=True).encode("utf-8")


def get_dataframe(
    retrievals: SearchResponse,
) -> pd.DataFrame:
    """Convert retrievals into dataframe."""
    rows: List[Dict[str, Any]] = []
    ids = []
    for retrieval in retrievals.retrievals:
        row = {**(retrieval.metadata if retrieval.metadata else {})}
        row["score"] = retrieval.score if retrieval.score else np.nan
        ids.append(retrieval.id)
        rows.append(row)
    return pd.DataFrame(rows, index=ids)


def parse_uploads(
    texts: Iterable[str],
    videos: Iterable[Any],
    embeddings: Iterable[Any],
) -> Generator[Union[TextQuery, VideoQuery], None, None]:
    for text in texts:
        yield TextQuery(text=text)
    for video in videos:
        yield VideoQuery(video=base64.b64encode(video.read()).decode("utf-8"))
    for embedding in embeddings:
        with io.BytesIO(embedding.read()) as fp:
            data = np.load(fp)
            if data.ndim == 1:
                yield data
            elif data.ndim == 2:
                for row in data:
                    yield data
            else:
                raise ValueError(
                    f"Embeddings numpy file has invalid shape {data.shape}"
                )


def construct_query(
    query: Tuple[Union[TextQuery, VideoQuery], ...],
    top_k: int,
    reconstruct: bool = False,
    filters: Optional[str] = None,
) -> Optional[SearchRequest]:
    """Construct suitable query."""

    retrieval_query: Optional[SearchRequest] = None
    if query:
        return SearchRequest(
            query=query,
            top_k=top_k,
            reconstruct=reconstruct,
            filters=filters,
        )
    return retrieval_query


def train_linear_probe(
    endpoint: str,
    index_name: str,
    labelled_documents: Dict[str, bool],
    grounding_queries: Tuple[Union[TextQuery, EmbeddingQuery, VideoQuery], ...],
) -> None:
    """Train linear probe callback and caches result."""

    st.session_state["last_anchor"] = grounding_queries
    req = LinearProbeRequest(
        grounding_queries=list(grounding_queries),
        labels=[
            LabelledDocuments(
                collection_name=index_name,
                labelled_documents=labelled_documents,
            )
        ],
    )
    json_inp = req.dict()
    response = requests.post(
        f"{endpoint}/linear_probe",
        json=json_inp,
    )
    response.raise_for_status()
    probe_result = LinearProbeResponse(**json.loads(response.text))
    st.session_state.probe_result = probe_result
    st.session_state["modality_idx"] = 4


def get_cookies():
    headers = _get_websocket_headers()
    return headers
    if headers and "Cookie" in headers:
        cookie_string = headers["Cookie"]
        cookies = dict(item.split("=") for item in cookie_string.split("; "))
        return cookies
    return {}


def draw_page(
    retrievals,
    nb_retrievals: int,
    items_per_page: int,
    labelled_documents: Dict[str, bool],
    uri_template: str,
    session_id: str,
    storage_secrets: Optional[str],
    width: Optional[int],
) -> None:
    """Callback for displaying results page."""
    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, nb_retrievals)

    for i in range(start_idx, end_idx):
        metadata = retrievals.retrievals[i].metadata
        score = retrievals.retrievals[i].score
        idx = retrievals.retrievals[i].id

        file_uri = uri_template
        for key, value in metadata.items():
            file_uri = file_uri.replace(f"{{{{{key}}}}}", value)
        caption = f"score={score:.3f} {file_uri}"

        # linear probe feedback inputs
        left, right = st.columns(2)
        with left:
            good = st.checkbox("Good retrieval", key=f"good/{idx}")
            if good:
                labelled_documents[idx] = True
        with right:
            bad = st.checkbox("Bad retrieval", key=f"bad/{idx}")
            if bad:
                labelled_documents[idx] = False

        byte_size, byte_offset = None, None
        if file_uri.endswith(".tar"):
            byte_size = int(metadata.get(TAR_BYTE_SIZE_KEY))
            byte_offset = int(metadata.get(TAR_BYTE_OFFSET_KEY))

        file: Optional[str | io.BytesIO] = None
        try:
            if file_uri.startswith("s3://"):
                file = read_s3_object(
                    file_uri,
                    storage_secrets,
                    byte_offset=byte_offset,
                    byte_size=byte_size,
                )
            elif is_url(file_uri):
                file = read_url(file_uri, session_id)
            else:
                if not Path(file_uri).exists():
                    raise FileNotFoundError(f"Could not read file {file_uri}")
                file = file_uri
        except Exception as e:
            st.write(f"Could not fetch {file_uri} ({e})")
            st.write(caption)

        if file:
            try:
                st.image(
                    file,
                    use_column_width=True if not width else False,
                    width=width,
                    caption=caption,
                )
            except Exception:
                st.video(file, muted=True)
                st.write(caption)


def set_page(page: int) -> None:
    """Callback to set page number for displaying results."""
    st.session_state.current_page = page


def app(media_key: str, endpoint: str) -> None:
    """Runs streamlit application.

    Args:
        media_key: default URI template.
        endpoint: retrieval API endpoint to send requests to.
    """

    st.set_page_config(page_title="Video retrieval", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 400px;
            margin-left: -400px;
        }

        """,
        unsafe_allow_html=True,
    )
    session_id = get_this_session_info()
    cookies = get_cookies()

    if "probe_result" not in st.session_state:
        st.session_state.probe_result = None

    with st.sidebar:
        st.title("Visual search")
        indices = get_indexes(endpoint, session_id, cookies)
        index_map = {index.id: index for index in indices}
        with st.expander(
            f"Choose a retrieval index to search (found {len(index_map)})",
            expanded=True,
        ):
            index_id = st.selectbox(
                "Retrieval index",
                index_map.keys(),
                format_func=lambda opt: index_map[opt].name,
            )
            index_ids = [index_id] if index_id else []

    if not index_ids:
        return

    tags = index_map[index_id].tags if index_map[index_id].tags is not None else {}
    index_uri_prefix = tags.get(STORAGE_PREFIX_TAG, "")
    index_uri_template = tags.get(STORAGE_TEMPLATE_TAG, "")
    storage_secrets = tags.get(STORAGE_SECRETS_TAG, None)

    with st.sidebar:
        nb_vecs = 0
        index_infos = get_index_info(endpoint, index_ids, session_id, cookies)
        for index_info in index_infos.values():
            nb_vecs += index_info.total_documents_count
        st.write(f"Searching {nb_vecs} vectors")
        with st.expander("Index details", expanded=False):
            st.json(index_map[index_id].model_dump())

        st.subheader("Query")
        st.write("Search via text or video input!")

        if "modality_idx" not in st.session_state:
            st.session_state["modality_idx"] = 0
        if st.session_state.probe_result is not None:
            modalities = ["text", "video", "embedding", "linear probe"]
        else:
            modalities = ["text", "video", "embedding"]
        query_type = st.selectbox(
            "Modality", modalities, index=st.session_state["modality_idx"]
        )

        learnt_query = None
        uploaded_videos, text_inputs, uploaded_embeddings = (
            [],
            [],
            [],
        )
        if query_type == "text":
            txt_input = st.text_input("Text query")
            if txt_input:
                text_inputs.append(txt_input)
        elif query_type == "video":
            uploaded_videos = st.file_uploader(
                "Video query", type=["mp4"], accept_multiple_files=True
            )
        elif query_type == "embedding":
            uploaded_embeddings = st.file_uploader(
                "Embedding query",
                type=["npy"],
                accept_multiple_files=True,
            )
        elif query_type == "linear probe":
            if "probe_result" in st.session_state:
                learnt_query = st.session_state["probe_result"].queries
                with io.BytesIO() as buffer:
                    queries_np = np.array(
                        [q.embedding for q in st.session_state["probe_result"].queries]
                    )
                    np.save(buffer, queries_np)
                    st.download_button(
                        label="Download linear probe embedding",
                        data=buffer,
                        file_name="linear_probe_embedding.npy",
                    )
        else:
            raise ValueError(
                "Type of query can only be `text` or `video` or `embedding`!"
            )
        if learnt_query:
            query = learnt_query
        else:
            query = tuple(
                parse_uploads(
                    text_inputs, uploaded_videos, uploaded_embeddings
                )
            )

        # Search parameters
        with st.expander("Search parameters", expanded=True):
            search_mode = st.selectbox("Mode", ["nearest"], index=0)
            if search_mode == "nearest":
                nb_neighbors = st.number_input(
                    "Number of neighbors",
                    min_value=1,
                    max_value=10000,
                    value=8,
                )
                filters = st.text_input(
                    "Pre-filters",
                    help="EBNF grammar (see https://milvus.io/docs/boolean.md)",
                )
                reconstruct = st.checkbox("Reconstruct", value=False)

    start = time.time()
    retrieval_query = construct_query(
        query,
        top_k=nb_neighbors,
        reconstruct=reconstruct,
        filters=filters,
    )
    retrievals = (
        get_retrievals(endpoint, index_id, retrieval_query.dict(), cookies)
        if retrieval_query is not None
        else None
    )

    period = time.time() - start
    nb_retrievals = len(retrievals.retrievals) if retrievals is not None else 0

    with st.sidebar:
        with st.expander("Customize display options", expanded=True):
            show_media = st.checkbox("Display media", value=True)
            if not index_uri_template:
                index_uri_template = index_uri_prefix + media_key
            uri_template = st.text_input(
                "URI template",
                value=index_uri_template,
            )
            if show_media:
                items_per_page = st.slider(
                    "Items per page",
                    min_value=1,
                    max_value=32,
                    value=8,
                )
            width = st.number_input(
                "Width (pixels)", min_value=100, value=None, step=1, format="%d"
            )
        if retrievals is not None:
            st.write(f"Retrieved {nb_retrievals} entries in {period:.3f} seconds.")

    # # Display results
    st.subheader("Retrieval results")
    media_tab, dataframe_tab = st.tabs(["Media", "Dataframe"])

    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    labelled_documents: Dict[str, bool] = {}
    with media_tab:
        if retrievals is not None and show_media:
            assert retrieval_query is not None
            total_pages = (nb_retrievals + items_per_page - 1) // items_per_page

            draw_page(
                retrievals=retrievals,
                nb_retrievals=nb_retrievals,
                items_per_page=items_per_page,
                labelled_documents=labelled_documents,
                uri_template=uri_template,
                session_id=session_id,
                storage_secrets=storage_secrets,
                width=width,
            )

            st.number_input(
                "Page number",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.current_page,
                on_change=lambda: set_page(st.session_state.page_value),
                key="page_value",
            )

    with st.sidebar:
        nb_positive = len([v for v in labelled_documents.values() if v is True])
        nb_negative = len([v for v in labelled_documents.values() if v is False])
        disable_training = not index_id or nb_positive < 2 or nb_negative < 2
        st.button(
            label="Train linear probe",
            key="probe_btn",
            on_click=train_linear_probe,
            disabled=disable_training,
            kwargs={
                "endpoint": endpoint,
                "index_name": index_id,
                "labelled_documents": labelled_documents,
                "grounding_queries": query,
            },
            help="Activates if at least 2 `Good` and 2 `Bad` retrievals are selected.",
        )

    if retrievals is not None:
        with dataframe_tab:
            df = get_dataframe(retrievals)
            st.dataframe(df, use_container_width=True)
            csv = convert_df_to_csv(df)
            st.download_button(
                "Download as CSV",
                csv,
                "search_results.csv",
                "text/csv",
                key="download-csv",
            )


def cli() -> None:
    """Streamlit app entrypoint.

    Args:
        endpoint: url of retrieval microservice.
    """
    endpoint = os.environ.get("ENDPOINT_URL", "http://0.0.0.0:8888/v1")
    uri_template = os.environ.get("ASSET_URI_TEMPLATE", "{{filename}}")
    if get_auth_token_endpoint() and get_auth_api_key():
        # likely AV deployment, switch default uri template
        uri_template = "https://cdn-prod.nvda.ai/v0/frame/{{session_id}}/{{camera_id}}/{{frame_id}}?height=480&width=720"
    app(uri_template, endpoint)


# Run the app
if __name__ == "__main__":
    cli()
