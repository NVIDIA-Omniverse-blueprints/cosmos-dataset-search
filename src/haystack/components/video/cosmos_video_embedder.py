# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Haystack components for embedding videos with Cosmos-Embed NIM service.

This is the primary video embedding service
"""

import base64
import logging
from copy import deepcopy
from typing import Any, Dict, List

import requests
from haystack import Document, component

from src.haystack.serializer import SerializerMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CosmosEmbedClient:
    """Client for Cosmos-Embed NIM service."""
    
    def __init__(self, url: str = "localhost:8000"):
        """Initialize the Cosmos-Embed client."""
        self.base_url = url.rstrip('/')
        if not self.base_url.startswith('http'):
            self.base_url = f"http://{self.base_url}"
        self.embeddings_endpoint = f"{self.base_url}/v1/embeddings"
    
    def embed_videos(self, video_inputs: List[str]) -> List[List[float]]:
        """Embed videos using cosmos-embed service with base64 data."""

        logger.error(f"Embedding videos: {video_inputs=}")
        if not video_inputs:
            return []
            
        if len(video_inputs) > 64:
            raise ValueError("cosmos-embed supports maximum 64 videos per request")

        # Determine request type based on content
        has_base64 = any(";base64," in vi for vi in video_inputs)
        has_presigned = any(";presigned_url," in vi for vi in video_inputs)
        
        # Validate mixed content types
        if has_base64 and has_presigned:
            raise ValueError("Cannot mix base64 and presigned URL inputs in same request")
        
        if len(video_inputs) == 1 and has_base64:
            # Single base64 video - use query mode
            payload = {
                "input": video_inputs[0],
                "request_type": "query",
                "encoding_format": "float",
                "model": "nvidia/cosmos-embed1",
            }
        elif has_base64:
            # Multiple base64 videos - use individual query mode calls
            embeddings = []
            for video_input in video_inputs:
                if ";base64," not in video_input:
                    raise ValueError(f"Expected base64 video input, got: {video_input}")
                single_payload = {
                    "input": video_input,
                    "request_type": "query",
                    "encoding_format": "float",
                    "model": "nvidia/cosmos-embed1",
                }
                try:
                    logger.info(f"Sending request to cosmos-embed: {self.embeddings_endpoint}")
                    response = requests.post(
                        self.embeddings_endpoint, 
                        json=single_payload, 
                        timeout=300,
                        headers={'Content-Type': 'application/json'}
                    )
                    response.raise_for_status()
                    result = response.json()
                    if "error" in result:
                        error_detail = result["error"].get("detail", "Unknown error")
                        error_status = result["error"].get("status_code", 500)
                        raise RuntimeError(f"Cosmos-embed error ({error_status}): {error_detail}")
                    embeddings.append(result["data"][0]["embedding"])
                except requests.exceptions.Timeout:
                    logger.error("Request to cosmos-embed timed out")
                    raise RuntimeError("Cosmos-embed service timeout - videos may be too large or service overloaded")
                except requests.exceptions.ConnectionError:
                    logger.error("Failed to connect to cosmos-embed service")
                    raise RuntimeError("Cannot connect to cosmos-embed service - check if service is running")
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 400:
                        logger.error("Bad request to cosmos-embed - check video format")
                        raise ValueError("Invalid video format or request format")
                    elif e.response.status_code == 503:
                        logger.error("Cosmos-embed service unavailable")
                        raise RuntimeError("Cosmos-embed service temporarily unavailable")
                    else:
                        logger.error(f"HTTP error from cosmos-embed: {e}")
                        body = e.response.text
                        logger.error(
                            "Cosmos-embed HTTP %s response body: %s",
                            e.response.status_code,
                            (body[:3000] if body else "(empty)"),
                        )
                        raise RuntimeError(f"Cosmos-embed service error: {e}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to get embeddings from cosmos-embed: {e}")
                    body = e.response.text
                    logger.error(
                        "Cosmos-embed HTTP %s response body: %s",
                        e.response.status_code,
                        (body[:3000] if body else "(empty)"),
                    )
                    raise RuntimeError(f"Cosmos-embed service error: {e}")
            return embeddings
        else:
            # Presigned URLs - use bulk_video mode
            for i, video_input in enumerate(video_inputs):
                if ";presigned_url," not in video_input:
                    raise ValueError(
                        f"Video input {i} must be presigned-url formatted: data:video/mp4;presigned_url,<URL>"
                    )
            payload = {
                "input": video_inputs,
                "request_type": "bulk_video",
                "encoding_format": "float",
                "model": "nvidia/cosmos-embed1",
            }
        
        try:
            response = requests.post(
                self.embeddings_endpoint, 
                json=payload, 
                timeout=300,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Check for error in response
            if "error" in result:
                error_detail = result["error"].get("detail", "Unknown error")
                error_status = result["error"].get("status_code", 500)
                raise RuntimeError(f"Cosmos-embed error ({error_status}): {error_detail}")
            
            # Extract embeddings from response
            embeddings = []
            for data_item in result["data"]:
                embeddings.append(data_item["embedding"])
            
            return embeddings
            
        except requests.exceptions.Timeout:
            logger.error("Request to cosmos-embed timed out")
            raise RuntimeError("Cosmos-embed service timeout - videos may be too large or service overloaded")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to cosmos-embed service")
            raise RuntimeError("Cannot connect to cosmos-embed service - check if service is running")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                logger.error("Bad request to cosmos-embed - check video format")
                raise ValueError("Invalid video format or request format")
            elif e.response.status_code == 503:
                logger.error("Cosmos-embed service unavailable")
                raise RuntimeError("Cosmos-embed service temporarily unavailable")
            else:
                logger.error(f"HTTP error from cosmos-embed: {e}")
                body = e.response.text
                logger.error(
                    "Cosmos-embed HTTP %s response body: %s",
                    e.response.status_code,
                    (body[:3000] if body else "(empty)"),
                )
                raise RuntimeError(f"Cosmos-embed service error: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get embeddings from cosmos-embed: {e}")
            body = e.response.text
            logger.error(
                "Cosmos-embed HTTP %s response body: %s",
                e.response.status_code,
                (body[:3000] if body else "(empty)"),
            )
            raise RuntimeError(f"Cosmos-embed service error: {e}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using cosmos-embed service."""
        logger.error(f"Embedding texts: {texts=}")
        if not texts:
            return []
            
        if len(texts) > 64:
            raise ValueError("cosmos-embed supports maximum 64 texts per request")
        
        payload = {
            "input": texts,
            "request_type": "query",
            "encoding_format": "float", 
            "model": "nvidia/cosmos-embed1"
        }
        
        try:
            response = requests.post(
                self.embeddings_endpoint, 
                json=payload, 
                timeout=60,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Check for error in response
            if "error" in result:
                error_detail = result["error"].get("detail", "Unknown error")
                error_status = result["error"].get("status_code", 500)
                raise RuntimeError(f"Cosmos-embed error ({error_status}): {error_detail}")
            
            # Extract embeddings from response
            embeddings = []
            for data_item in result["data"]:
                embeddings.append(data_item["embedding"])
            
            return embeddings
            
        except requests.exceptions.Timeout:
            logger.error("Request to cosmos-embed timed out")
            raise RuntimeError("Cosmos-embed service timeout")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to cosmos-embed service")
            raise RuntimeError("Cannot connect to cosmos-embed service - check if service is running")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                logger.error("Bad request to cosmos-embed - check text format")
                raise ValueError("Invalid text format or request format")
            elif e.response.status_code == 503:
                logger.error("Cosmos-embed service unavailable")
                raise RuntimeError("Cosmos-embed service temporarily unavailable")
            else:
                logger.error(f"HTTP error from cosmos-embed: {e}")
                body = e.response.text
                logger.error(
                    "Cosmos-embed HTTP %s response body: %s",
                    e.response.status_code,
                    (body[:3000] if body else "(empty)"),
                )
                raise RuntimeError(f"Cosmos-embed service error: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get text embeddings from cosmos-embed: {e}")
            body = e.response.text
            logger.error(
                "Cosmos-embed HTTP %s response body: %s",
                e.response.status_code,
                (body[:3000] if body else "(empty)"),
            )
            raise RuntimeError(f"Cosmos-embed service error: {e}")


class CosmosEmbedMixin:
    """Mixin for cosmos-embed functionality."""
    
    def __init__(
        self,
        url: str = "localhost:8000",
        **kwargs
    ) -> None:
        """Initialize cosmos-embed mixin."""
        self.url = url
        self._client = CosmosEmbedClient(url=url)
    
    def _get_mime_type_extension(self, mime_type: str) -> str:
        """Get file extension from mime type."""
        mime_to_ext = {
            "video/mp4": "mp4",
            "video/avi": "avi", 
            "video/mov": "mov",
            "video/webm": "webm",
        }
        return mime_to_ext.get(mime_type.lower(), "mp4")


@component
class CosmosVideoDocumentEmbedder(SerializerMixin, CosmosEmbedMixin):
    """
    A component for embedding video documents using Cosmos-Embed NIM service.
    Processes documents in batches of up to 64 videos for optimal performance.
    """
    
    def __init__(self, batch_size: int = 64, **kwargs):
        """
        Initialize the embedder with configurable batch size.
        
        Args:
            batch_size: Maximum number of videos to process per cosmos-embed API call (max 64)
            **kwargs: Additional arguments passed to parent classes
        """
        CosmosEmbedMixin.__init__(self, **kwargs)
        if batch_size > 64:
            raise ValueError("cosmos-embed maximum batch size is 64 videos")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size

    @staticmethod
    def input_checks(doc: Document) -> bool:
        """Checks on input documents.
        Accepts either raw blob data (bytes) or a presigned URL in `content`.
        """
        has_url = isinstance(doc.content, str) and doc.content.startswith("http")
        has_b64 = isinstance(doc.content, str) and ";base64," in doc.content
        if not (has_url or has_b64):
            msg = (
                f"Document {getattr(doc, 'id', '<no-id>')} must provide a presigned URL "
                "or base64 data URI in `content`."
            )
            logger.error(msg)
            raise ValueError(msg)
        return True

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        if not documents:
            return {"documents": documents}
        
        assert all(self.input_checks(inp) for inp in documents)

        total_docs = len(documents)
        logger.info(f"Cosmos-embed processing: {total_docs} videos in batches of {self.batch_size}")
        
        # Process documents in batches to optimize cosmos-embed API usage
        all_new_documents: List[Document] = []
        total_api_calls = 0
        
        for batch_start in range(0, total_docs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_docs)
            batch_docs = documents[batch_start:batch_end]
            batch_size = len(batch_docs)
            
            logger.debug(f"Processing batch {batch_start//self.batch_size + 1}: documents {batch_start+1}-{batch_end}")
            
            # Prepare inputs for this batch presigned URLs
            video_inputs: List[str] = []
            for doc in batch_docs:
                if isinstance(doc.content, str):
                    if doc.content.startswith("data:video/"):
                        # Already a data URI (base64 or presigned_url)
                        video_inputs.append(doc.content)
                    else:
                        # Plain URL - wrap as presigned_url
                        video_inputs.append(f"data:video/mp4;presigned_url,{doc.content}")
                else:
                    raise ValueError("Document must provide URL or data URI in content field")
            
            if not video_inputs:
                logger.warning(f"No video inputs prepared for batch {batch_start//self.batch_size + 1}")
                continue
            
            print(f"{video_inputs=}")
            # Single API call for this batch (up to 64 videos)
            embeddings_batch = self._client.embed_videos(video_inputs)
            total_api_calls += 1

            if len(embeddings_batch) != len(batch_docs):
                raise RuntimeError(
                    f"Embeddings batch size {len(embeddings_batch)} does not match "
                    f"document batch size {len(batch_docs)} for batch {batch_start//self.batch_size + 1}."
                )

            # Create new documents with embeddings for this batch
            for doc, embedding in zip(batch_docs, embeddings_batch):
                source_id = doc.id
                meta = deepcopy(doc.meta)
                meta["source_id"] = source_id
                new_doc = Document(embedding=embedding, meta=meta)
                all_new_documents.append(new_doc)
        
        logger.info(
            f"Cosmos-embed completed: {total_docs} videos processed with {total_api_calls} API calls "
            f"(avg {total_docs/total_api_calls:.1f} videos/call)"
        )
        
        return {"documents": all_new_documents}


@component
class CosmosVideoEmbedder(SerializerMixin, CosmosEmbedMixin):
    """
    A component for embedding videos using Cosmos-Embed NIM service.
    Expects presigned URLs.
    """

    @component.output_types(embeddings=List[List[float]])
    def run(self, video_urls: List[str]) -> Dict[str, List[List[float]]]:
        """Embed videos using presigned URLs."""
        if not video_urls:
            return {"embeddings": []}
        formatted = []
        for url in video_urls:
            if url.startswith("data:video/"):
                formatted.append(url)
            else:
                formatted.append(f"data:video/mp4;presigned_url,{url}")
        embeddings = self._client.embed_videos(formatted)
        return {"embeddings": embeddings}


@component
class CosmosTextEmbedder(SerializerMixin, CosmosEmbedMixin):
    """
    A component for embedding text using Cosmos-Embed NIM service.
    """

    @component.output_types(embeddings=List[List[float]])
    def run(self, texts: List[str]) -> Dict[str, List[List[float]]]:
        """Embed texts."""
        embeddings = self._client.embed_texts(texts)
        return {"embeddings": embeddings}


@component
class CosmosSessionSegmentEmbedder(SerializerMixin, CosmosEmbedMixin):
    """
    A component for embedding session segments using Cosmos-Embed NIM service.
    """

    @component.output_types(embeddings=List[List[float]])
    def run(self, clips: List[Dict[str, str]]) -> Dict[str, List[List[float]]]:  # Accept dicts from QueryTypeRouter
        """Embed session clips (video segments).
        Note: This component expects clips to be base64-encoded video data strings.
        """
        import json
        video_inputs = []
        for clip in clips:
            # Accept both dict (preferred) and already-formatted base64 strings for backward compatibility
            if isinstance(clip, dict):
                try:
                    # Temporary strategy: turn the JSON representation into a dummy "video" payload.
                    # In production this should be replaced by real clip extraction.
                    clip_json = json.dumps(clip, separators=(",", ":"))
                    b64_payload = base64.b64encode(clip_json.encode()).decode("utf-8")
                    video_inputs.append(f"data:video/mp4;base64,{b64_payload}")
                except Exception as exc:
                    logger.warning("Skipping clip dict %s due to error: %s", clip, exc)
            elif isinstance(clip, str) and clip.startswith("data:video/") and ";base64," in clip:
                video_inputs.append(clip)
            else:
                logger.warning("Skipping clip without valid data: %s", clip)
        
        if not video_inputs:
            return {"embeddings": []}
        
        embeddings = self._client.embed_videos(video_inputs)
        return {"embeddings": embeddings}