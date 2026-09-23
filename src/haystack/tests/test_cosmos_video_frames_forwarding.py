# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for forwarding CE1 video frame bundles."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from haystack import Document
from src.haystack.components.video import cosmos_video_embedder
from src.haystack.components.video.cosmos_video_embedder import (
    CosmosEmbedClient,
    CosmosEmbedInputError,
)


def _video_frames_data_uri() -> str:
    frames = ",".join(["dGVzdA=="] * 8)
    return f"data:video_frames/jpeg;base64,{{{frames}}}"


def test_document_embedder_accepts_video_frames_data_uri() -> None:
    document = Document(id="frames", content=_video_frames_data_uri(), meta={})

    assert cosmos_video_embedder.CosmosVideoDocumentEmbedder.input_checks(document)


def test_document_embedder_forwards_video_frames_data_uri_unchanged() -> None:
    content = _video_frames_data_uri()
    document = Document(id="frames", content=content, meta={})
    embedder = cosmos_video_embedder.CosmosVideoDocumentEmbedder(url="http://test:8000")

    with patch.object(
        embedder._client,
        "embed_videos",
        return_value=[[0.1, 0.2, 0.3]],
    ) as mock_embed:
        result = embedder.run([document])

    mock_embed.assert_called_once_with([content])
    assert result["documents"][0].embedding == [0.1, 0.2, 0.3]


@patch("requests.post")
def test_cosmos_embed_client_http_422_preserves_detail(mock_post: MagicMock) -> None:
    """CE1 input errors retain their status and actionable detail."""

    mock_response = MagicMock()
    mock_response.status_code = 422
    mock_response.json.return_value = {
        "error": {
            "detail": "video_frames input must contain exactly 8 frames",
            "status_code": 422,
        }
    }
    mock_error = requests.exceptions.HTTPError()
    mock_error.response = mock_response
    mock_response.raise_for_status.side_effect = mock_error
    mock_post.return_value = mock_response

    client = CosmosEmbedClient("http://test:8000")

    with pytest.raises(CosmosEmbedInputError) as raised:
        client.embed_videos(["data:video_frames/png;base64,{dGVzdA==}"])

    assert raised.value.status_code == 422
    assert str(raised.value) == "video_frames input must contain exactly 8 frames"
