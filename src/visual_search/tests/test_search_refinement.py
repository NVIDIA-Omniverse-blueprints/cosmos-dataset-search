# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import pytest
from types import SimpleNamespace
from src.visual_search.v1.apis import search_refinement as sr

# A top-level class is picklable; mimics the real LinearClassifier for _serialize_model
class DummyLinearClassifier:
    pass


@pytest.mark.asyncio
async def test_train_linear_probe_happy_path(monkeypatch):
    # get_collections -> single collection on one pipeline
    monkeypatch.setattr(
        sr,
        "get_collections",
        lambda ids: [SimpleNamespace(id=ids[0], pipeline="pipe-a", name="c")],
    )

    # pipeline -> provides a query pipeline and inputs (inputs not used here but must exist)
    qp = object()
    monkeypatch.setattr(
        sr,
        "pipeline_handler",
        lambda _coll: SimpleNamespace(query_pipeline=qp, query_pipeline_inputs={}),
    )

    # Document store not Milvus: ensures branch that calls filter_documents(filters=..., return_embedding=True)
    class DS:
        def __init__(self):
            self.calls = []
        def filter_documents(self, filters=None, return_embedding=False):
            self.calls.append((filters, return_embedding))
            ids = filters["value"] if isinstance(filters, dict) else []
            return [SimpleNamespace(id=i, embedding=[0.1, 0.2]) for i in ids]

    ds = DS()
    monkeypatch.setattr(sr, "get_document_stores", lambda _qp: [ds])

    # Linear probe pipeline returns a single learned embedding
    monkeypatch.setattr(sr, "run_linear_probe_pipeline", lambda **_: [[0.5, 0.4]])

    req = SimpleNamespace(
        model_type=sr.SearchRefinementMode.LINEAR_PROBE,
        grounding_queries=[{"text": "q"}],
        labels=[SimpleNamespace(collection_name="coll-1", labelled_documents={"d1": True, "d2": False})],
        regularization_strength=0.01,
    )

    resp = await sr.train(req)
    assert hasattr(resp, "queries") and len(resp.queries) == 1
    assert ds.calls and ds.calls[-1][1] is True  # return_embedding=True used


@pytest.mark.asyncio
async def test_train_mixed_pipelines_raises_422(monkeypatch):
    def _fake_get_cols(ids):
        # Two collections on different pipelines triggers the mixed pipelines error
        return [
            SimpleNamespace(id=ids[0], pipeline="pipe-a", name="c1"),
            SimpleNamespace(id=ids[-1], pipeline="pipe-b", name="c2"),
        ]
    monkeypatch.setattr(sr, "get_collections", _fake_get_cols)

    req = SimpleNamespace(
        model_type=sr.SearchRefinementMode.LINEAR_PROBE,
        grounding_queries=[],
        labels=[
            SimpleNamespace(collection_name="c1", labelled_documents={}),
            SimpleNamespace(collection_name="c2", labelled_documents={}),
        ],
        regularization_strength=0.1,
    )

    with pytest.raises(sr.HTTPException) as exc:
        await sr.train(req)
    assert exc.value.status_code == 422
    assert "mixed pipelines" in exc.value.detail


@pytest.mark.asyncio
async def test_train_linear_classifier_happy_path(monkeypatch):
    monkeypatch.setattr(
        sr,
        "get_collections",
        lambda ids: [SimpleNamespace(id=ids[0], pipeline="pipe-a", name="c")],
    )
    qp = object()
    monkeypatch.setattr(
        sr,
        "pipeline_handler",
        lambda _coll: SimpleNamespace(query_pipeline=qp, query_pipeline_inputs={}),
    )

    class DS:
        def filter_documents(self, filters=None, return_embedding=False):
            ids = filters["value"] if isinstance(filters, dict) else []
            return [SimpleNamespace(id=i, embedding=[0.3, 0.7]) for i in ids]

    monkeypatch.setattr(sr, "get_document_stores", lambda _qp: [DS()])

    # Return a picklable classifier instance plus a weights-like dict that Pydantic can coerce
    monkeypatch.setattr(
        sr,
        "run_linear_classifier_training",
        lambda labelled_embeddings: (DummyLinearClassifier(), {"coef": [[0.11, 0.22]], "intercept": [0.33]}),
    )

    req = SimpleNamespace(
        model_type=sr.SearchRefinementMode.LINEAR_CLASSIFIER,
        grounding_queries=[{"text": "ignored"}],
        labels=[SimpleNamespace(collection_name="coll-x", labelled_documents={"a": True, "b": False})],
        regularization_strength=0.05,
    )

    resp = await sr.train(req)
    assert hasattr(resp, "weights") and resp.weights.coef == [[0.11, 0.22]] and resp.weights.intercept == [0.33]
    assert resp.model == ""  # model field is deprecated; filtering now uses weights

