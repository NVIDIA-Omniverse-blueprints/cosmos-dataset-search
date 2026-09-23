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

"""Haystack rankers."""


from typing import List, Literal, Optional

from haystack import Document, component, logging

from src.haystack.serializer import SerializerMixin

logger = logging.getLogger(__name__)


@component
class ScoreRanker(SerializerMixin):
    """
    Ranks Documents based on the value of their score attribute.

    The ranking can be performed in descending order or ascending order.

    Usage example:
    ```
    from haystack import Document
    from src.haystack.components.rankers import ScoreRanker

    ranker = ScoreRanker()
    docs = [
        Document(content="Paris", score=1.3),
        Document(content="Berlin", score=0.7),
        Document(content="Barcelona", score=2.1}),
    ]

    output = ranker.run(documents=docs)
    docs = output["documents"]
    assert docs[0].content == "Barcelona"
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        dedup: bool = True,
        sort_order: Literal["ascending", "descending"] = "descending",
    ):
        """
        Creates an instance of ScoreRanker.

        :param top_k:
            The maximum number of Documents to return per query.
            If not provided, the Ranker returns all documents it receives in the new ranking order.
        :param dedup:
            Whether or not to dedup documents that have the same ID. Default True.
        :param sort_order:
            Whether to sort the score field by ascending or descending order.
            Possible values are `descending` (default) and `ascending`.
        """

        self.top_k = top_k
        self.dedup = dedup
        self.sort_order = sort_order
        self._validate_params(
            top_k=self.top_k,
            sort_order=self.sort_order,
        )

    def _validate_params(
        self,
        top_k: Optional[int],
        sort_order: Literal["ascending", "descending"],
    ):
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0, but got %s" % top_k)

        if sort_order not in ["ascending", "descending"]:
            raise ValueError(
                "The value of parameter <sort_order> must be 'ascending' or 'descending', "
                "but is currently set to '%s'.\n"
                "Change the <sort_order> value to 'ascending' or 'descending' when initializing the "
                "ScoreFieldRanker." % sort_order
            )

    @component.output_types(documents=List[Document])
    def run(
        self,
        documents: List[Document],
        top_k: Optional[int] = None,
        dedup: bool = True,
        sort_order: Optional[Literal["ascending", "descending"]] = None,
    ):
        """
        Ranks a list of Documents based on the score by:
        1. Sorting the Documents by the score field in descending or ascending order.
        2. Dedup based on document ID.
        2. Returning the top-k documents.

        :param documents:
            Documents to be ranked.
        :param top_k:
            The maximum number of Documents to return per query.
            If not provided, the top_k provided at initialization time is used.
        :param dedup:
            Whether or not to dedup documents that have the same ID. Default True.
        :param sort_order:
            Whether to sort the score field by ascending or descending order.
            Possible values are `descending` (default) and `ascending`.
            If not provided, the sort_order provided at initialization time is used.
        :returns:
            A dictionary with the following keys:
            - `documents`: List of Documents sorted by the specified score field.

        :raises ValueError:
            If `top_k` is not > 0.
            If `sort_order` is not 'ascending' or 'descending'.
        """
        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        sort_order = sort_order or self.sort_order
        dedup = dedup or self.dedup
        self._validate_params(
            top_k=top_k,
            sort_order=sort_order,
        )

        parsed_score = self._parse_score(documents)
        tuple_parsed_score_and_docs = list(zip(parsed_score, documents))

        # Sort the documents by doc.score
        reverse = sort_order == "descending"
        try:
            tuple_sorted_by_score = sorted(
                tuple_parsed_score_and_docs, key=lambda x: x[0], reverse=reverse
            )
        except TypeError as error:
            # Return original documents if mixed types that are not comparable are returned (e.g. int and list)
            logger.warning(
                "Tried to sort Documents with IDs {document_ids}, but got TypeError with the message: {error}\n"
                "Returning the <top_k> of the original Documents since score ranking is not possible.",
                document_ids=",".join([doc.id for doc in documents]),
                error=error,
            )
            return {"documents": documents[:top_k]}

        sorted_documents = [doc for _, doc in tuple_sorted_by_score]

        # dedup based on ID
        deduped = sorted_documents
        if dedup:
            seen, deduped = set(), []
            for doc in sorted_documents:
                if doc.id in seen:
                    continue
                seen.add(doc.id)
                deduped.append(doc)

        return {"documents": deduped[:top_k]}

    def _parse_score(self, documents: List[Document]) -> List[float]:
        """Parse score."""
        return [doc.score for doc in documents]
