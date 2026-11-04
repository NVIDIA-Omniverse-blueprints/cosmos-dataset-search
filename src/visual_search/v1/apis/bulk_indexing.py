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

"""Bulk parquet import endpoints.

VastDB bulk-insert has been removed. Prefer the standard document ingest API
or a VAST DataEngine vastdb-writer pipeline for large-scale loads.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()

_MSG = (
    "VastDB bulk parquet import is not available in this VastDB build. "
    "Use POST /v1/collections/{id}/documents or a VAST DataEngine ingest pipeline."
)


@router.post("/insert-data", status_code=501)
async def insert_data(_: dict | None = None):
    raise HTTPException(status_code=501, detail=_MSG)


@router.get("/jobs", status_code=501)
async def list_jobs():
    raise HTTPException(status_code=501, detail=_MSG)


@router.get("/jobs/{job_id}", status_code=501)
async def get_job(job_id: str):
    raise HTTPException(status_code=501, detail=_MSG)
