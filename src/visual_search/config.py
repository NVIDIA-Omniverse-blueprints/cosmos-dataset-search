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

import os
from typing import ClassVar, Dict, List

from pydantic import BaseModel


class Settings(BaseModel):
    debug: bool = False
    cors_allowed_domains: List[str] = [
        "http://localhost:8080",
        "https://localhost:8080",
        "http://localhost:3000",
        "https://ui.dev.vius.cv.nvidia.com",
        "https://ui.staging.vius.cv.nvidia.com",
        "https://ui.prod.vius.cv.nvidia.com",
        "https://ui.vius.cv.nvidia.com",
        "*",  # Waabi # TODO - this is a security hole. For cvds_blueprint, we need to modify our ingress to inject localhost:8080 as the client header.
    ]
    openapi_tags: ClassVar[List[Dict[str, str]]] = [
        {"name": "Collections", "description": "Operations related to collections."},
        {
            "name": "Document Indexing",
            "description": "Operations related to documents.",
        },
        {"name": "Health", "description": "Operations related to health."},
        {
            "name": "Retrieval",
            "description": "Operations related to document retrieval.",
        },
    ]
    if os.getenv("EXPOSE_BACKFILL_ENDPOINT"):
        openapi_tags.append(
            {
                "name": "Backfill",
                "description": "Operations related to backfilling sessions.",
            }
        )


settings = Settings()
