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

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import FastAPI

from src.visual_search.common.models import StoredIngestProcess  # noqa: F401
from src.visual_search.common.pipelines import load_pipelines


@asynccontextmanager
async def app_dependencies(app: FastAPI) -> AsyncIterator[None]:
    # The elastic_transport client throws and logs errors about not
    # being able to connect, this is expected during the load phase
    # so we temporarily silence this logger.
    l1 = logging.getLogger("elastic_transport.transport").getEffectiveLevel()
    l2 = logging.getLogger("elastic_transport.node_pool").getEffectiveLevel()

    logging.getLogger("elastic_transport.transport").setLevel(logging.CRITICAL)
    logging.getLogger("elastic_transport.node_pool").setLevel(logging.CRITICAL)

    # Load the pipelines
    await load_pipelines()

    logging.getLogger("elastic_transport.transport").setLevel(l1)
    logging.getLogger("elastic_transport.node_pool").setLevel(l2)

    # This line changes the default number of threads in the AnyIO thread pool.
    # Increasing it allows for better concurrency in this heavy IO service while
    # using syncronous endpoint handlers
    RunVar("_default_thread_limiter").set(CapacityLimiter(100))  # type: ignore
    yield
