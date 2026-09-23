# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard s3fs range reads at the actual HTTP connection, not just URI parsing."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
from contextlib import contextmanager
from contextvars import ContextVar

import s3fs
from aiobotocore.httpsession import AIOHTTPSession
from aiohttp.abc import AbstractResolver

from .remote_fetch import (
    FetchPolicy,
    FetchPolicyError,
    _origins,
    _parse_url,
    validate_s3_endpoint,
    validate_s3_paths,
)

_request_url: ContextVar[str] = ContextVar("cds_s3_request_url")


def _validate_s3_url(url: str, policy: FetchPolicy):
    parsed, origin = policy.validate_url(url)
    approved = set(_origins("CDS_FETCH_S3_ENDPOINTS"))
    for name in ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL"):
        if os.getenv(name):
            approved.add(_parse_url(os.environ[name])[1])
    # Path-style requests avoid using a caller-controlled bucket as authority.
    aws_s3 = (
        origin[0] == "https"
        and origin[2] == 443
        and re.fullmatch(
            r"s3(?:[.-](?:dualstack\.)?[a-z0-9-]+)?\.amazonaws\.com(?:\.cn)?", origin[1]
        )
    )
    if not aws_s3 and origin not in approved:
        raise FetchPolicyError(
            "S3 request destination is not an approved storage endpoint"
        )
    return parsed, origin


class _Resolver(AbstractResolver):
    async def resolve(self, host, port=0, family=socket.AF_INET):
        url = _request_url.get()
        policy = FetchPolicy.from_env()
        _, origin = _validate_s3_url(url, policy)
        if (host.lower().rstrip("."), port) != origin[1:]:
            raise FetchPolicyError(
                "S3 resolver destination differs from the validated request"
            )
        addresses = await asyncio.to_thread(policy.resolve, url)
        return [
            {
                "hostname": host,
                "host": address[4][0],
                "port": address[4][1],
                "family": address[0],
                "proto": address[2],
                "flags": socket.AI_NUMERICHOST,
            }
            for address in addresses
            if family == socket.AF_UNSPEC or address[0] == family
        ]

    async def close(self):
        pass


class _NoRedirectSession:
    def __init__(self, session):
        self._session = session

    def request(self, *args, **kwargs):
        # aiohttp redirects run below AIOHTTPSession.send. Let botocore handle
        # S3 regional redirects so every retry passes our send/resolver guards.
        kwargs["allow_redirects"] = False
        return self._session.request(*args, **kwargs)


class _GuardedS3Session(AIOHTTPSession):
    async def _get_session(self, proxy_url):
        # Version-sensitive aiobotocore hook: covered by transport regression.
        return _NoRedirectSession(await super()._get_session(proxy_url))

    async def send(self, request):
        policy = FetchPolicy.from_env()
        _, origin = _validate_s3_url(request.url, policy)
        try:
            ipaddress.ip_address(origin[1])
        except ValueError:
            pass
        else:
            # aiohttp skips its resolver for numeric hosts.
            policy.resolve(request.url)
        token = _request_url.set(request.url)
        try:
            return await super().send(request)
        finally:
            _request_url.reset(token)


@contextmanager
def open_s3_import(path: str, storage_options: dict):
    validate_s3_paths([path])
    # Options are built by our API, never accepted as an arbitrary request dict.
    opts = dict(storage_options)
    client_kwargs = dict(opts.get("client_kwargs", {}))
    endpoint = (
        client_kwargs.get("endpoint_url")
        or os.getenv("AWS_ENDPOINT_URL_S3")
        or os.getenv("AWS_ENDPOINT_URL")
    )
    validate_s3_endpoint(endpoint)
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    opts["client_kwargs"] = client_kwargs
    fs = s3fs.S3FileSystem(
        **opts,
        config_kwargs={
            "http_session_cls": _GuardedS3Session,
            "connector_args": {"resolver": _Resolver(), "use_dns_cache": False},
            "proxies": {},
            "s3": {"addressing_style": "path"},
        },
        skip_instance_cache=True,
    )
    # s3fs registers its own session finalizer. Do not close that session twice;
    # the filesystem is uncached and released with the closed stream.
    with fs.open(path, "rb") as stream:
        yield stream
