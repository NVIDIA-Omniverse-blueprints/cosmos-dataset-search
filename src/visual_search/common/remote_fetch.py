# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outbound policy for caller-supplied imports, independent of API authentication.

Public HTTPS works by default. Operators can restrict it to exact origins and
explicitly permit private object stores. Request fields never extend this policy.
Connections dial the checked DNS results, not a second resolution of the name.
"""

from __future__ import annotations

import http.client
import ipaddress
import os
import re
import socket
import ssl
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit


class FetchPolicyError(ValueError):
    """The request is outside the operator's outbound policy."""


class FetchError(OSError):
    """A permitted remote resource could not be read."""


def validate_s3_paths(paths: list[str]) -> list[str]:
    """Validate the entire batch before fsspec, DNS or schema reads run."""
    if not paths:
        raise FetchPolicyError("At least one S3 Parquet path is required")
    keys = []
    for path in paths:
        if (
            not isinstance(path, str)
            or not path.startswith("s3://")
            or "::" in path
            or "\\" in path
            or any(ord(c) < 32 or ord(c) == 127 for c in path)
        ):
            raise FetchPolicyError(
                "Parquet imports require plain s3://bucket/key paths"
            )
        try:
            parsed = urlsplit(path)
        except ValueError as error:
            raise FetchPolicyError(
                "Parquet imports require plain s3://bucket/key paths"
            ) from error
        if (
            parsed.scheme != "s3"
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", parsed.netloc)
            or ".." in parsed.netloc
            or not parsed.path.lstrip("/")
            or parsed.query
            or parsed.fragment
            or any(c in parsed.path for c in "*?[]")
            or any(part in {".", ".."} for part in parsed.path.split("/"))
        ):
            raise FetchPolicyError(
                "Parquet imports require plain s3://bucket/key paths without globbing"
            )
        keys.append(parsed.path[1:])
    return keys


def validate_s3_endpoint(endpoint_url: str | None) -> None:
    """Custom S3 endpoints must be server-configured, not supplied as authority."""
    if not endpoint_url:
        return
    parsed, origin = _parse_url(str(endpoint_url))
    if parsed.path not in {"", "/"} or parsed.query:
        raise FetchPolicyError("S3 endpoint must be an exact origin")
    configured = set(_origins("CDS_FETCH_S3_ENDPOINTS"))
    for name in ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL"):
        if os.getenv(name):
            configured.add(_parse_url(os.environ[name])[1])
    if origin not in configured:
        raise FetchPolicyError(
            "S3 endpoint is not operator-configured in CDS_FETCH_S3_ENDPOINTS"
        )


def _parse_url(url: str):
    # Reject parser ambiguities instead of silently normalizing a signed URL.
    if (
        not isinstance(url, str)
        or any(ord(c) <= 32 or ord(c) == 127 for c in url)
        or "\\" in url
    ):
        raise FetchPolicyError("Import URL contains invalid characters")
    try:
        parsed = urlsplit(url)
        port = parsed.port
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or "%" in parsed.hostname
            or port == 0
        ):
            raise ValueError
        host = parsed.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except (ValueError, UnicodeError) as error:
        raise FetchPolicyError(
            "Import URL must be an absolute HTTP(S) URL without credentials or fragments"
        ) from error
    return parsed, (
        parsed.scheme,
        host,
        port or (443 if parsed.scheme == "https" else 80),
    )


def _origins(variable: str) -> frozenset[tuple[str, str, int]]:
    result = set()
    for raw in os.getenv(variable, "").split(","):
        if not raw.strip():
            continue
        parsed, origin = _parse_url(raw.strip())
        if parsed.path not in {"", "/"} or parsed.query:
            raise FetchPolicyError(
                f"{variable} must contain exact scheme://host[:port] origins"
            )
        result.add(origin)
    return frozenset(result)


@dataclass(frozen=True)
class FetchPolicy:
    allowed_origins: frozenset[tuple[str, str, int]] = frozenset()
    private_origins: frozenset[tuple[str, str, int]] = frozenset()

    @classmethod
    def from_env(cls) -> FetchPolicy:
        return cls(
            _origins("CDS_FETCH_ALLOWED_ORIGINS"), _origins("CDS_FETCH_PRIVATE_ORIGINS")
        )

    def validate_url(self, url: str):
        parsed, origin = _parse_url(url)
        private = origin in self.private_origins
        if origin[0] != "https" and not private:
            raise FetchPolicyError(
                "Import URL must use HTTPS unless its exact origin is operator-configured"
            )
        if self.allowed_origins and origin not in self.allowed_origins and not private:
            raise FetchPolicyError(
                "Import URL origin is not in CDS_FETCH_ALLOWED_ORIGINS"
            )
        return parsed, origin

    def resolve(self, url: str) -> list:
        _, origin = self.validate_url(url)
        try:
            addresses = socket.getaddrinfo(
                origin[1], origin[2], type=socket.SOCK_STREAM
            )
        except socket.gaierror as error:
            raise FetchError("Import host could not be resolved") from error
        if not addresses:
            raise FetchError("Import host did not resolve")
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if isinstance(ip, ipaddress.IPv6Address) and (
                ip in ipaddress.ip_network("64:ff9b::/96")
                or ip in ipaddress.ip_network("64:ff9b:1::/48")
                or ip.sixtofour is not None
                or ip.teredo is not None
            ):
                raise FetchPolicyError(
                    "Import URL resolves to an IPv6 translation address"
                )
            if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
                ip = ip.ipv4_mapped
            # A private-store exception never permits metadata/link-local,
            # loopback, unspecified or multicast addresses.
            private_network = any(
                ip in network
                for network in _PRIVATE_NETWORKS
                if ip.version == network.version
            )
            if not ip.is_global and not (
                origin in self.private_origins and private_network
            ):
                raise FetchPolicyError(
                    "Import URL resolves to a forbidden private or reserved address"
                )
            if (
                ip.is_multicast
                or ip.is_unspecified
                or ip.is_loopback
                or ip.is_link_local
                or str(ip) == "fd00:ec2::254"
            ):
                raise FetchPolicyError("Import URL resolves to a forbidden address")
        return addresses


_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)


def _connect(addresses, timeout):
    last_error = None
    for family, kind, protocol, _, address in addresses:
        sock = socket.socket(family, kind, protocol)
        try:
            sock.settimeout(timeout)
            sock.connect(address)
            return sock
        except OSError as error:
            sock.close()
            last_error = error
    raise FetchError("Could not connect to import host") from last_error


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host, port, *, url, policy, timeout):
        super().__init__(host, port, timeout=timeout)
        self.url = url
        self.policy = policy

    def connect(self):
        self.sock = _connect(self.policy.resolve(self.url), self.timeout)


class _PinnedHTTPSConnection(_PinnedHTTPConnection):
    default_port = 443

    def connect(self):
        super().connect()
        try:
            self.sock = ssl.create_default_context().wrap_socket(
                self.sock, server_hostname=self.host
            )
        except BaseException:
            self.close()
            raise


@contextmanager
def open_remote(
    url: str,
    *,
    policy: FetchPolicy | None = None,
    method="GET",
    headers=None,
    timeout=30,
):
    """Open only checked destinations, including each redirect; do not use ambient proxies.

    Preserve the original path/query (including presigned signatures). TLS still
    verifies the original hostname while the socket connects to a checked IP.
    """
    policy = policy or FetchPolicy.from_env()
    for hop in range(4):
        parsed, origin = policy.validate_url(url)
        connection_type = (
            _PinnedHTTPSConnection if origin[0] == "https" else _PinnedHTTPConnection
        )
        connection = connection_type(
            origin[1], origin[2], url=url, policy=policy, timeout=timeout
        )
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        try:
            connection.request(
                method, target, headers={"Host": parsed.netloc, **(headers or {})}
            )
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location or hop == 3:
                    raise FetchError(
                        "Import redirect limit exceeded or missing destination"
                    )
                url = urljoin(url, location)
                continue
            if response.status >= 400:
                raise FetchError(f"Remote import returned HTTP {response.status}")
            yield response
            return
        except FetchPolicyError:
            raise
        except (OSError, http.client.HTTPException) as error:
            # Do not expose credentials, signed URLs, remote bodies or socket
            # diagnostics through the API or logs.
            raise FetchError("Remote import failed") from error
        finally:
            connection.close()


def download_text(url: str) -> bytes:
    max_bytes = int(os.getenv("CDS_FETCH_MAX_TEXT_BYTES", str(16 * 1024 * 1024)))
    if max_bytes <= 0:
        raise FetchPolicyError("CDS_FETCH_MAX_TEXT_BYTES must be positive")
    with open_remote(url) as response:
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise FetchPolicyError("Text import exceeds CDS_FETCH_MAX_TEXT_BYTES")
        return body
