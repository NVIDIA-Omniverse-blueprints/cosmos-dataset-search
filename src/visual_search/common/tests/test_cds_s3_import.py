# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise guarded s3fs reads without a cloud account, Milvus or GPU packages.

The real resolver validates a synthetic private address. Only the socket dial is
mapped to the loopback fixture; every other outbound connection fails the test.
No S3FileSystem, guarded session, policy or open_s3_import implementation is mocked.
"""

from __future__ import annotations

import io
import ipaddress
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.awsrequest import AWSRequest
from botocore.exceptions import BotoCoreError, ClientError

from src.visual_search.common import s3_import as s3
from src.visual_search.common.remote_fetch import FetchPolicyError


PRIVATE_IP = "10.20.30.40"
OBJECT_PATH = "/test-bucket/rows.parquet"
S3_PATH = "s3://test-bucket/rows.parquet"


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    for name in (
        "CDS_FETCH_ALLOWED_ORIGINS",
        "CDS_FETCH_PRIVATE_ORIGINS",
        "CDS_FETCH_S3_ENDPOINTS",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_PROFILE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key-not-a-real-credential")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-not-a-real-credential")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    # Keep deliberate failures bounded; this is not a retry-behavior test.
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")


@pytest.fixture
def local_s3(monkeypatch):
    table = pa.table({"id": ["one", "two"], "vector": [[1.0, 2.0], [3.0, 4.0]]})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    state = SimpleNamespace(
        body=buffer.getvalue(),
        requests=[],
        connections=[],
        dns=[],
        dns_ips=[PRIVATE_IP],
        redirect=None,
        redirect_status=303,
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_HEAD(self):
            self.respond()

        def do_GET(self):
            self.respond()

        def respond(self):
            state.requests.append(
                (
                    self.command,
                    self.path,
                    self.headers.get("Range"),
                    self.headers.get("Host"),
                )
            )
            if self.command == "GET" and state.redirect:
                self.send_response(state.redirect_status)
                self.send_header("Location", state.redirect)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path != OBJECT_PATH:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            content = state.body
            byte_range = self.headers.get("Range")
            if byte_range:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", byte_range)
                assert match is not None, byte_range
                start = int(match[1])
                end = min(
                    int(match[2]) if match[2] else len(content) - 1, len(content) - 1
                )
                content = content[start : end + 1]
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {start}-{end}/{len(state.body)}"
                )
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("ETag", '"fixture-etag"')
            self.send_header("Last-Modified", "Fri, 18 Sep 2026 00:00:00 GMT")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if self.command == "GET":
                self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state.port = server.server_address[1]
    state.endpoint = f"http://storage.test:{state.port}"
    state.options = {
        "key": "test-key-not-a-real-credential",
        "secret": "test-secret-not-a-real-credential",
        "client_kwargs": {"endpoint_url": state.endpoint, "region_name": "us-east-1"},
    }
    monkeypatch.setenv("CDS_FETCH_S3_ENDPOINTS", state.endpoint)
    monkeypatch.setenv("CDS_FETCH_PRIVATE_ORIGINS", state.endpoint)
    real_connect = socket.socket.connect

    def getaddrinfo(host, port, *args, **kwargs):
        state.dns.append((host, port, s3._request_url.get(None)))
        if host == "storage.test":
            ips = state.dns_ips
        else:
            try:
                ips = [str(ipaddress.ip_address(host))]
            except ValueError:
                pytest.fail(f"Unexpected outbound DNS lookup: {host!r}")
        result = []
        for ip in ips:
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            address = (ip, port, 0, 0) if family == socket.AF_INET6 else (ip, port)
            result.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", address))
        return result

    def connect(sock, address):
        state.connections.append(address)
        if address != (PRIVATE_IP, state.port):
            pytest.fail(f"Unexpected outbound connection: {address!r}")
        return real_connect(sock, ("127.0.0.1", state.port))

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_open_s3_import_reads_and_seeks_through_real_guarded_transport(local_s3):
    with s3.open_s3_import(S3_PATH, local_s3.options) as stream:
        assert stream.read(4) == b"PAR1"
        stream.seek(-8, io.SEEK_END)
        assert stream.read() == local_s3.body[-8:]
    assert stream.closed
    assert local_s3.connections
    assert all(
        address == (PRIVATE_IP, local_s3.port) for address in local_s3.connections
    )
    assert any(method == "HEAD" for method, *_ in local_s3.requests)
    assert any(
        method == "GET" and byte_range for method, _, byte_range, _ in local_s3.requests
    )
    assert all(path == OBJECT_PATH for _, path, _, _ in local_s3.requests)
    assert all(
        host == f"storage.test:{local_s3.port}" for _, _, _, host in local_s3.requests
    )
    # ContextVar must survive s3fs's thread bridge and the resolver's to_thread.
    assert local_s3.dns
    assert all(url == local_s3.endpoint + OBJECT_PATH for _, _, url in local_s3.dns)


def test_parquet_schema_and_embedding_sample_use_real_s3_reads(local_s3):
    with s3.open_s3_import(S3_PATH, local_s3.options) as stream:
        assert pq.read_schema(stream).names == ["id", "vector"]
    with s3.open_s3_import(S3_PATH, local_s3.options) as stream:
        batch = next(
            pq.ParquetFile(stream).iter_batches(columns=["vector"], batch_size=1)
        )
        assert batch.column(0).to_pylist() == [[1.0, 2.0]]
    assert len([request for request in local_s3.requests if request[0] == "HEAD"]) == 2


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "target",
    ["http://169.254.169.254/latest/meta-data/", "https://attacker.test/secret"],
)
def test_s3_redirect_is_not_followed_by_aiohttp(local_s3, target, status):
    local_s3.redirect = target
    local_s3.redirect_status = status
    with pytest.raises((OSError, ClientError, BotoCoreError, FetchPolicyError)):
        with s3.open_s3_import(S3_PATH, local_s3.options) as stream:
            stream.read()
    assert any(request[0] == "GET" for request in local_s3.requests)
    assert all(host == "storage.test" for host, _, _ in local_s3.dns)
    assert all(
        address == (PRIVATE_IP, local_s3.port) for address in local_s3.connections
    )
    # Botocore may probe the same bucket for its region; it must not follow the
    # untrusted Location header to a different host or path.
    assert all(
        path in {OBJECT_PATH, "/test-bucket"} for _, path, _, _ in local_s3.requests
    )


@pytest.mark.parametrize("literal", ["127.0.0.1", "[::1]"])
def test_configured_literal_loopback_is_blocked_before_connection(
    monkeypatch, local_s3, literal
):
    endpoint = f"http://{literal}:{local_s3.port}"
    monkeypatch.setenv("CDS_FETCH_S3_ENDPOINTS", endpoint)
    monkeypatch.setenv("CDS_FETCH_PRIVATE_ORIGINS", endpoint)
    options = {**local_s3.options, "client_kwargs": {"endpoint_url": endpoint}}
    with pytest.raises((FetchPolicyError, BotoCoreError, OSError)):
        with s3.open_s3_import(S3_PATH, options) as stream:
            stream.read()
    assert local_s3.dns
    assert local_s3.connections == []
    assert local_s3.requests == []


@pytest.mark.parametrize(
    "ip", ["127.0.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1", "fd00:ec2::254"]
)
def test_configured_hostname_resolving_to_forbidden_ip_is_blocked(local_s3, ip):
    local_s3.dns_ips = [ip]
    with pytest.raises((FetchPolicyError, BotoCoreError, OSError)):
        with s3.open_s3_import(S3_PATH, local_s3.options) as stream:
            stream.read()
    assert local_s3.dns
    assert local_s3.connections == []
    assert local_s3.requests == []


@pytest.mark.asyncio
async def test_context_resolver_returns_checked_numeric_destination(local_s3):
    token = s3._request_url.set(local_s3.endpoint + OBJECT_PATH)
    try:
        results = await s3._Resolver().resolve(
            "storage.test", local_s3.port, socket.AF_UNSPEC
        )
    finally:
        s3._request_url.reset(token)
    assert results == [
        {
            "hostname": "storage.test",
            "host": PRIVATE_IP,
            "port": local_s3.port,
            "family": socket.AF_INET,
            "proto": socket.IPPROTO_TCP,
            "flags": socket.AI_NUMERICHOST,
        }
    ]
    assert local_s3.connections == []


@pytest.mark.parametrize("mismatch", ["host", "port"])
@pytest.mark.asyncio
async def test_context_resolver_rejects_destination_mismatch_before_dns(
    local_s3, mismatch
):
    token = s3._request_url.set(local_s3.endpoint + OBJECT_PATH)
    try:
        host = "unexpected.test" if mismatch == "host" else "storage.test"
        port = local_s3.port + 1 if mismatch == "port" else local_s3.port
        with pytest.raises(
            FetchPolicyError, match="differs from the validated request"
        ):
            await s3._Resolver().resolve(host, port)
    finally:
        s3._request_url.reset(token)
    assert local_s3.dns == []
    assert local_s3.connections == []


@pytest.mark.parametrize(
    "host",
    ["example.com", "s3.amazonaws.com.attacker.test", "test-bucket.s3.amazonaws.com"],
)
@pytest.mark.asyncio
async def test_session_rejects_unexpected_s3_host_before_dns(local_s3, host):
    request = AWSRequest(
        method="GET", url=f"https://{host}/test-bucket/rows.parquet"
    ).prepare()
    async with s3._GuardedS3Session() as session:
        with pytest.raises(FetchPolicyError, match="not an approved storage endpoint"):
            await session.send(request)
    assert local_s3.dns == []
    assert local_s3.connections == []
    assert local_s3.requests == []
