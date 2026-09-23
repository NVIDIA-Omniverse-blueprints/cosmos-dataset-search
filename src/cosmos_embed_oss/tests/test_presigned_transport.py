# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the downloader's socket boundary without external network access."""

import io
import socket
import ssl
from collections import deque

import pytest
from src.cosmos_embed_oss import backends

PUBLIC_IP = "93.184.216.34"
PUBLIC_IPV6 = "2606:4700:4700::1111"
MEDIA_HOST = "media.example.test"
MEDIA_URL = f"https://{MEDIA_HOST}/clip.mp4?X-Amz-Signature=synthetic%2Bvalue"
OK_RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nclip"


def _address(ip, port=443):
    ipv6 = ":" in ip
    return (
        socket.AF_INET6 if ipv6 else socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        (ip, port, 0, 0) if ipv6 else (ip, port),
    )


@pytest.fixture(autouse=True)
def isolated_download_config(monkeypatch):
    for key in (
        backends.PRESIGNED_URL_ENDPOINT_ENV,
        backends.PRESIGNED_URL_ALLOWED_HOSTS_ENV,
        backends.PRESIGNED_URL_ALLOWED_HTTP_ORIGINS_ENV,
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(backends.urllib.request, "getproxies", lambda: {})
    monkeypatch.setattr(backends.urllib.request, "proxy_bypass", lambda _host: False)


@pytest.fixture
def wire(monkeypatch):
    """Record numeric dials, HTTP bytes and TLS names; never create a real socket."""
    state = {"sockets": [], "responses": deque([OK_RESPONSE]), "tls_names": []}

    class FakeSocket:
        def __init__(self, *args):
            self.args = args
            self.sent = []
            self.address = None
            self.timeout = None
            self.bound = None
            self.closed = False
            state["sockets"].append(self)

        def settimeout(self, timeout):
            self.timeout = timeout

        def setsockopt(self, *_args):
            pass

        def bind(self, address):
            self.bound = address

        def connect(self, address):
            self.address = address
            if address[0] in state.get("failed_ips", set()):
                raise OSError("synthetic connection failure")

        def sendall(self, data):
            self.sent.append(data)

        def makefile(self, *_args):
            return io.BytesIO(state["responses"].popleft())

        def close(self):
            self.closed = True

    def wrap_socket(context, sock, *, server_hostname):
        assert context.check_hostname is True
        assert context.verify_mode == ssl.CERT_REQUIRED
        state["tls_names"].append(server_hostname)
        return sock

    monkeypatch.setattr(socket, "socket", FakeSocket)
    monkeypatch.setattr(ssl.SSLContext, "wrap_socket", wrap_socket)
    return state


@pytest.mark.parametrize(
    "replacement", ["169.254.169.254", "127.0.0.1", "10.0.0.1", "::1", "fd00::1"]
)
def test_rebinding_is_rejected_before_socket_creation(
    replacement, monkeypatch, wire, tmp_path
):
    answers = iter([PUBLIC_IP, replacement])
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, port, **_kw: [_address(next(answers), port)],
    )

    with pytest.raises(backends.EmbeddingInputError, match="private or reserved"):
        backends.download_presigned_video(MEDIA_URL, tmp_path / "clip.mp4", 2, 100)

    assert wire["sockets"] == []


def test_all_dial_answers_are_checked_before_any_connection(monkeypatch, wire):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kw: [_address(PUBLIC_IP), _address("10.0.0.1")],
    )
    connection = backends._PinnedHTTPSConnection(
        MEDIA_HOST, media_host=MEDIA_HOST, media_port=443, allow_private=False
    )

    with pytest.raises(backends.EmbeddingInputError, match="private or reserved"):
        connection.connect()

    assert wire["sockets"] == []


@pytest.mark.parametrize("ip", [PUBLIC_IP, PUBLIC_IPV6])
def test_public_download_pins_ip_and_preserves_tls_host_and_query(
    ip, monkeypatch, wire, tmp_path
):
    resolutions = []

    def resolve(host, port, **_kw):
        resolutions.append((host, port))
        return [_address(ip, port)]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    destination = tmp_path / "clip.mp4"
    backends.download_presigned_video(MEDIA_URL, destination, 7, 100)

    # URL validation and dial validation only, with no third hostname lookup.
    assert resolutions == [(MEDIA_HOST, 443), (MEDIA_HOST, 443)]
    assert wire["sockets"][0].address == _address(ip)[4]
    assert wire["sockets"][0].timeout == 7
    assert wire["tls_names"] == [MEDIA_HOST]
    request = b"".join(wire["sockets"][0].sent)
    assert b"GET /clip.mp4?X-Amz-Signature=synthetic%2Bvalue HTTP/1.1\r\n" in request
    assert f"Host: {MEDIA_HOST}\r\n".encode() in request
    assert destination.read_bytes() == b"clip"


@pytest.mark.parametrize("ip", [PUBLIC_IP, PUBLIC_IPV6])
def test_proxy_connect_is_pinned_and_auth_is_not_sent_to_media_server(
    ip, monkeypatch, wire, tmp_path
):
    monkeypatch.setattr(
        backends.urllib.request,
        "getproxies",
        lambda: {"https": "http://test-user:test-password@proxy.example.test:8080"},
    )
    resolutions = []

    def resolve(host, port, *_args, **_kw):
        resolutions.append((host, port))
        # The configured proxy is trusted infrastructure and may be private.
        return [_address("127.0.0.1" if host == "proxy.example.test" else ip, port)]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    wire["responses"].appendleft(b"HTTP/1.1 200 Connection established\r\n\r\n")
    backends.download_presigned_video(MEDIA_URL, tmp_path / "clip.mp4", 2, 100)

    assert resolutions == [
        (MEDIA_HOST, 443),
        (MEDIA_HOST, 443),
        ("proxy.example.test", 8080),
    ]
    sock = wire["sockets"][0]
    assert sock.address == ("127.0.0.1", 8080)
    tunnel, request = sock.sent
    target = f"[{ip}]" if ":" in ip else ip
    assert tunnel.startswith(f"CONNECT {target}:443 HTTP/1.".encode())
    assert b"Proxy-Authorization: Basic " in tunnel
    assert b"Proxy-Authorization" not in request
    assert f"Host: {MEDIA_HOST}\r\n".encode() in request
    assert b"X-Amz-Signature=synthetic%2Bvalue" in request
    assert wire["tls_names"] == [MEDIA_HOST]


def test_proxy_cannot_bypass_rebinding_check(monkeypatch, wire, tmp_path):
    monkeypatch.setattr(
        backends.urllib.request,
        "getproxies",
        lambda: {"https": "http://proxy.example.test:8080"},
    )
    answers = iter([PUBLIC_IP, "169.254.169.254"])

    def resolve(host, port, **_kw):
        assert host == MEDIA_HOST  # Must reject before even dialing the proxy.
        return [_address(next(answers), port)]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(backends.EmbeddingInputError, match="private or reserved"):
        backends.download_presigned_video(MEDIA_URL, tmp_path / "clip.mp4", 2, 100)
    assert wire["sockets"] == []


def test_proxy_fallback_uses_next_validated_ip_and_restores_tls_name(
    monkeypatch, wire, tmp_path
):
    monkeypatch.setattr(
        backends.urllib.request,
        "getproxies",
        lambda: {"https": "http://proxy.example.test:8080"},
    )

    def resolve(host, port, *_args, **_kw):
        if host == "proxy.example.test":
            return [_address("127.0.0.1", port)]
        return [_address(PUBLIC_IP, port), _address(PUBLIC_IPV6, port)]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    wire["responses"] = deque(
        [
            b"HTTP/1.1 502 Bad Gateway\r\n\r\n",
            b"HTTP/1.1 200 Connection established\r\n\r\n",
            OK_RESPONSE,
        ]
    )
    backends.download_presigned_video(MEDIA_URL, tmp_path / "clip.mp4", 2, 100)

    first, second = wire["sockets"]
    assert first.closed
    assert first.sent[0].startswith(f"CONNECT {PUBLIC_IP}:443 ".encode())
    assert second.sent[0].startswith(f"CONNECT [{PUBLIC_IPV6}]:443 ".encode())
    assert wire["tls_names"] == [MEDIA_HOST]


def test_failed_dial_resolution_does_not_create_a_socket(monkeypatch, wire, tmp_path):
    calls = []

    def resolve(host, port, **_kw):
        calls.append(host)
        if len(calls) == 1:
            return [_address(PUBLIC_IP, port)]
        raise socket.gaierror("synthetic DNS failure")

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(backends.MediaDownloadError, match="could not be resolved"):
        backends.download_presigned_video(MEDIA_URL, tmp_path / "clip.mp4", 2, 100)
    assert wire["sockets"] == []


def test_redirect_target_is_revalidated_at_dial_time(monkeypatch, wire, tmp_path):
    answers = deque([PUBLIC_IP, PUBLIC_IP, PUBLIC_IP, "169.254.169.254"])
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, port, **_kw: [_address(answers.popleft(), port)],
    )
    wire["responses"] = deque(
        [b"HTTP/1.1 302 Found\r\nLocation: https://other.example.test/clip.mp4\r\n\r\n"]
    )

    with pytest.raises(backends.EmbeddingInputError, match="private or reserved"):
        backends.download_presigned_video(MEDIA_URL, tmp_path / "clip.mp4", 2, 100)

    assert len(wire["sockets"]) == 1
    assert wire["sockets"][0].address == (PUBLIC_IP, 443)
    assert not answers


def test_only_exact_operator_https_endpoint_can_use_private_ip(
    monkeypatch, wire, tmp_path
):
    monkeypatch.setenv(backends.PRESIGNED_URL_ENDPOINT_ENV, f"https://{MEDIA_HOST}")
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda _host, port, **_kw: [_address("10.0.0.1", port)]
    )
    backends.download_presigned_video(MEDIA_URL, tmp_path / "clip.mp4", 2, 100)
    assert wire["sockets"][0].address == ("10.0.0.1", 443)
    assert wire["tls_names"] == [MEDIA_HOST]

    with pytest.raises(backends.EmbeddingInputError, match="private or reserved"):
        backends.download_presigned_video(
            f"https://{MEDIA_HOST}:444/clip.mp4", tmp_path / "other.mp4", 2, 100
        )
    assert len(wire["sockets"]) == 1


def test_exact_operator_http_origin_keeps_existing_behavior(
    monkeypatch, wire, tmp_path
):
    monkeypatch.setenv(
        backends.PRESIGNED_URL_ALLOWED_HTTP_ORIGINS_ENV,
        "http://fixture.example.test:8680",
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, port, *_args, **_kw: [_address("127.0.0.1", port)],
    )
    backends.download_presigned_video(
        "http://fixture.example.test:8680/clip.mp4", tmp_path / "clip.mp4", 2, 100
    )
    assert wire["sockets"][0].address == ("127.0.0.1", 8680)
    assert wire["tls_names"] == []

    with pytest.raises(backends.EmbeddingInputError, match="must use https"):
        backends.download_presigned_video(
            "http://fixture.example.test:8681/clip.mp4", tmp_path / "other.mp4", 2, 100
        )


def test_numeric_connect_fallback_closes_failed_socket_and_keeps_options(wire):
    wire["failed_ips"] = {PUBLIC_IP}
    addresses = [_address(PUBLIC_IP), _address(PUBLIC_IPV6)]
    connected = backends._connect_resolved_addresses(addresses, 3, ("::", 0))
    first, second = wire["sockets"]
    assert first.closed is True
    assert second is connected
    assert second.address == addresses[1][4]
    assert second.timeout == 3
    assert second.bound == ("::", 0)


def test_numeric_connect_failure_closes_every_socket(wire):
    wire["failed_ips"] = {PUBLIC_IP, PUBLIC_IPV6}
    with pytest.raises(OSError, match="synthetic connection failure"):
        backends._connect_resolved_addresses(
            [_address(PUBLIC_IP), _address(PUBLIC_IPV6)], 1, None
        )
    assert all(sock.closed for sock in wire["sockets"])


@pytest.mark.parametrize(
    "response,error,match",
    [
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: 200\r\n\r\n",
            backends.EmbeddingInputError,
            "maximum",
        ),
        (
            b"HTTP/1.1 200 OK\r\n\r\n" + b"x" * 101,
            backends.EmbeddingInputError,
            "maximum",
        ),
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort",
            backends.MediaDownloadError,
            "Content-Length",
        ),
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: bad\r\n\r\n",
            backends.MediaDownloadError,
            "Content-Length",
        ),
    ],
)
def test_transport_keeps_size_checks_and_materialization_cleanup(
    response, error, match, monkeypatch, wire, tmp_path
):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda _host, port, **_kw: [_address(PUBLIC_IP, port)]
    )
    wire["responses"] = deque([response])
    with pytest.raises(error, match=match):
        with backends.materialize_video_data_uri(
            "data:video/mp4;presigned_url," + MEDIA_URL,
            tmp_dir=tmp_path,
            download_timeout_seconds=2,
            max_media_bytes=100,
        ):
            pytest.fail("invalid response must not be materialized")
    assert list(tmp_path.iterdir()) == []
