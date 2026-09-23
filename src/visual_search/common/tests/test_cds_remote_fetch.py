# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outbound import policy regressions without CUDA or service dependencies."""

from __future__ import annotations

import io
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest

from src.visual_search.common import remote_fetch as fetch


PUBLIC_IP = "93.184.216.34"
PUBLIC_IPV6 = "2606:4700:4700::1111"


def test_s3_keys_preserve_spaces_and_unicode():
    assert fetch.validate_s3_paths(
        ["s3://test-bucket/customer uploads/café.parquet"]
    ) == ["customer uploads/café.parquet"]


@pytest.fixture(autouse=True)
def isolated_network_and_policy(monkeypatch):
    """A missing network mock fails the test instead of contacting real hosts."""
    for name in (
        "CDS_FETCH_ALLOWED_ORIGINS",
        "CDS_FETCH_PRIVATE_ORIGINS",
        "CDS_FETCH_MAX_TEXT_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)

    def unexpected_dns(*args, **kwargs):
        pytest.fail("Test attempted an unmocked DNS lookup")

    def unexpected_connect(*args, **kwargs):
        pytest.fail("Test attempted an unmocked outbound connection")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_dns)
    monkeypatch.setattr(socket.socket, "connect", unexpected_connect)


def address_info(ip, port=443):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    address = (ip, port, 0, 0) if family == socket.AF_INET6 else (ip, port)
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", address)


def resolve_to(monkeypatch, *ips):
    resolver = Mock(
        side_effect=lambda host, port, **kwargs: [address_info(ip, port) for ip in ips]
    )
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    return resolver


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/data.txt",
        "https://example.com:8443/data.txt",
        "HTTPS://EXAMPLE.COM/data.txt",
        "https://example.com./data.txt",
        "https://[2606:4700:4700::1111]/data.txt",
    ],
)
def test_public_https_is_allowed_by_default(monkeypatch, url):
    resolve_to(monkeypatch, PUBLIC_IP, PUBLIC_IPV6)
    assert len(fetch.FetchPolicy.from_env().resolve(url)) == 2


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "example.com/file",
        "//example.com/file",
        "http://example.com/file",
        "file:///etc/passwd",
        "s3://bucket/file",
        "ftp://example.com/file",
        "data:text/plain,hello",
        "https://user:password@example.com/file",
        "https://user@example.com/file",
        "https://@example.com/file",
        "https://example.com/file#fragment",
        "https://example.com:0/file",
        "https://example.com:-1/file",
        "https://example.com:65536/file",
        "https://example.com:bad/file",
        "https://[::1/file",
        "https:///file",
        "https://example.com\\@127.0.0.1/file",
        "https://example.com/file\r\nX-Header:injected",
        " https://example.com/file",
        "https://example.com/a b",
        "https://example.com/\x00",
        "https://example.com/\x7f",
        "https://%31%32%37.0.0.1/file",
        "https://[fe80::1%25eth0]/file",
    ],
)
def test_invalid_urls_are_rejected_before_dns(url):
    with pytest.raises(fetch.FetchPolicyError):
        fetch.FetchPolicy.from_env().resolve(url)


def test_allowlist_normalizes_case_and_default_port(monkeypatch):
    monkeypatch.setenv(
        "CDS_FETCH_ALLOWED_ORIGINS",
        " https://EXAMPLE.COM:443/ , https://other.example:8443 ",
    )
    policy = fetch.FetchPolicy.from_env()
    assert policy.validate_url("https://example.com/a?signature=kept")[1] == (
        "https",
        "example.com",
        443,
    )
    assert policy.validate_url("https://other.example:8443/a")[1] == (
        "https",
        "other.example",
        8443,
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://sub.example.com/a",
        "https://example.com.attacker.test/a",
        "https://another.example/a",
        "https://example.com:8443/a",
        "http://example.com/a",
    ],
)
def test_allowlist_matches_exact_origin_not_suffix_or_request_data(monkeypatch, url):
    monkeypatch.setenv("CDS_FETCH_ALLOWED_ORIGINS", "https://example.com")
    with pytest.raises(fetch.FetchPolicyError):
        fetch.FetchPolicy.from_env().resolve(url)


@pytest.mark.parametrize(
    "variable", ["CDS_FETCH_ALLOWED_ORIGINS", "CDS_FETCH_PRIVATE_ORIGINS"]
)
@pytest.mark.parametrize(
    "origin",
    [
        "https://example.com/path",
        "https://example.com/?secret=x",
        "https://example.com/#fragment",
        "https://user@example.com",
        "file:///tmp",
    ],
)
def test_operator_origins_cannot_include_path_query_credentials_or_fragment(
    monkeypatch, variable, origin
):
    monkeypatch.setenv(variable, origin)
    with pytest.raises(fetch.FetchPolicyError):
        fetch.FetchPolicy.from_env()


@pytest.mark.parametrize(
    "ip",
    [
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "fd12:3456::1",
        "fc00::123",
        "::ffff:10.0.0.1",
    ],
)
def test_private_store_requires_explicit_operator_origin(monkeypatch, ip):
    resolve_to(monkeypatch, ip)
    url = "https://store.example:9000/object"
    with pytest.raises(fetch.FetchPolicyError):
        fetch.FetchPolicy.from_env().resolve(url)
    monkeypatch.setenv("CDS_FETCH_PRIVATE_ORIGINS", "https://store.example:9000")
    assert fetch.FetchPolicy.from_env().resolve(url)


def test_private_http_store_is_explicit_and_independent_of_public_allowlist(
    monkeypatch,
):
    monkeypatch.setenv("CDS_FETCH_ALLOWED_ORIGINS", "https://public.example")
    monkeypatch.setenv("CDS_FETCH_PRIVATE_ORIGINS", "http://minio.example:9000")
    resolve_to(monkeypatch, "10.20.30.40")
    policy = fetch.FetchPolicy.from_env()
    assert policy.resolve("http://minio.example:9000/bucket/key")
    for url in (
        "https://minio.example:9000/bucket/key",
        "http://minio.example:9001/bucket/key",
    ):
        with pytest.raises(fetch.FetchPolicyError):
            policy.resolve(url)


@pytest.mark.parametrize("private_override", [False, True])
@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "127.1.2.3",
        "169.254.169.254",
        "169.254.170.2",
        "169.254.1.2",
        "100.100.100.200",
        "0.0.0.0",
        "224.0.0.1",
        "192.0.2.1",
        "::1",
        "::",
        "fe80::1",
        "ff02::1",
        "2001:db8::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "fd00:ec2::254",
        "64:ff9b::a9fe:a9fe",
        "64:ff9b::7f00:1",
        "64:ff9b:1::a9fe:a9fe",
        "2002:7f00:1::",
    ],
)
def test_reserved_and_metadata_addresses_never_allowed(
    monkeypatch, ip, private_override
):
    resolve_to(monkeypatch, ip)
    if private_override:
        monkeypatch.setenv("CDS_FETCH_PRIVATE_ORIGINS", "https://store.example")
    with pytest.raises(fetch.FetchPolicyError):
        fetch.FetchPolicy.from_env().resolve("https://store.example/object")


@pytest.mark.parametrize(
    "ips", [(PUBLIC_IP, "10.0.0.1"), ("127.0.0.1", PUBLIC_IP), (PUBLIC_IPV6, "::1")]
)
def test_any_forbidden_dns_answer_rejects_entire_destination_before_dial(
    monkeypatch, ips
):
    resolve_to(monkeypatch, *ips)
    with pytest.raises(fetch.FetchPolicyError):
        with fetch.open_remote("https://mixed.example/object"):
            pytest.fail("A mixed public/private DNS destination was opened")


@pytest.mark.parametrize("failure", [socket.gaierror("internal DNS diagnostics"), []])
def test_dns_failure_is_a_sanitized_fetch_error(monkeypatch, failure):
    resolver = (
        Mock(side_effect=failure)
        if isinstance(failure, Exception)
        else Mock(return_value=failure)
    )
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    with pytest.raises(fetch.FetchError) as exc:
        fetch.FetchPolicy.from_env().resolve("https://example.com/?token=secret")
    assert "secret" not in str(exc.value)
    assert "diagnostics" not in str(exc.value)


class WireSocket:
    """Minimal socket exposing the actual HTTP request and numeric connect target."""

    def __init__(self, response):
        self.response = response
        self.sent = bytearray()
        self.address = None
        self.timeout = None
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, address):
        self.address = address

    def sendall(self, data):
        self.sent.extend(data)

    def makefile(self, *args, **kwargs):
        return io.BytesIO(self.response)

    def close(self):
        self.closed = True


def install_wire(monkeypatch, *responses):
    sockets = [WireSocket(response) for response in responses]
    factory = Mock(side_effect=sockets)
    monkeypatch.setattr(socket, "socket", factory)
    context = Mock()
    context.wrap_socket.side_effect = lambda sock, **kwargs: sock
    create_context = Mock(return_value=context)
    monkeypatch.setattr(fetch.ssl, "create_default_context", create_context)
    return sockets, factory, context, create_context


def test_connection_pins_dns_preserves_signed_request_and_tls_hostname(monkeypatch):
    resolver = Mock(
        side_effect=[[address_info(PUBLIC_IP)], [address_info("127.0.0.1")]]
    )
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    sockets, factory, context, create_context = install_wire(
        monkeypatch, b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
    )
    target = (
        "/bucket/a%2Fb%20c.txt?X-Amz-Credential=a%2Fb&X-Amz-Signature=abc%2B123&k=1&k=2"
    )
    with fetch.open_remote("https://storage.example" + target, timeout=7) as response:
        assert response.read() == b"ok"
    resolver.assert_called_once_with("storage.example", 443, type=socket.SOCK_STREAM)
    factory.assert_called_once_with(
        socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP
    )
    assert sockets[0].address == (PUBLIC_IP, 443)
    assert sockets[0].timeout == 7
    assert bytes(sockets[0].sent).startswith(f"GET {target} HTTP/1.1\r\n".encode())
    create_context.assert_called_once_with()
    context.wrap_socket.assert_called_once_with(
        sockets[0], server_hostname="storage.example"
    )
    assert sockets[0].closed
    # Adding a default port changes the canonical Host of a presigned request.
    assert b"Host: storage.example\r\n" in sockets[0].sent


def test_ipv6_socket_dials_validated_numeric_address(monkeypatch):
    resolve_to(monkeypatch, PUBLIC_IPV6)
    sockets, factory, _, _ = install_wire(
        monkeypatch, b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
    )
    with fetch.open_remote("https://storage.example/object"):
        pass
    factory.assert_called_once_with(
        socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP
    )
    assert sockets[0].address == (PUBLIC_IPV6, 443, 0, 0)


def test_rebinding_on_redirect_cannot_make_second_socket_connection(monkeypatch):
    resolver = Mock(
        side_effect=[[address_info(PUBLIC_IP)], [address_info("169.254.169.254")]]
    )
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    sockets, factory, _, _ = install_wire(
        monkeypatch,
        b"HTTP/1.1 302 Found\r\nLocation: /next\r\nContent-Length: 0\r\n\r\n",
    )
    with pytest.raises(fetch.FetchPolicyError):
        with fetch.open_remote("https://storage.example/object"):
            pytest.fail("Redirect reached a rebound DNS address")
    assert resolver.call_count == 2
    assert factory.call_count == 1
    assert sockets[0].closed


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "destination",
    ["http://public.example/data", "https://other.example/data", "file:///etc/passwd"],
)
def test_redirects_recheck_scheme_and_origin_before_network(
    monkeypatch, status, destination
):
    monkeypatch.setenv("CDS_FETCH_ALLOWED_ORIGINS", "https://storage.example")
    resolver = resolve_to(monkeypatch, PUBLIC_IP)
    sockets, factory, _, _ = install_wire(
        monkeypatch,
        f"HTTP/1.1 {status} Redirect\r\nLocation: {destination}\r\n"
        "Content-Length: 0\r\n\r\n".encode(),
    )
    with pytest.raises(fetch.FetchPolicyError):
        with fetch.open_remote("https://storage.example/object"):
            pytest.fail("Redirect escaped the operator policy")
    assert resolver.call_count == 1
    assert factory.call_count == 1
    assert sockets[0].closed


def test_relative_redirect_preserves_new_signed_query(monkeypatch):
    resolve_to(monkeypatch, PUBLIC_IP)
    target = "/bucket/final%20name?signature=x%2By%2Fz&key=1&key=2"
    sockets, _, _, _ = install_wire(
        monkeypatch,
        f"HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 0\r\n\r\n".encode(),
        b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\ndone",
    )
    with fetch.open_remote("https://storage.example/start") as response:
        assert response.read() == b"done"
    assert bytes(sockets[1].sent).startswith(f"GET {target} HTTP/1.1\r\n".encode())
    assert all(sock.closed for sock in sockets)


def test_redirect_loop_is_bounded_and_connections_closed(monkeypatch):
    resolver = resolve_to(monkeypatch, PUBLIC_IP)
    redirect = b"HTTP/1.1 302 Found\r\nLocation: /loop\r\nContent-Length: 0\r\n\r\n"
    sockets, factory, _, _ = install_wire(monkeypatch, *([redirect] * 4))
    with pytest.raises(fetch.FetchError):
        with fetch.open_remote("https://storage.example/loop"):
            pytest.fail("Unbounded redirect loop was accepted")
    assert factory.call_count == resolver.call_count == 4
    assert all(sock.closed for sock in sockets)


def test_redirect_without_location_is_an_error(monkeypatch):
    resolve_to(monkeypatch, PUBLIC_IP)
    sockets, factory, _, _ = install_wire(
        monkeypatch, b"HTTP/1.1 302 Found\r\nContent-Length: 0\r\n\r\n"
    )
    with pytest.raises(fetch.FetchError):
        with fetch.open_remote("https://storage.example/data"):
            pytest.fail("Redirect without a destination was accepted")
    factory.assert_called_once()
    assert sockets[0].closed


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
def test_error_status_is_not_returned_as_text_and_does_not_leak_secrets(
    monkeypatch, status
):
    resolve_to(monkeypatch, PUBLIC_IP)
    body = b"server-secret-body"
    sockets, _, _, _ = install_wire(
        monkeypatch,
        f"HTTP/1.1 {status} Error\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + body,
    )
    with pytest.raises(fetch.FetchError) as exc:
        fetch.download_text("https://storage.example/data?signature=secret-signature")
    assert "secret" not in str(exc.value)
    assert sockets[0].closed


@pytest.mark.parametrize("body", [b"", b"123", b"1234"])
def test_text_download_accepts_up_to_the_configured_byte_limit(monkeypatch, body):
    monkeypatch.setenv("CDS_FETCH_MAX_TEXT_BYTES", "4")
    response = Mock()
    response.read.return_value = body

    @contextmanager
    def opened(url):
        yield response

    monkeypatch.setattr(fetch, "open_remote", opened)
    assert fetch.download_text("https://storage.example/data") == body
    response.read.assert_called_once_with(5)


@pytest.mark.parametrize("length_header", [b"", b"Content-Length: 5\r\n"])
def test_oversized_text_is_rejected_even_without_content_length(
    monkeypatch, length_header
):
    monkeypatch.setenv("CDS_FETCH_MAX_TEXT_BYTES", "4")
    resolve_to(monkeypatch, PUBLIC_IP)
    sockets, _, _, _ = install_wire(
        monkeypatch, b"HTTP/1.1 200 OK\r\n" + length_header + b"\r\n12345"
    )
    with pytest.raises(fetch.FetchPolicyError, match="exceeds"):
        fetch.download_text("https://storage.example/data")
    assert sockets[0].closed


def test_chunked_text_still_obeys_byte_limit(monkeypatch):
    monkeypatch.setenv("CDS_FETCH_MAX_TEXT_BYTES", "4")
    resolve_to(monkeypatch, PUBLIC_IP)
    sockets, _, _, _ = install_wire(
        monkeypatch,
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
        b"3\r\n123\r\n3\r\n456\r\n0\r\n\r\n",
    )
    with pytest.raises(fetch.FetchPolicyError, match="exceeds"):
        fetch.download_text("https://storage.example/data")
    assert sockets[0].closed


def test_ambient_proxy_cannot_change_checked_destination(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://user:proxy-secret@127.0.0.1:3128")
    resolver = resolve_to(monkeypatch, PUBLIC_IP)
    sockets, _, _, _ = install_wire(
        monkeypatch, b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
    )
    assert fetch.download_text("https://storage.example/data") == b"ok"
    resolver.assert_called_once_with("storage.example", 443, type=socket.SOCK_STREAM)
    assert sockets[0].address == (PUBLIC_IP, 443)
    assert b"proxy-secret" not in sockets[0].sent
    assert b"Proxy-Authorization" not in sockets[0].sent


def test_connection_failure_closes_all_attempts_and_redacts_diagnostics(monkeypatch):
    resolve_to(monkeypatch, PUBLIC_IP, "93.184.216.35")
    sockets, factory, _, _ = install_wire(monkeypatch, b"", b"")
    for sock in sockets:
        sock.connect = Mock(side_effect=TimeoutError("private network diagnostic"))
    with pytest.raises(fetch.FetchError) as exc:
        fetch.download_text("https://storage.example/data?signature=secret")
    assert factory.call_count == 2
    assert all(sock.closed for sock in sockets)
    assert "private" not in str(exc.value)
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_invalid_text_limit_is_rejected_before_network(monkeypatch, limit):
    monkeypatch.setenv("CDS_FETCH_MAX_TEXT_BYTES", limit)
    with pytest.raises(fetch.FetchPolicyError, match="positive"):
        fetch.download_text("https://storage.example/data")


def test_tls_verification_failure_closes_socket_and_is_sanitized(monkeypatch):
    resolve_to(monkeypatch, PUBLIC_IP)
    sockets, _, context, _ = install_wire(monkeypatch, b"")
    context.wrap_socket.side_effect = fetch.ssl.SSLCertVerificationError(
        "private TLS diagnostic"
    )
    with pytest.raises(fetch.FetchError) as exc:
        with fetch.open_remote("https://storage.example/data?signature=secret"):
            pytest.fail("TLS verification failure was ignored")
    assert sockets[0].closed
    assert "private" not in str(exc.value)
    assert "secret" not in str(exc.value)


def test_real_http_server_success_and_blocked_redirect(monkeypatch):
    """Use real HTTP parsing/I/O; only map a policy-approved numeric dial to loopback."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header(
                    "Location", "https://metadata.example/latest/meta-data"
                )
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Length", "4")
                self.end_headers()
                self.wfile.write(b"text")

        def log_message(self, *args):
            pass

    # Save the native dial from the class's descriptor, not the autouse guard.
    import _socket

    native_socket = socket.socket
    native_connect = _socket.socket.connect
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    worker = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    worker.start()
    dialed = []

    class RoutedSocket(native_socket):
        def connect(self, address):
            dialed.append(address)
            assert address == (
                PUBLIC_IP,
                port,
            ), "Only the approved numeric target may be dialed"
            return native_connect(self, ("127.0.0.1", port))

    def resolver(host, resolved_port, **kwargs):
        if host == "store.example":
            return [address_info(PUBLIC_IP, resolved_port)]
        assert host == "metadata.example"
        return [address_info("169.254.169.254", resolved_port)]

    monkeypatch.setattr(socket, "socket", RoutedSocket)
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    monkeypatch.setenv("CDS_FETCH_PRIVATE_ORIGINS", f"http://store.example:{port}")
    origin = f"http://store.example:{port}"
    try:
        signed_path = "/bucket/a%2Fb?signature=x%2By&key=1&key=2"
        assert fetch.download_text(origin + signed_path) == b"text"
        with pytest.raises(fetch.FetchPolicyError):
            fetch.download_text(origin + "/redirect")
        assert requests == [signed_path, "/redirect"]
        assert dialed == [(PUBLIC_IP, port), (PUBLIC_IP, port)]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        assert not worker.is_alive()
