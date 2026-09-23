# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real loopback HTTPS/CONNECT checks, including certificate rejection."""

import datetime
import http.server
import select
import socket
import socketserver
import ssl
import threading
from contextlib import ExitStack, contextmanager

import pytest
from src.cosmos_embed_oss import backends


@contextmanager
def _running(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _certificate(tmp_path, name):
    # These private test keys are generated per run, never committed or reused.
    x509 = pytest.importorskip("cryptography.x509")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(name)]), critical=False
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "test-ca.pem", tmp_path / "test-key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    return cert_path, key_path


@pytest.mark.parametrize("use_proxy", [False, True], ids=["direct", "connect-proxy"])
@pytest.mark.parametrize("certificate", ["valid", "wrong-host", "untrusted"])
def test_live_https_verifies_tls_through_pinned_connection(
    use_proxy, certificate, monkeypatch, tmp_path
):
    hostname = "media.example.test"
    cert_path, key_path = _certificate(
        tmp_path, "wrong.example.test" if certificate == "wrong-host" else hostname
    )
    requests, tls_names, tunnel_targets = [], [], []

    class MediaHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers["Host"]))
            self.send_response(200)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"clip")

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MediaHandler)
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    server_context.set_servername_callback(
        lambda _sock, name, _ctx: tls_names.append(name)
    )
    server.socket = server_context.wrap_socket(server.socket, server_side=True)
    server_port = server.server_port

    class ProxyHandler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(3)
            first_line = self.rfile.readline()
            tunnel_targets.append(first_line.split()[1].decode())
            assert tunnel_targets[-1] == f"127.0.0.1:{server_port}"
            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                pass
            with socket.create_connection(
                ("127.0.0.1", server_port), timeout=3
            ) as upstream:
                self.connection.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                sockets = (self.connection, upstream)
                while True:
                    ready, _, _ = select.select(sockets, (), (), 3)
                    if not ready:
                        return
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        destination = (
                            upstream if source is self.connection else self.connection
                        )
                        destination.sendall(data)

    monkeypatch.delenv(backends.PRESIGNED_URL_ALLOWED_HOSTS_ENV, raising=False)
    monkeypatch.delenv(backends.PRESIGNED_URL_ALLOWED_HTTP_ORIGINS_ENV, raising=False)
    monkeypatch.setenv(
        backends.PRESIGNED_URL_ENDPOINT_ENV, f"https://{hostname}:{server_port}"
    )
    monkeypatch.setattr(backends.urllib.request, "proxy_bypass", lambda _host: False)
    real_resolve = socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        return real_resolve(
            "127.0.0.1" if host == hostname else host, port, *args, **kwargs
        )

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if certificate != "untrusted":
        client_context.load_verify_locations(cafile=str(cert_path))
    assert client_context.check_hostname is True
    assert client_context.verify_mode == ssl.CERT_REQUIRED
    monkeypatch.setattr(ssl, "_create_default_https_context", lambda: client_context)

    with ExitStack() as stack:
        stack.enter_context(_running(server))
        proxies = {}
        if use_proxy:
            proxy = socketserver.ThreadingTCPServer(("127.0.0.1", 0), ProxyHandler)
            proxy.daemon_threads = True
            stack.enter_context(_running(proxy))
            proxies["https"] = f"http://127.0.0.1:{proxy.server_address[1]}"
        monkeypatch.setattr(backends.urllib.request, "getproxies", lambda: proxies)
        path = "/clip.mp4?X-Amz-Signature=synthetic%2Bvalue"
        url = f"https://{hostname}:{server_port}{path}"
        destination = tmp_path / "clip.mp4"
        if certificate == "valid":
            backends.download_presigned_video(url, destination, 3, 100)
            assert destination.read_bytes() == b"clip"
            assert requests == [(path, f"{hostname}:{server_port}")]
        else:
            with pytest.raises(backends.MediaDownloadError) as caught:
                backends.download_presigned_video(url, destination, 3, 100)
            cause = caught.value.__cause__
            assert isinstance(cause.reason, ssl.SSLCertVerificationError)
            assert not requests
            assert not destination.exists()
        assert tls_names == [hostname]
        assert tunnel_targets == ([f"127.0.0.1:{server_port}"] if use_proxy else [])
