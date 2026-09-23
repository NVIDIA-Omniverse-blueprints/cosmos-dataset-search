"""Exercise the production mounted app, including its metrics middleware.

Router-only tests miss instrumentation failures before endpoint dispatch. No
lifespan is started here: pipelines and external stores are deliberately absent.
"""

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families


@pytest.fixture(params=["", "/api"])
def instrumented_client(request, monkeypatch):
    from src.visual_search.main import app
    from src.visual_search.v1.apis import pipelines

    monkeypatch.setattr(app, "root_path", request.param)
    monkeypatch.setattr(pipelines, "get_all_pipelines", lambda: {})
    monkeypatch.setattr(pipelines, "draw_haystack_pipeline", lambda *_: b"PNG")
    # Using TestClient without entering its context skips external-service
    # startup, but retains the real included routers and middleware stack.
    client = TestClient(app, root_path=request.param, raise_server_exceptions=False)
    yield client, request.param + "/v1"
    client.close()


def test_included_routes_work_with_metrics_enabled(instrumented_client):
    client, prefix = instrumented_client
    response = client.get(prefix + "/pipelines")
    assert response.status_code == 200, response.text
    assert response.json() == {"pipelines": []}
    assert "x-trace-id" in response.headers

    response = client.get(prefix + "/pipelines/draw/instrumentation-test")
    assert response.status_code == 200, response.text
    assert response.content == b"PNG"

    response = client.get(prefix + "/metrics")
    assert response.status_code == 200, response.text
    samples = [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == "http_requests_total"
    ]
    assert any(
        sample.labels.get("handler", "").endswith("/pipelines")
        and sample.labels.get("method") == "GET"
        and sample.labels.get("status") == "2xx"
        and sample.value >= 1
        for sample in samples
    )
    assert any(
        sample.labels.get("handler", "").endswith("/pipelines/draw/{name}")
        and sample.value >= 1
        for sample in samples
    )
    # Path parameters must remain templated, not become unbounded labels.
    assert all(
        "instrumentation-test" not in sample.labels.get("handler", "")
        for sample in samples
    )


def test_instrumentation_preserves_validation_and_not_found(instrumented_client):
    client, prefix = instrumented_client
    response = client.post(prefix + "/collections", json={})
    assert response.status_code == 422, response.text
    response = client.get(prefix + "/not-a-route")
    assert response.status_code == 404, response.text
