import json
import time

import fakes
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.api import create_app
from src.settings import get_settings

TOPIC = {"topic": "urban forests", "max_analyst": 2}


def make_client(**kwargs):
    saver = MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[("src.utils.objects", "Analyst")]))
    return TestClient(create_app(get_settings(), checkpointer=saver), **kwargs)


@pytest.fixture
def client(llm):
    with make_client() as c:
        yield c


def read_events(client, thread_id, after=0):
    events = []
    with client.stream("GET", f"/api/research/{thread_id}/events?after={after}") as response:
        assert response.status_code == 200
        name = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
    return events


def names(events):
    return [name for name, _ in events]


def start(client, **extra):
    response = client.post("/api/research", json={**TOPIC, **extra})
    assert response.status_code == 201, response.text
    thread_id = response.json()["thread_id"]
    return thread_id, read_events(client, thread_id)


def approve(client, thread_id, **body):
    response = client.post(f"/api/research/{thread_id}/feedback", json={"feedback": "", **body})
    assert response.status_code == 200, response.text
    return read_events(client, thread_id)


def finished_run(client, **extra):
    thread_id, _ = start(client, **extra)
    approve(client, thread_id)
    return thread_id


def test_health_reports_keys_and_options(client):
    data = client.get("/api/health").json()
    assert data["ok"] and data["openai"] and data["tavily"] and not data["auth"]
    assert {"code": "tr", "name": "Türkçe"} in data["languages"]


def test_full_flow_streams_progress_and_persists_the_report(client):
    thread_id, events = start(client)
    assert names(events)[-1] == "done"
    assert events[-1][1] == {"status": "awaiting_approval"}
    analysts = next(data for name, data in events if name == "analysts")["analysts"]
    assert [a["name"] for a in analysts] == ["Analyst 1", "Analyst 2"]

    record = client.get(f"/api/research/{thread_id}").json()
    assert record["status"] == "awaiting_approval" and not record["running"]
    assert [a["name"] for a in record["analysts"]] == ["Analyst 1", "Analyst 2"]

    events = approve(client, thread_id)
    kinds = names(events)
    assert kinds.count("interview_start") == 2
    started = [data for name, data in events if name == "interview_start"]
    assert {s["analyst"] for s in started} == {"Analyst 1", "Analyst 2"}
    assert {"message", "search", "stage", "draft", "usage", "report"} <= set(kinds)
    assert kinds[-1] == "done" and events[-1][1] == {"status": "done"}

    sources_searched = {data["source"] for name, data in events if name == "search"}
    assert sources_searched == {"web", "wikipedia"}

    report = next(data for name, data in events if name == "report")
    assert report["final_report"].startswith("# Big Title")
    assert report["sources"] and report["usage"]["total_tokens"] > 0
    assert "".join(d["delta"] for n, d in events if n == "draft").startswith("## Insights")

    record = client.get(f"/api/research/{thread_id}").json()
    assert record["status"] == "done" and record["report"] == report["final_report"]
    assert record["usage"]["calls"] > 0


def test_events_can_be_replayed_from_a_position(client):
    thread_id, events = start(client)
    again = read_events(client, thread_id, after=0)
    assert names(again) == names(events)
    tail = read_events(client, thread_id, after=len(events) - 1)
    assert names(tail) == ["done"]


def test_regenerate_then_approve(client):
    thread_id, _ = start(client)
    response = client.post(f"/api/research/{thread_id}/feedback", json={"feedback": "add an economist"})
    assert response.status_code == 200
    events = read_events(client, thread_id)
    team = next(data for name, data in events if name == "analysts")["analysts"]
    assert all("add an economist" in a["description"] for a in team)
    assert events[-1][1] == {"status": "awaiting_approval"}

    events = approve(client, thread_id)
    assert events[-1][1] == {"status": "done"}


def test_manual_team_edits_are_used(client):
    thread_id, _ = start(client)
    edited = [{"name": "Dr. Edit", "role": "Economist", "affiliation": "Bank", "description": "Costs"}]
    events = approve(client, thread_id, analysts=edited)
    started = [data for name, data in events if name == "interview_start"]
    assert [s["analyst"] for s in started] == ["Dr. Edit"]


def test_feedback_is_rejected_when_not_waiting_or_invalid(client):
    thread_id = finished_run(client)
    assert client.post(f"/api/research/{thread_id}/feedback", json={"feedback": ""}).status_code == 409

    waiting, _ = start(client)
    assert client.post(f"/api/research/{waiting}/feedback", json={"analysts": []}).status_code == 422
    too_long = {"name": "x" * 500, "role": "r"}
    assert client.post(f"/api/research/{waiting}/feedback", json={"analysts": [too_long]}).status_code == 422
    assert client.post("/api/research/nope/feedback", json={"feedback": ""}).status_code == 404


def test_start_validation(client):
    assert client.post("/api/research", json={"topic": "ab"}).status_code == 422
    assert client.post("/api/research", json={**TOPIC, "max_analyst": 9}).status_code == 422
    response = client.post("/api/research", json={**TOPIC, "depth": "quick", "fact_check": False})
    assert response.status_code == 201
    record = client.get(f"/api/research/{response.json()['thread_id']}").json()
    assert record["settings"] == {"max_analyst": 2}
    read_events(client, response.json()["thread_id"])


def test_every_research_uses_the_most_thorough_settings(client, llm, monkeypatch):
    searches = []

    def web(query, search_depth="basic", max_results=3):
        searches.append((search_depth, max_results))
        return fakes.fake_tavily(query)

    monkeypatch.setattr("src.utils.nodes.tavily_search", web)
    thread_id, _ = start(client, max_analyst=1)
    approve(client, thread_id)

    assert llm.calls.count("answer") == 3
    assert "review" in llm.calls
    assert searches == [("advanced", 10)] * 3


def test_missing_keys_are_reported(client, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY")
    assert client.post("/api/research", json=TOPIC).status_code == 503
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    monkeypatch.delenv("OPENAI_API_KEY")
    assert client.post("/api/research", json=TOPIC).status_code == 503


def test_errors_are_sanitized_and_logged_with_a_reference(client, llm):
    llm.fail_on = "analysts"
    response = client.post("/api/research", json=TOPIC)
    thread_id = response.json()["thread_id"]
    events = read_events(client, thread_id)

    error = next(data for name, data in events if name == "error")
    assert "secret internal detail" not in json.dumps(events)
    assert error["code"] == "unknown" and len(error["ref"]) == 8
    record = client.get(f"/api/research/{thread_id}").json()
    assert record["status"] == "error" and record["error"]["ref"] == error["ref"]


def test_cancel_stops_the_run_and_resume_finishes_it(client, llm):
    thread_id, _ = start(client, max_analyst=1)
    llm.delay = 0.3

    response = client.post(f"/api/research/{thread_id}/feedback", json={"feedback": ""})
    assert response.status_code == 200
    with client.stream("GET", f"/api/research/{thread_id}/events") as stream:
        for line in stream.iter_lines():
            if line == "event: interview_start":
                break

    assert client.post(f"/api/research/{thread_id}/cancel").json() == {"cancelled": True}
    assert client.get(f"/api/research/{thread_id}").json()["status"] == "cancelled"
    assert client.post(f"/api/research/{thread_id}/cancel").json() == {"cancelled": False}

    llm.delay = 0
    time.sleep(0.8)
    assert client.post(f"/api/research/{thread_id}/resume").status_code == 200
    events = read_events(client, thread_id)
    assert events[-1][1] == {"status": "done"}
    assert client.get(f"/api/research/{thread_id}").json()["report"]


def test_resume_is_rejected_for_finished_runs(client):
    thread_id = finished_run(client)
    assert client.post(f"/api/research/{thread_id}/resume").status_code == 409


def test_concurrent_run_limit(llm, monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_RUNS", "1")
    llm.delay = 0.3
    with make_client() as client:
        first = client.post("/api/research", json=TOPIC)
        assert first.status_code == 201
        second = client.post("/api/research", json=TOPIC)
        assert second.status_code == 429 and second.headers["retry-after"]
        read_events(client, first.json()["thread_id"])
        assert client.post("/api/research", json=TOPIC).status_code == 201
        assert len(client.get("/api/research").json()["runs"]) == 2


def test_second_segment_cannot_start_while_one_is_running(client, llm):
    thread_id, _ = start(client)
    llm.delay = 0.3
    assert client.post(f"/api/research/{thread_id}/feedback", json={"feedback": ""}).status_code == 200
    assert client.post(f"/api/research/{thread_id}/feedback", json={"feedback": ""}).status_code == 409
    client.post(f"/api/research/{thread_id}/cancel")


def test_rate_limit(llm, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "2")
    with make_client() as client:
        for _ in range(2):
            assert client.post("/api/research", json=TOPIC).status_code == 201
        blocked = client.post("/api/research", json=TOPIC)
        assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) > 0


def test_history_lists_and_deletes(client):
    thread_id = finished_run(client)
    runs = client.get("/api/research").json()["runs"]
    assert runs[0]["thread_id"] == thread_id and runs[0]["status"] == "done" and runs[0]["has_report"]

    assert client.delete(f"/api/research/{thread_id}").status_code == 204
    assert client.get(f"/api/research/{thread_id}").status_code == 404
    assert client.get("/api/research").json()["runs"] == []


def test_export_downloads_the_markdown_report(client):
    thread_id = finished_run(client)
    md = client.get(f"/api/research/{thread_id}/export")
    assert md.status_code == 200 and md.text.startswith("# Big Title")
    assert md.headers["content-type"].startswith("text/markdown")
    assert 'filename="urban-forests.md"' in md.headers["content-disposition"]

    waiting, _ = start(client)
    assert client.get(f"/api/research/{waiting}/export").status_code == 409


def test_api_key_protects_the_api_but_not_the_shell(llm, monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "secret-key")
    with make_client() as client:
        assert client.get("/api/health").json()["auth"] is True
        assert client.get("/").status_code == 200
        assert client.get("/static/style.css").status_code == 200

        assert client.post("/api/research", json=TOPIC).status_code == 401
        assert client.get("/api/research").status_code == 401
        assert client.get("/api/research", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/research", headers={"X-API-Key": "secret-key"}).status_code == 200
        assert client.get("/api/research", headers={"Authorization": "Bearer secret-key"}).status_code == 200
        assert client.get("/api/auth/check").status_code == 401
        assert client.get("/docs").status_code == 404


def test_security_headers(client):
    for path in ("/", "/api/health"):
        headers = client.get(path).headers
        assert "script-src 'self'" in headers["content-security-policy"]
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
    assert client.get("/api/health").headers["cache-control"] == "no-store"
