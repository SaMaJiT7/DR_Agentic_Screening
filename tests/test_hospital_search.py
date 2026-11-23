"""Tests for the hospital search node using a fake HTTP client."""

from agents.nodes.hospital_search import (
    hospital_search_node,
    parse_results,
    search_hospitals,
)
from agents.state import DR_State

PAYLOAD = [
    {
        "name": "Sankara Nethralaya",
        "display_name": "Sankara Nethralaya, College Road, Chennai, Tamil Nadu, India",
        "lat": "13.06",
        "lon": "80.27",
    },
    {
        "display_name": "Aravind Eye Hospital, Madurai, Tamil Nadu, India",
        "lat": "9.92",
        "lon": "78.13",
    },
    {"display_name": ""},
    "not a dict",
    {
        "name": "sankara nethralaya",
        "display_name": "duplicate",
        "lat": "0",
        "lon": "0",
    },
]


class FakeResponse:
    def __init__(self, payload) -> None:
        self._payload = payload
        self.raised = False

    def raise_for_status(self) -> None:
        self.raised = True

    def json(self):
        return self._payload


def _state(**overrides) -> DR_State:
    values = {"image_path": "scan.png", "city": "Chennai"}
    values.update(overrides)
    return DR_State(**values)


def test_parse_results_dedupes_and_skips_junk() -> None:
    hospitals = parse_results(PAYLOAD)
    assert [h.name for h in hospitals] == [
        "Sankara Nethralaya",
        "Aravind Eye Hospital",
    ]
    assert hospitals[0].lat == 13.06
    assert hospitals[1].address.startswith("Aravind Eye Hospital")


def test_parse_results_handles_non_list() -> None:
    assert parse_results({"unexpected": True}) == []


def test_search_builds_query_and_headers() -> None:
    seen = {}
    response = FakeResponse(PAYLOAD)

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params, headers=headers, timeout=timeout)
        return response

    hospitals = search_hospitals("Chennai", limit=3, http_get=fake_get)

    assert len(hospitals) == 2
    assert response.raised is True
    assert "Chennai" in seen["params"]["q"]
    assert "eye hospital" in seen["params"]["q"]
    assert seen["params"]["limit"] == "3"
    assert "User-Agent" in seen["headers"]
    assert seen["timeout"] is not None


def test_search_rejects_blank_city() -> None:
    try:
        search_hospitals("   ", http_get=lambda *a, **k: None)
    except ValueError as exc:
        assert "city" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_node_fills_hospitals() -> None:
    import agents.nodes.hospital_search as module

    original = module.search_hospitals
    module.search_hospitals = lambda city, **kwargs: parse_results(PAYLOAD)
    try:
        update = hospital_search_node(_state())
    finally:
        module.search_hospitals = original

    assert len(update["hospitals"]) == 2
    assert "errors" not in update


def test_node_keeps_run_alive_on_failure() -> None:
    def boom(*args, **kwargs):
        raise TimeoutError("nominatim timed out")

    import agents.nodes.hospital_search as module

    original = module.search_hospitals
    module.search_hospitals = boom
    try:
        state = _state()
        update = hospital_search_node(state)
    finally:
        module.search_hospitals = original

    assert update["hospitals"] == []
    assert any("hospital search failed" in error for error in update["errors"])
    assert len(update["errors"]) == len(state.errors) + 1


if __name__ == "__main__":
    test_parse_results_dedupes_and_skips_junk()
    test_parse_results_handles_non_list()
    test_search_builds_query_and_headers()
    test_search_rejects_blank_city()
    test_node_fills_hospitals()
    test_node_keeps_run_alive_on_failure()
    print("hospital_search tests passed")
