"""
GBIF reference-photograph lookup, with the network stubbed out.

The cases here are the ones that are easy to get wrong and expensive to debug
against a live API: which names get looked up, which life stages come back,
what the cache does on a second run, and that a failed request degrades to an
empty panel instead of failing the run.
"""
import json

import pandas as pd
import pytest

from biocluster.annotation import gbif_media
from biocluster.annotation.interactive_report import _ref_srcs, displayed_species


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _match(usage_key, confidence=99, match_type="EXACT"):
    return {"usageKey": usage_key, "confidence": confidence,
            "matchType": match_type}


def _occurrences(*pairs):
    """pairs are (url, lifeStage); lifeStage None means the field is absent."""
    results = []
    for url, stage in pairs:
        occ = {"media": [{"type": "StillImage", "identifier": url}]}
        if stage is not None:
            occ["lifeStage"] = stage
        results.append(occ)
    return {"results": results}


def _stub(monkeypatch, filtered, unfiltered=None):
    """Answer the match call, then the stage-filtered and unfiltered searches."""
    def fake_request(url, params=None):
        if url == gbif_media.GBIF_MATCH_URL:
            return _FakeResponse(_match(1))
        if params and "lifeStage" in params:
            return _FakeResponse(filtered)
        return _FakeResponse(unfiltered if unfiltered is not None else filtered)

    monkeypatch.setattr(gbif_media, "_request", fake_request)


@pytest.fixture(autouse=True)
def no_pacing(monkeypatch):
    """The collector paces itself for GBIF's sake; tests should not wait."""
    monkeypatch.setattr(gbif_media, "GBIF_DELAY", 0)
    monkeypatch.setattr(gbif_media, "GBIF_RETRY_DELAY", 0)


def test_displayed_species_is_top_choice_and_runner_up():
    df = pd.DataFrame({
        "ai_annotation":    ["Vulpes vulpes", "Vulpes vulpes", "Noise / Unknown"],
        "ai_second_choice": ["Vulpes lagopus", "", None],
    })
    assert displayed_species(df) == ["Vulpes vulpes", "Vulpes lagopus"]


def test_noise_and_blank_names_are_not_looked_up():
    df = pd.DataFrame({"ai_annotation": ["Noise / Unknown", "noise", "", "Bufo bufo"]})
    assert displayed_species(df) == ["Bufo bufo"]


def test_photos_carry_their_life_stage(monkeypatch):
    _stub(monkeypatch, _occurrences(("http://a/1.jpg", "Adult"),
                                    ("http://b/2.jpg", "Imago")))
    out = gbif_media.collect_reference_urls(["Vulpes vulpes"], images_per_species=2)
    assert out == {"Vulpes vulpes": [
        {"src": "http://a/1.jpg", "stage": "Adult"},
        {"src": "http://b/2.jpg", "stage": "Imago"},
    ]}


def test_adult_stages_are_requested():
    # Both spellings are needed: Darwin Core records the insect adult as
    # either, and asking for one discards most of the available photographs.
    assert "Adult" in gbif_media.ADULT_STAGES
    assert "Imago" in gbif_media.ADULT_STAGES


def test_requested_stages_are_sent_to_gbif(monkeypatch):
    sent = {}

    def fake_request(url, params=None):
        if url == gbif_media.GBIF_MATCH_URL:
            return _FakeResponse(_match(1))
        sent.update(params or {})
        return _FakeResponse(_occurrences(("http://a/1.jpg", "Adult")))

    monkeypatch.setattr(gbif_media, "_request", fake_request)
    gbif_media.collect_reference_urls(["X y"], images_per_species=1)
    assert sent["lifeStage"] == ["Adult", "Imago"]


def test_larva_is_never_returned_even_unfiltered(monkeypatch):
    # The filtered search comes back empty, so the top-up search runs. It is
    # unfiltered, so it must drop the immature records itself.
    _stub(monkeypatch,
          filtered={"results": []},
          unfiltered=_occurrences(("http://a/larva.jpg", "Larva"),
                                  ("http://a/cat.jpg", "Caterpillar"),
                                  ("http://a/pupa.jpg", "Pupa"),
                                  ("http://a/adult.jpg", "Adult")))
    out = gbif_media.collect_reference_urls(["X y"], images_per_species=3)
    assert out == {"X y": [{"src": "http://a/adult.jpg", "stage": "Adult"}]}


def test_absent_stage_is_labelled_not_dropped(monkeypatch):
    _stub(monkeypatch,
          filtered={"results": []},
          unfiltered=_occurrences(("http://a/1.jpg", None)))
    out = gbif_media.collect_reference_urls(["X y"], images_per_species=1)
    assert out["X y"] == [{"src": "http://a/1.jpg",
                           "stage": gbif_media.STAGE_UNRECORDED}]


def test_topup_runs_only_when_filtered_is_short(monkeypatch):
    searches = []

    def fake_request(url, params=None):
        if url == gbif_media.GBIF_MATCH_URL:
            return _FakeResponse(_match(1))
        searches.append("filtered" if "lifeStage" in (params or {}) else "plain")
        return _FakeResponse(_occurrences(("http://a/1.jpg", "Adult"),
                                          ("http://a/2.jpg", "Adult")))

    monkeypatch.setattr(gbif_media, "_request", fake_request)
    gbif_media.collect_reference_urls(["X y"], images_per_species=2)
    assert searches == ["filtered"]


def test_custom_life_stages_are_honoured(monkeypatch):
    sent = {}

    def fake_request(url, params=None):
        if url == gbif_media.GBIF_MATCH_URL:
            return _FakeResponse(_match(1))
        sent.update(params or {})
        return _FakeResponse(_occurrences(("http://a/1.jpg", "Juvenile")))

    monkeypatch.setattr(gbif_media, "_request", fake_request)
    out = gbif_media.collect_reference_urls(
        ["X y"], images_per_species=1, life_stages=("Juvenile",))
    assert sent["lifeStage"] == ["Juvenile"]
    # Juvenile is on the immature list, but an explicit request for it wins.
    assert out["X y"] == [{"src": "http://a/1.jpg", "stage": "Juvenile"}]


def test_unmatched_name_yields_empty_list_not_an_error(monkeypatch):
    monkeypatch.setattr(gbif_media, "_request",
                        lambda url, params=None: _FakeResponse(
                            {"matchType": "NONE", "confidence": 0}))
    out = gbif_media.collect_reference_urls(["Not a species"])
    assert out == {"Not a species": []}


def test_failed_request_yields_empty_list_not_an_error(monkeypatch):
    # _request returns None once its retries are exhausted. An annotation run
    # must survive that: a missing reference panel is not a failed run.
    monkeypatch.setattr(gbif_media, "_request", lambda url, params=None: None)
    out = gbif_media.collect_reference_urls(["Vulpes vulpes"])
    assert out == {"Vulpes vulpes": []}


def test_non_still_image_media_is_ignored(monkeypatch):
    def fake_request(url, params=None):
        if url == gbif_media.GBIF_MATCH_URL:
            return _FakeResponse(_match(1))
        return _FakeResponse({"results": [{"lifeStage": "Adult", "media": [
            {"type": "Sound", "identifier": "http://a/call.mp3"},
            {"type": "StillImage", "identifier": "ftp://not-http/x.jpg"},
            {"type": "StillImage", "identifier": "http://a/good.jpg"},
        ]}]})

    monkeypatch.setattr(gbif_media, "_request", fake_request)
    out = gbif_media.collect_reference_urls(["X y"], images_per_species=3)
    assert out == {"X y": [{"src": "http://a/good.jpg", "stage": "Adult"}]}


def test_second_run_is_served_from_cache(monkeypatch, tmp_path):
    calls = []

    def fake_request(url, params=None):
        calls.append(url)
        if url == gbif_media.GBIF_MATCH_URL:
            return _FakeResponse(_match(1))
        return _FakeResponse(_occurrences(("http://a/1.jpg", "Adult")))

    monkeypatch.setattr(gbif_media, "_request", fake_request)
    cache = str(tmp_path)

    first = gbif_media.collect_reference_urls(["Vulpes vulpes"],
                                              images_per_species=1,
                                              cache_dir=cache)
    n_after_first = len(calls)
    second = gbif_media.collect_reference_urls(["Vulpes vulpes"],
                                               images_per_species=1,
                                               cache_dir=cache)

    assert first == second
    assert len(calls) == n_after_first   # second run hit no network at all

    written = json.loads((tmp_path / gbif_media.CACHE_FILENAME).read_text())
    assert written["Vulpes vulpes"]["photos"] == [
        {"src": "http://a/1.jpg", "stage": "Adult"}]


def test_changing_life_stages_invalidates_the_cache(monkeypatch, tmp_path):
    calls = []

    def fake_request(url, params=None):
        calls.append(url)
        if url == gbif_media.GBIF_MATCH_URL:
            return _FakeResponse(_match(1))
        return _FakeResponse(_occurrences(("http://a/1.jpg", "Adult")))

    monkeypatch.setattr(gbif_media, "_request", fake_request)
    cache = str(tmp_path)

    gbif_media.collect_reference_urls(["X y"], images_per_species=1,
                                      cache_dir=cache)
    n = len(calls)
    gbif_media.collect_reference_urls(["X y"], images_per_species=1,
                                      cache_dir=cache,
                                      life_stages=("Adult",))
    assert len(calls) > n


def test_cache_from_the_previous_layout_is_discarded(monkeypatch, tmp_path):
    # Version 1 stored bare URLs with no stage. Reusing those would show
    # photographs the reviewer cannot stage.
    (tmp_path / gbif_media.CACHE_FILENAME).write_text(json.dumps(
        {"X y": {"key": 1, "confidence": 99, "urls": ["http://old/1.jpg"],
                 "requested": 3}}))
    _stub(monkeypatch, _occurrences(("http://new/1.jpg", "Adult")))
    out = gbif_media.collect_reference_urls(["X y"], images_per_species=1,
                                            cache_dir=str(tmp_path))
    assert out["X y"] == [{"src": "http://new/1.jpg", "stage": "Adult"}]


def test_asking_for_more_images_than_cached_requeries(monkeypatch, tmp_path):
    _stub(monkeypatch, _occurrences(("http://a/1.jpg", "Adult"),
                                    ("http://a/2.jpg", "Adult")))
    cache = str(tmp_path)
    gbif_media.collect_reference_urls(["X y"], images_per_species=1,
                                      cache_dir=cache)
    out = gbif_media.collect_reference_urls(["X y"], images_per_species=2,
                                            cache_dir=cache)
    assert len(out["X y"]) == 2


def test_duplicate_names_are_looked_up_once(monkeypatch):
    calls = []

    def fake_request(url, params=None):
        calls.append(url)
        if url == gbif_media.GBIF_MATCH_URL:
            return _FakeResponse(_match(1))
        return _FakeResponse(_occurrences(("http://a/1.jpg", "Adult")))

    monkeypatch.setattr(gbif_media, "_request", fake_request)
    gbif_media.collect_reference_urls(["X y", "X y", "X y"],
                                      images_per_species=1)
    assert calls.count(gbif_media.GBIF_MATCH_URL) == 1


def test_progress_callback_can_cancel(monkeypatch):
    monkeypatch.setattr(gbif_media, "_request",
                        lambda url, params=None: _FakeResponse(
                            {"matchType": "NONE", "confidence": 0}))
    seen = []

    def stop_after_two(done, total, name):
        seen.append(name)
        return done < 2

    out = gbif_media.collect_reference_urls(["a b", "c d", "e f", "g h"],
                                            progress=stop_after_two)
    assert len(seen) == 2
    assert set(out) == {"a b", "c d"}


def test_unreadable_cache_does_not_raise(monkeypatch, tmp_path):
    (tmp_path / gbif_media.CACHE_FILENAME).write_text("{ not json")
    monkeypatch.setattr(gbif_media, "_request",
                        lambda url, params=None: _FakeResponse(
                            {"matchType": "NONE", "confidence": 0}))
    out = gbif_media.collect_reference_urls(["X y"], cache_dir=str(tmp_path))
    assert out == {"X y": []}


def test_report_accepts_both_entry_shapes():
    # GBIF entries carry a stage; dataset reference images are bare paths and
    # must keep working, with no stage asserted for them.
    gbif = _ref_srcs("A b", {"A b": [{"src": "http://a/1.jpg", "stage": "Adult"}]}, {})
    assert gbif == [{"src": "http://a/1.jpg", "stage": "Adult"}]

    plain = _ref_srcs("A b", {"A b": ["http://a/1.jpg"]}, {})
    assert plain == [{"src": "http://a/1.jpg", "stage": ""}]
