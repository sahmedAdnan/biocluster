"""
GBIF reference photographs for the review report.

A reviewer checking a proposed species name needs photographs from outside the
collection, since the collection's own images are what the proposal came from.
This fetches them from GBIF, filtered to the adult stage and captioned with the
stage each record reports.

The photographs support a visual comparison, not a determination: GBIF
occurrence media is contributor-submitted and not necessarily expert-verified.
The name itself is checked against the GBIF backbone taxonomy in taxonomy.py,
which returns a match confidence.

Only URLs are collected, so the report needs a connection to show the panels.
Species names are the only thing sent.
"""

import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import requests

GBIF_MATCH_URL      = "https://api.gbif.org/v1/species/match"
GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"

GBIF_DELAY       = 0.5     # between requests, to stay within GBIF's limits
GBIF_RETRIES     = 3
GBIF_RETRY_DELAY = 10.0

_USER_AGENT = "biocluster/0.1 (https://github.com/sahmedAdnan/biocluster)"

CACHE_FILENAME = "reference_urls.json"
CACHE_VERSION = 2          # 1 stored bare URLs, with no life stage

# Darwin Core carries the insect adult as both Adult and Imago, so both are
# requested. Imago matches nothing for vertebrates, which is harmless.
ADULT_STAGES = ("Adult", "Imago")

# Dropped even when a record carries no stage filter. Caterpillar is a
# separate GBIF vocabulary value from Larva, not a synonym.
IMMATURE_STAGES = frozenset(s.lower() for s in (
    "Larva", "Caterpillar", "Pupa", "Egg", "Immature", "Nymph",
    "Juvenile", "Subadult", "Deutonymph", "Tadpole", "Instar",
    "Seedling", "Vegetative", "Flowering",
))

STAGE_UNRECORDED = "stage not recorded"


def _request(url: str, params: dict = None) -> Optional[requests.Response]:
    """GET with retries. Returns None once the retries are exhausted, because
    a missing reference panel is not a reason to fail an annotation run."""
    for attempt in range(1, GBIF_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=15,
                             headers={"User-Agent": _USER_AGENT})
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            if attempt < GBIF_RETRIES:
                time.sleep(GBIF_RETRY_DELAY)
            else:
                print(f"    [fail] {url}: {e}")
                return None


def resolve_species_key(name: str,
                        kingdom: str = "Animalia") -> Tuple[Optional[int], int]:
    """
    Resolve a species name to its GBIF backbone key.

    Returns (usage_key, confidence). The key is None when GBIF reports no
    match, in which case there is nothing to fetch photographs for. The
    confidence is GBIF's own 0-100 score for the name match, and is worth
    recording: a low score means the reference photographs that follow may be
    of a different species than intended.
    """
    r = _request(GBIF_MATCH_URL,
                 {"name": name, "kingdom": kingdom, "rank": "SPECIES"})
    if r is None:
        return None, 0

    data = r.json()
    if data.get("matchType", "NONE") == "NONE":
        return None, int(data.get("confidence", 0) or 0)

    key = data.get("usageKey")
    return (int(key) if key else None), int(data.get("confidence", 0) or 0)


def _photos_from(response, max_images: int, seen: set,
                 allowed: frozenset = frozenset()) -> List[dict]:
    """Pull {src, stage} entries out of an occurrence-search response,
    dropping immature records. A stage the caller asked for is kept even if it
    is on the immature list, so requesting Juvenile returns juveniles."""
    out = []
    for occ in response.json().get("results", []):
        stage = (occ.get("lifeStage") or "").strip()
        low = stage.lower()
        if low in IMMATURE_STAGES and low not in allowed:
            continue
        for media in occ.get("media", []):
            if media.get("type") != "StillImage":
                continue
            url = (media.get("identifier") or "").strip()
            if not url.startswith("http") or url in seen:
                continue
            seen.add(url)
            out.append({"src": url, "stage": stage or STAGE_UNRECORDED})
            if len(out) >= max_images:
                return out
    return out


def image_urls_for_key(species_key: int, max_images: int = 3,
                       life_stages: Tuple[str, ...] = ADULT_STAGES
                       ) -> List[dict]:
    """
    Photographs for one species key, as {src, stage} entries.

    Asks GBIF for the requested stages first. Repeating the lifeStage
    parameter is an OR, so Adult and Imago come back together. Records of an
    immature stage are the common case in GBIF's media and are not a usable
    reference for an adult, so a second unfiltered request tops up only when
    the filtered one came up short, and immature records are dropped from it.
    """
    limit = max(max_images * 3, 10)
    seen: set = set()
    photos: List[dict] = []
    allowed = frozenset(s.lower() for s in life_stages)

    r = _request(GBIF_OCCURRENCE_URL,
                 {"speciesKey": species_key,
                  "mediaType":  "StillImage",
                  "lifeStage":  list(life_stages),
                  "limit":      limit})
    if r is not None:
        photos = _photos_from(r, max_images, seen, allowed)

    if len(photos) < max_images:
        time.sleep(GBIF_DELAY)
        r = _request(GBIF_OCCURRENCE_URL,
                     {"speciesKey": species_key,
                      "mediaType":  "StillImage",
                      "limit":      limit})
        if r is not None:
            photos += _photos_from(r, max_images - len(photos), seen, allowed)

    return photos[:max_images]


def _load_cache(cache_dir: Optional[str]) -> Tuple[dict, Optional[str]]:
    if not cache_dir:
        return {}, None
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, CACHE_FILENAME)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Entries from an older layout carry no life stage, so they are
            # discarded rather than shown without one.
            return {k: v for k, v in data.items()
                    if isinstance(v, dict) and v.get("v") == CACHE_VERSION}, path
        except (OSError, ValueError, AttributeError):
            print(f"    [warn] unreadable cache at {path}, starting fresh")
    return {}, path


def collect_reference_urls(species_names: List[str],
                           images_per_species: int = 3,
                           cache_dir: Optional[str] = None,
                           kingdom: str = "Animalia",
                           life_stages: Tuple[str, ...] = ADULT_STAGES,
                           progress: Optional[Callable] = None
                           ) -> Dict[str, List[dict]]:
    """
    Collect GBIF photographs for each species name.

    Args:
        species_names:      Names to look up. Normally the names the report
                            will display, which is each cluster's top choice
                            and runner-up rather than the whole candidate list.
        images_per_species: Photographs to keep per species.
        cache_dir:          Where to cache lookups. Two requests per species
                            are needed, so caching matters as soon as a run is
                            repeated. Pass None to disable.
        kingdom:            GBIF kingdom filter for the name match.
        life_stages:        Darwin Core lifeStage values to ask for. The
                            default is the adult, which is what a trap or
                            camera image shows.
        progress:           Called as progress(done, total, name) after each
                            species. Return False to stop early and keep what
                            has been collected so far. Supplying it also
                            silences the per-species output, since a caller
                            with a progress display has nowhere to print.

    Returns:
        Dict mapping species name -> list of {src, stage} entries. A name with
        no match and a name with no photographs both map to an empty list; the
        report shows no reference panel for those.
    """
    cache, cache_path = _load_cache(cache_dir)
    out: Dict[str, List[dict]] = {}
    queried = cached = 0
    quiet = progress is not None
    stage_key = ",".join(life_stages)

    names = [str(n) for n in dict.fromkeys(species_names)
             if n and str(n).strip()]
    if not quiet:
        print(f"[gbif] Reference photographs for {len(names)} species "
              f"({images_per_species} per species, {stage_key})")

    for i, name in enumerate(names, start=1):
        entry = cache.get(name)

        # Reusable when the stages asked for match and it either holds enough
        # photographs or was a complete lookup that found fewer.
        if entry and entry.get("stages") == stage_key and (
                len(entry.get("photos", [])) >= images_per_species
                or entry.get("requested", 0) >= images_per_species):
            out[name] = entry["photos"][:images_per_species]
            cached += 1
        else:
            key, confidence = resolve_species_key(name, kingdom)
            time.sleep(GBIF_DELAY)

            if key is None:
                photos = []
                if not quiet:
                    print(f"    [{i}/{len(names)}] {name}: no GBIF match")
            else:
                photos = image_urls_for_key(key, images_per_species, life_stages)
                time.sleep(GBIF_DELAY)
                if not quiet:
                    note = ("" if confidence >= 70
                            else f", low name match ({confidence})")
                    n_unrec = sum(1 for p in photos
                                  if p["stage"] == STAGE_UNRECORDED)
                    unrec = f", {n_unrec} unrecorded stage" if n_unrec else ""
                    plural = "" if len(photos) == 1 else "s"
                    print(f"    [{i}/{len(names)}] {name}: "
                          f"{len(photos)} photograph{plural}{unrec}{note}")

            cache[name] = {"v": CACHE_VERSION, "key": key,
                           "confidence": confidence, "stages": stage_key,
                           "photos": photos, "requested": images_per_species}
            out[name] = photos
            queried += 1

        if progress is not None and progress(i, len(names), name) is False:
            break

    if cache_path:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except OSError as e:
            print(f"    [warn] could not write cache: {e}")

    if not quiet:
        with_photos = sum(1 for v in out.values() if v)
        print(f"[gbif] {with_photos}/{len(names)} species have photographs "
              f"({queried} queried, {cached} from cache)")
    return out
