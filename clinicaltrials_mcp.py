"""
Clinical Trials MCP Server

Exposes ClinicalTrials.gov and CTIS (euclinicaltrials.eu) data as MCP tools:
  - get_current_trials      : recruiting trials by disease + country
  - get_all_trials          : all trials by disease (optional country)
  - get_specialized_centers : treatment centres ranked by trial count
  - get_available_treatments: interventions from interventional studies
  - check_eligibility       : AI-powered eligibility check via OpenAI

Both ClinicalTrials.gov (NCT IDs) and CTIS (EUCT IDs like 2023-505701-14-00)
are queried. CTIS results are deduplicated against CT.gov using the nct_number
cross-reference field — only trials absent from CT.gov are added.

Supports two transport modes:
  - stdio (default): For local MCP clients
  - sse: For remote HTTP access with Bearer token authentication

Usage:
  python clinicaltrials_mcp.py                          # stdio mode
  python clinicaltrials_mcp.py --transport sse          # SSE on 0.0.0.0:8080
  python clinicaltrials_mcp.py --transport sse --port 9000 --host 127.0.0.1
"""

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_CACHE_FILE = os.path.join(BASE_DIR, "api_cache.json")
RESPONSE_CACHE_FILE = os.path.join(BASE_DIR, "response_cache_mcp.json")
CTIS_CACHE_FILE = os.path.join(BASE_DIR, "ctis_cache.json")
CACHE_TIMEOUT = 86400  # 24 h
API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
SIMILARITY_THRESHOLD = 50

ALLOWED_BEARER_TOKENS = [
    token.strip()
    for token in os.getenv("ALLOWED_API_KEYS", "").split(",")
    if token.strip()
]

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mcp = FastMCP("Clinical Trials")

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def _save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


api_cache: dict = _load_json(API_CACHE_FILE)
response_cache: dict = _load_json(RESPONSE_CACHE_FILE)
ctis_cache: dict = _load_json(CTIS_CACHE_FILE)


def _api_key(disease: str) -> str:
    return hashlib.md5(disease.encode()).hexdigest()


def _response_key(endpoint: str, disease: str, country: str) -> str:
    return hashlib.md5(f"{endpoint}_{disease}_{country}".encode()).hexdigest()


def _is_valid(entry: dict) -> bool:
    return (datetime.now().timestamp() - entry["timestamp"]) < CACHE_TIMEOUT


def _get_api_cache(disease: str) -> Optional[dict]:
    key = _api_key(disease)
    entry = api_cache.get(key)
    if entry and _is_valid(entry):
        return entry["data"]
    return None


def _set_api_cache(disease: str, data: dict) -> None:
    api_cache[_api_key(disease)] = {"timestamp": datetime.now().timestamp(), "data": data}
    _save_json(API_CACHE_FILE, api_cache)


def _get_response_cache(endpoint: str, disease: str, country: str) -> Optional[list]:
    key = _response_key(endpoint, disease, country)
    entry = response_cache.get(key)
    if entry and _is_valid(entry):
        return entry["data"]
    return None


def _set_response_cache(endpoint: str, disease: str, country: str, data: list) -> None:
    response_cache[_response_key(endpoint, disease, country)] = {
        "timestamp": datetime.now().timestamp(),
        "data": data,
    }
    _save_json(RESPONSE_CACHE_FILE, response_cache)


# ---------------------------------------------------------------------------
# ClinicalTrials.gov fetch
# ---------------------------------------------------------------------------

def _fetch_trials(disease: str) -> Optional[dict]:
    """Return {'studies': [...]} for *disease*, using cache when available."""
    cached = _get_api_cache(disease)
    if cached:
        return cached

    all_studies = []
    page_token = None

    while True:
        params = {
            "query.cond": disease,
            "filter.overallStatus": (
                "ACTIVE_NOT_RECRUITING,COMPLETED,ENROLLING_BY_INVITATION,"
                "NOT_YET_RECRUITING,RECRUITING,APPROVED_FOR_MARKETING"
            ),
            "pageSize": 512,
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(API_BASE_URL, params=params, timeout=30)
        if resp.status_code != 200:
            return None

        data = resp.json()
        all_studies.extend(data.get("studies", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    result = {"studies": all_studies}
    _set_api_cache(disease, result)
    return result


# ---------------------------------------------------------------------------
# Shared study simplification (CT.gov)
# ---------------------------------------------------------------------------

def _simplify(study: dict) -> dict:
    ps = study.get("protocolSection", {})
    id_mod = ps.get("identificationModule", {})
    nct_id = id_mod.get("nctId")
    locations = ps.get("contactsLocationsModule", {}).get("locations", [])
    interventions = ps.get("armsInterventionsModule", {}).get("interventions", [])
    return {
        "NCTId": nct_id,
        "BriefTitle": id_mod.get("briefTitle"),
        "StudyUrl": f"https://clinicaltrials.gov/study/{nct_id}",
        "BriefSummary": ps.get("descriptionModule", {}).get("briefSummary"),
        "InterventionType": [i.get("type") for i in interventions],
        "InterventionName": [i.get("name") for i in interventions],
        "CompletionDate": ps.get("statusModule", {}).get("completionDateStruct", {}).get("date"),
        "Locations": [
            {
                "facility": loc.get("facility"),
                "city": loc.get("city"),
                "state": loc.get("state"),
                "country": loc.get("country"),
            }
            for loc in locations
        ],
        "Phases": ps.get("designModule", {}).get("phases", []),
        "StudyType": ps.get("designModule", {}).get("studyType"),
        "EligibilityModule": ps.get("eligibilityModule", {}),
    }


# ---------------------------------------------------------------------------
# CTIS constants and mappings
# ---------------------------------------------------------------------------

_CTIS_SEARCH   = "https://euclinicaltrials.eu/ctis-public-api/search"
_CTIS_RETRIEVE = "https://euclinicaltrials.eu/ctis-public-api/retrieve"
_CTIS_HEADERS  = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (research script)",
    "Origin": "https://euclinicaltrials.eu",
    "Referer": "https://euclinicaltrials.eu/ctis-public/search",
}

_CTIS_STATUS_ALIASES: dict = {
    "ongoing": [2, 3],   # Authorised + Ongoing
    "all":     [1, 2, 3, 4, 5, 6, 7, 8, 9],
}

_CTIS_ISO_TO_COUNTRY: dict = {
    "IT": "Italy",       "DE": "Germany",     "FR": "France",
    "ES": "Spain",       "PL": "Poland",      "NL": "Netherlands",
    "BE": "Belgium",     "AT": "Austria",     "PT": "Portugal",
    "CZ": "Czechia",     "HU": "Hungary",     "RO": "Romania",
    "SE": "Sweden",      "DK": "Denmark",     "NO": "Norway",
    "FI": "Finland",     "GR": "Greece",      "BG": "Bulgaria",
    "HR": "Croatia",     "SK": "Slovakia",    "SI": "Slovenia",
    "LT": "Lithuania",   "LV": "Latvia",      "EE": "Estonia",
    "LU": "Luxembourg",  "MT": "Malta",       "CY": "Cyprus",
    "IE": "Ireland",     "IS": "Iceland",     "LI": "Liechtenstein",
}
_CTIS_COUNTRY_TO_ISO: dict = {v: k for k, v in _CTIS_ISO_TO_COUNTRY.items()}

_EUCT_RE = re.compile(r"^\d{4}-\d{6}-\d{2}-\d{2}$")


def _is_euct_id(trial_id: str) -> bool:
    """Return True if trial_id matches the EUCT format (e.g. '2023-505701-14-00')."""
    return bool(_EUCT_RE.match(trial_id.strip()))


# ---------------------------------------------------------------------------
# CTIS API functions (adapted from ctis_scraper.py, no print statements)
# ---------------------------------------------------------------------------

def _ctis_build_payload(disease: str, status_codes: Optional[list],
                         page: int = 1, page_size: int = 100) -> dict:
    criteria = {k: None for k in [
        "containAll", "containAny", "containNot", "title", "number", "status",
        "medicalCondition", "sponsor", "endPoint", "productName", "productRole",
        "populationType", "orphanDesignation", "msc", "ageGroupCode",
        "therapeuticAreaCode", "trialPhaseCode", "sponsorTypeCode", "gender",
        "protocolCode", "rareDisease", "pip", "haveOrphanDesignation",
        "hasStudyResults", "hasClinicalStudyReport", "isLowIntervention",
        "hasSeriousBreach", "hasUnexpectedEvent", "hasUrgentSafetyMeasure",
        "isTransitioned", "eudraCtCode", "trialRegion", "vulnerablePopulation",
        "mscStatus",
    ]}
    criteria["containAll"] = disease
    criteria["status"] = status_codes
    return {
        "pagination": {"page": page, "size": page_size},
        "sort": {"property": "decisionDate", "direction": "DESC"},
        "searchCriteria": criteria,
    }


def _ctis_search_overview(disease: str, status_codes: Optional[list],
                           max_results: int = 200) -> list:
    """Paginated CTIS search; returns list of overview dicts."""
    all_trials: list = []
    page = 1
    total_pages = 1

    while page <= total_pages and len(all_trials) < max_results:
        payload = _ctis_build_payload(
            disease, status_codes, page=page,
            page_size=min(100, max_results - len(all_trials)),
        )
        try:
            r = requests.post(_CTIS_SEARCH, headers=_CTIS_HEADERS,
                              json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException:
            break

        total_pages = data.get("pagination", {}).get("totalPages", 1)
        batch = data.get("data", [])
        if not batch:
            break
        all_trials.extend(batch)
        page += 1
        if page <= total_pages:
            time.sleep(0.5)

    return all_trials[:max_results]


def _ctis_get_detail(euct_code: str) -> Optional[dict]:
    """Fetch full CTIS trial detail JSON, with per-trial caching."""
    cache_key = f"ctis_detail_{hashlib.md5(euct_code.encode()).hexdigest()}"
    entry = ctis_cache.get(cache_key)
    if entry and _is_valid(entry):
        return entry["data"]

    try:
        r = requests.get(f"{_CTIS_RETRIEVE}/{euct_code}",
                         headers=_CTIS_HEADERS, timeout=30)
        r.raise_for_status()
        raw = r.json()
    except requests.RequestException:
        return None

    ctis_cache[cache_key] = {"timestamp": datetime.now().timestamp(), "data": raw}
    _save_json(CTIS_CACHE_FILE, ctis_cache)
    return raw


def _ctis_find_nct(raw: dict, other_ids: dict) -> str:
    """Search for an NCT cross-reference number inside a CTIS trial JSON."""
    for c in [other_ids.get("nctNumber", ""), other_ids.get("NCTNumber", ""),
               raw.get("nctNumber", ""), raw.get("NCTId", "")]:
        if c and re.match(r"NCT\d{8}", str(c), re.I):
            return c.upper()
    matches = re.findall(r"NCT\d{8}", json.dumps(raw), re.I)
    return matches[0].upper() if matches else ""


def _ctis_parse_trial(raw: dict) -> dict:
    """Normalize a raw CTIS trial JSON into a flat dict."""
    part1 = raw.get("authorizedApplication", {}).get("authorizedPartI", {})
    sci   = part1.get("trialDetails", {})
    proto = part1.get("protocolInformation", {})
    pop   = part1.get("populationDetails", {})
    elig  = part1.get("eligibilityCriteria", {})

    euct = raw.get("ctNumber", "")
    result = {
        "euct_number":        euct,
        "ctis_status":        raw.get("ctStatus", ""),
        "ctis_url":           f"https://euclinicaltrials.eu/ctis-public/search#{euct}",
        "title":              sci.get("fullTitle") or raw.get("ctTitle", ""),
        "short_title":        sci.get("shortTitle") or raw.get("shortTitle", ""),
        "lay_summary":        sci.get("laySummary", ""),
        "primary_objective":  sci.get("primaryObjective", ""),
        "trial_phase":        raw.get("trialPhase") or proto.get("trialPhaseName", ""),
        "trial_type":         proto.get("trialTypeName", ""),
        "gender":             raw.get("gender", ""),
        "min_age_months":     pop.get("minAgeValue"),
        "max_age_months":     pop.get("maxAgeValue"),
        "inclusion_criteria": elig.get("inclusionCriteria", ""),
        "exclusion_criteria": elig.get("exclusionCriteria", ""),
        "sponsor":            raw.get("sponsor", ""),
        "trial_countries":    raw.get("trialCountries", []),
    }

    products = part1.get("products", [])
    result["interventions"] = [
        {
            "name": (p.get("productDictionaryInfo", {}).get("prodName")
                     or p.get("productName", "")),
            "active_substance": p.get("productDictionaryInfo", {}).get("activeSubstanceName", ""),
        }
        for p in products
        if p.get("productDictionaryInfo", {}).get("prodName") or p.get("productName")
    ]

    mscs = raw.get("authorizedApplication", {}).get("memberStateConcerned", [])
    sites = []
    for msc in mscs:
        cc = msc.get("mscCode", "")
        for site in msc.get("trialSites", []):
            sites.append({
                "country":     cc,
                "name":        site.get("siteName", ""),
                "city":        site.get("city", ""),
                "institution": site.get("institution", ""),
                "pi":          site.get("principalInvestigator", ""),
            })
    result["sites"]   = sites
    result["n_sites"] = len(sites)

    other_ids = part1.get("otherIdentifiers", {})
    result["nct_number"] = _ctis_find_nct(raw, other_ids)
    result["eudract"]    = other_ids.get("eudraCtNumber", "")

    return {k: v for k, v in result.items()
            if v is not None and v != "" and v != [] and v != {}}


def _ctis_age_str(months: Optional[int]) -> str:
    """Convert CTIS month-based age to a human-readable string."""
    if months is None:
        return "not specified"
    if months % 12 == 0:
        years = months // 12
        return f"{years} Year{'s' if years != 1 else ''}"
    return f"{months} Month{'s' if months != 1 else ''}"


# ---------------------------------------------------------------------------
# CTIS high-level helpers
# ---------------------------------------------------------------------------

def _ctis_fetch_parsed(disease: str, ctis_status: str = "all",
                        max_results: int = 150) -> list:
    """
    Fetch, parse, and cache CTIS trials for *disease*.
    Each trial detail is cached individually in ctis_cache.json.
    The full parsed list is cached per (disease, status) combination.
    Returns list of parsed trial dicts.
    """
    list_key = f"ctis_list_{hashlib.md5(f'{disease}_{ctis_status}'.encode()).hexdigest()}"
    entry = ctis_cache.get(list_key)
    if entry and _is_valid(entry):
        return entry["data"]

    status_codes = _CTIS_STATUS_ALIASES.get(ctis_status.lower())
    overviews = _ctis_search_overview(disease, status_codes, max_results=max_results)

    results = []
    for ov in overviews:
        euct = ov.get("ctNumber", "")
        if not euct:
            continue
        raw = _ctis_get_detail(euct)
        if raw:
            try:
                results.append(_ctis_parse_trial(raw))
            except Exception:
                pass
        time.sleep(0.3)

    ctis_cache[list_key] = {"timestamp": datetime.now().timestamp(), "data": results}
    _save_json(CTIS_CACHE_FILE, ctis_cache)
    return results


def _ctis_normalize(parsed: dict) -> dict:
    """Convert a parsed CTIS trial to the CT.gov simplified study format."""
    euct = parsed.get("euct_number", "")
    interventions = parsed.get("interventions", [])
    sites = parsed.get("sites", [])

    eligibility: dict = {
        "minimumAge": _ctis_age_str(parsed.get("min_age_months")),
        "maximumAge": _ctis_age_str(parsed.get("max_age_months")),
        "gender":     parsed.get("gender", "All"),
    }
    parts = []
    if parsed.get("inclusion_criteria"):
        parts.append(f"Inclusion Criteria:\n{parsed['inclusion_criteria']}")
    if parsed.get("exclusion_criteria"):
        parts.append(f"Exclusion Criteria:\n{parsed['exclusion_criteria']}")
    if parts:
        eligibility["criteria"] = "\n\n".join(parts)

    return {
        "NCTId": euct,
        "BriefTitle": parsed.get("title") or parsed.get("short_title", ""),
        "StudyUrl": parsed.get("ctis_url",
                               f"https://euclinicaltrials.eu/ctis-public/search#{euct}"),
        "BriefSummary": parsed.get("lay_summary") or parsed.get("primary_objective", ""),
        "InterventionType": ["DRUG"] * len(interventions),
        "InterventionName": [i.get("name", "") for i in interventions],
        "CompletionDate": None,
        "Locations": [
            {
                "facility": s.get("name") or s.get("institution", ""),
                "city":     s.get("city", ""),
                "state":    None,
                "country":  _CTIS_ISO_TO_COUNTRY.get(s.get("country", ""), s.get("country", "")),
            }
            for s in sites
        ],
        "Phases":          [parsed["trial_phase"]] if parsed.get("trial_phase") else [],
        "StudyType":       "INTERVENTIONAL",
        "EligibilityModule": eligibility,
        "_source":         "ctis",
    }


def _ctis_dedup(ctgov_results: list, ctis_parsed: list) -> list:
    """Return only CTIS parsed trials not already present in *ctgov_results*."""
    ctgov_ids = {
        t["NCTId"].upper()
        for t in ctgov_results
        if t.get("NCTId") and not _is_euct_id(str(t["NCTId"]))
    }
    return [
        t for t in ctis_parsed
        if not (t.get("nct_number") and t["nct_number"].upper() in ctgov_ids)
    ]


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_current_trials(disease: str, country: str) -> dict:
    """
    Return clinical trials with RECRUITING status for *disease* that have at
    least one active site in *country*.  Results include both ClinicalTrials.gov
    and CTIS (euclinicaltrials.eu) trials; CTIS trials already registered on
    ClinicalTrials.gov are not duplicated.

    Args:
        disease: Disease or condition name (e.g. "lung cancer").
        country: Full country name as used by ClinicalTrials.gov (e.g. "Italy").

    Returns:
        {"trials": [...]} where each trial contains NCTId, BriefTitle,
        StudyUrl, BriefSummary, InterventionType, InterventionName,
        CompletionDate, Locations, Phases, StudyType, EligibilityModule.
        CTIS-only trials also carry "_source": "ctis".
    """
    endpoint = "current_trials"
    cached = _get_response_cache(endpoint, disease, country)
    if cached:
        return {"trials": cached}

    trial_data = _fetch_trials(disease)
    if not trial_data:
        return {"error": "No data found for the requested disease"}

    result = []
    for study in trial_data.get("studies", []):
        try:
            ps = study.get("protocolSection", {})
            status = ps.get("statusModule", {}).get("overallStatus")
            if status != "RECRUITING":
                continue
            locations = ps.get("contactsLocationsModule", {}).get("locations", [])
            if any(loc.get("country") == country for loc in locations):
                result.append(_simplify(study))
        except Exception as exc:
            print(f"[get_current_trials] study processing error: {exc}")

    # CTIS supplemental trials (status "ongoing" ≈ RECRUITING)
    try:
        ctis_parsed = _ctis_fetch_parsed(disease, "ongoing")
        ctis_unique = _ctis_dedup(result, ctis_parsed)
        country_iso = _CTIS_COUNTRY_TO_ISO.get(country)
        if country_iso:
            ctis_unique = [
                t for t in ctis_unique
                if any(s.get("country") == country_iso for s in t.get("sites", []))
            ]
        result.extend(_ctis_normalize(t) for t in ctis_unique)
    except Exception as exc:
        print(f"[get_current_trials] CTIS error: {exc}")

    _set_response_cache(endpoint, disease, country, result)
    return {"trials": result}


@mcp.tool()
def get_all_trials(disease: str, country: Optional[str] = None) -> dict:
    """
    Return all trials (any status) for *disease*, optionally filtered by
    *country*.  Results include both ClinicalTrials.gov and CTIS trials;
    CTIS trials already on ClinicalTrials.gov are not duplicated.

    Args:
        disease: Disease or condition name (e.g. "breast cancer").
        country: Optional country filter. If omitted, all countries are returned.

    Returns:
        {"trials": [...]} — same structure as get_current_trials.
    """
    endpoint = "all_trials"
    cache_country = country or "None"
    cached = _get_response_cache(endpoint, disease, cache_country)
    if cached:
        return {"trials": cached}

    trial_data = _fetch_trials(disease)
    if not trial_data:
        return {"error": "No data found for the requested disease"}

    result = []
    for study in trial_data.get("studies", []):
        try:
            locations = (
                study.get("protocolSection", {})
                .get("contactsLocationsModule", {})
                .get("locations", [])
            )
            if country and not any(loc.get("country") == country for loc in locations):
                continue
            result.append(_simplify(study))
        except Exception as exc:
            print(f"[get_all_trials] study processing error: {exc}")

    # CTIS supplemental trials (all statuses)
    try:
        ctis_parsed = _ctis_fetch_parsed(disease, "all")
        ctis_unique = _ctis_dedup(result, ctis_parsed)
        if country:
            country_iso = _CTIS_COUNTRY_TO_ISO.get(country)
            if country_iso:
                ctis_unique = [
                    t for t in ctis_unique
                    if any(s.get("country") == country_iso for s in t.get("sites", []))
                ]
            else:
                ctis_unique = []
        result.extend(_ctis_normalize(t) for t in ctis_unique)
    except Exception as exc:
        print(f"[get_all_trials] CTIS error: {exc}")

    _set_response_cache(endpoint, disease, cache_country, result)
    return {"trials": result}


@mcp.tool()
def get_specialized_centers(disease: str, country: str) -> dict:
    """
    Return treatment centres in *country* that participate in trials for
    *disease*, ranked by trial count.  Only centres with more than 4 trials
    are included.  Similar facility names are deduplicated via fuzzy matching.
    Includes sites from both ClinicalTrials.gov and CTIS.

    Args:
        disease: Disease or condition name.
        country: Country to filter centres by.

    Returns:
        {"centers": [{"facility", "city", "trialCount", "interventions"}, ...]}
        sorted descending by trialCount.
    """
    endpoint = "specialized_centers"
    cached = _get_response_cache(endpoint, disease, country)
    if cached:
        return {"centers": cached}

    trial_data = _fetch_trials(disease)
    if not trial_data:
        return {"error": "No data found for the requested disease"}

    centers: dict = {}
    ctgov_nct_ids: set = set()

    for study in trial_data.get("studies", []):
        try:
            ps = study.get("protocolSection", {})
            nct = ps.get("identificationModule", {}).get("nctId", "")
            if nct:
                ctgov_nct_ids.add(nct.upper())
            locations = ps.get("contactsLocationsModule", {}).get("locations", [])
            interventions = [
                i.get("name")
                for i in ps.get("armsInterventionsModule", {}).get("interventions", [])
            ]
            for loc in locations:
                if loc.get("country") != country:
                    continue
                facility = loc.get("facility")
                city = loc.get("city")
                matched = next(
                    (k for k in centers if fuzz.ratio(facility, k) >= SIMILARITY_THRESHOLD),
                    None,
                )
                if matched:
                    facility = matched
                else:
                    centers[facility] = {"city": city, "count": 0, "interventions": set()}
                centers[facility]["count"] += 1
                centers[facility]["interventions"].update(interventions)
        except Exception as exc:
            print(f"[get_specialized_centers] study processing error: {exc}")

    # CTIS supplemental sites
    try:
        country_iso = _CTIS_COUNTRY_TO_ISO.get(country)
        ctis_parsed = _ctis_fetch_parsed(disease, "all")
        ctis_unique = [
            t for t in ctis_parsed
            if not (t.get("nct_number") and t["nct_number"].upper() in ctgov_nct_ids)
        ]
        for trial in ctis_unique:
            for site in trial.get("sites", []):
                if country_iso and site.get("country") != country_iso:
                    continue
                facility = site.get("name") or site.get("institution", "")
                if not facility:
                    continue
                city = site.get("city", "")
                intervention_names = [i.get("name", "") for i in trial.get("interventions", [])]
                matched = next(
                    (k for k in centers if fuzz.ratio(facility, k) >= SIMILARITY_THRESHOLD),
                    None,
                )
                if matched:
                    facility = matched
                else:
                    centers[facility] = {"city": city, "count": 0, "interventions": set()}
                centers[facility]["count"] += 1
                centers[facility]["interventions"].update(intervention_names)
    except Exception as exc:
        print(f"[get_specialized_centers] CTIS error: {exc}")

    result = sorted(
        [
            {
                "facility": fac,
                "city": data["city"],
                "trialCount": data["count"],
                "interventions": list(data["interventions"]),
            }
            for fac, data in centers.items()
            if data["count"] > 4
        ],
        key=lambda x: x["trialCount"],
        reverse=True,
    )

    _set_response_cache(endpoint, disease, country, result)
    return {"centers": result}


@mcp.tool()
def get_available_treatments(disease: str) -> dict:
    """
    Return interventions investigated in trials for *disease*.  Only
    interventions that appear in at least 2 trials are included.  Similar
    names are deduplicated via fuzzy matching.  Includes interventions from
    both ClinicalTrials.gov (interventional studies only) and CTIS (all trials
    are interventional by definition — they are IMP trials).

    Args:
        disease: Disease or condition name.

    Returns:
        {"treatments": [{"interventionName", "interventionType", "trialCount"}, ...]}
        sorted descending by trialCount.
    """
    endpoint = "available_treatments"
    cached = _get_response_cache(endpoint, disease, "None")
    if cached:
        return {"treatments": cached}

    trial_data = _fetch_trials(disease)
    if not trial_data:
        return {"error": "No data found for the requested disease"}

    treatments: dict = {}
    ctgov_nct_ids: set = set()

    for study in trial_data.get("studies", []):
        ps = study.get("protocolSection", {})
        nct = ps.get("identificationModule", {}).get("nctId", "")
        if nct:
            ctgov_nct_ids.add(nct.upper())
        if ps.get("designModule", {}).get("studyType") != "INTERVENTIONAL":
            continue
        try:
            for intervention in ps.get("armsInterventionsModule", {}).get("interventions", []):
                name = intervention.get("name")
                itype = intervention.get("type")
                matched = next(
                    (k for k in treatments if fuzz.ratio(name, k) >= SIMILARITY_THRESHOLD),
                    None,
                )
                if matched:
                    name = matched
                else:
                    treatments[name] = {"type": itype, "count": 0}
                treatments[name]["count"] += 1
        except Exception as exc:
            print(f"[get_available_treatments] study processing error: {exc}")

    # CTIS supplemental interventions (all CTIS trials are IMP trials)
    try:
        ctis_parsed = _ctis_fetch_parsed(disease, "all")
        ctis_unique = [
            t for t in ctis_parsed
            if not (t.get("nct_number") and t["nct_number"].upper() in ctgov_nct_ids)
        ]
        for trial in ctis_unique:
            for intervention in trial.get("interventions", []):
                name = intervention.get("name")
                if not name:
                    continue
                matched = next(
                    (k for k in treatments if fuzz.ratio(name, k) >= SIMILARITY_THRESHOLD),
                    None,
                )
                if matched:
                    name = matched
                else:
                    treatments[name] = {"type": "DRUG", "count": 0}
                treatments[name]["count"] += 1
    except Exception as exc:
        print(f"[get_available_treatments] CTIS error: {exc}")

    result = sorted(
        [
            {
                "interventionName": name,
                "interventionType": data["type"],
                "trialCount": data["count"],
            }
            for name, data in treatments.items()
            if data["count"] >= 2
        ],
        key=lambda x: x["trialCount"],
        reverse=True,
    )

    _set_response_cache(endpoint, disease, "None", result)
    return {"treatments": result}


def _split_eligibility_criteria(criteria_text: str) -> tuple[str, str]:
    """Split raw criteria text into inclusion and exclusion lists."""
    inclusion, exclusion = "", ""
    if "Inclusion Criteria:" in criteria_text:
        parts = criteria_text.split("Exclusion Criteria:")
        inclusion = parts[0].replace("Inclusion Criteria:", "").strip()
        exclusion = parts[1].strip() if len(parts) > 1 else ""
    else:
        inclusion = criteria_text.strip()
    return inclusion, exclusion


@mcp.tool()
def check_eligibility(nct_id: str, disease: str, patient_info: str) -> dict:
    """
    Use OpenAI o3-mini to evaluate whether a patient is eligible for a
    specific clinical trial.

    Args:
        nct_id: NCT identifier (e.g. "NCT04512345") for ClinicalTrials.gov trials,
                or EUCT identifier (e.g. "2023-505701-14-00") for CTIS trials.
        disease: Disease used to look up the trial in the cache (required for
                 ClinicalTrials.gov trials; used as context for CTIS trials).
        patient_info: Free-text description of the patient's demographics,
                      medical history, current medications, and relevant
                      clinical parameters.

    Returns:
        {"nctId": ..., "eligibility": {"result": "yes"|"no"|"unknown",
                                        "explanation": "...",
                                        "inclusion_criteria_met": [...],
                                        "exclusion_criteria_violated": [...],
                                        "uncertain_criteria": [...]}}
    """
    # ── CTIS path (EUCT ID) ──────────────────────────────────────────────────
    if _is_euct_id(nct_id):
        raw = _ctis_get_detail(nct_id)
        if not raw:
            return {"error": f"CTIS trial {nct_id} not found"}
        parsed = _ctis_parse_trial(raw)
        inclusion_criteria = parsed.get("inclusion_criteria", "")
        exclusion_criteria = parsed.get("exclusion_criteria", "")
        min_age      = _ctis_age_str(parsed.get("min_age_months"))
        max_age      = _ctis_age_str(parsed.get("max_age_months"))
        gender       = parsed.get("gender", "not specified")
        std_ages: list = []
        brief_summary = parsed.get("lay_summary") or parsed.get("primary_objective", "")

    # ── ClinicalTrials.gov path (NCT ID) ─────────────────────────────────────
    else:
        trial_data = _fetch_trials(disease)
        if not trial_data:
            return {"error": "No trial data found for the specified disease"}

        trial = next(
            (
                s
                for s in trial_data.get("studies", [])
                if s.get("protocolSection", {})
                .get("identificationModule", {})
                .get("nctId")
                == nct_id
            ),
            None,
        )
        if not trial:
            return {"error": f"Trial {nct_id} not found"}

        ps = trial.get("protocolSection", {})
        eligibility_module = ps.get("eligibilityModule", {})
        criteria_text  = eligibility_module.get("criteria", "No criteria provided")
        min_age        = eligibility_module.get("minimumAge", "not specified")
        max_age        = eligibility_module.get("maximumAge", "not specified")
        gender         = eligibility_module.get("gender", "not specified")
        std_ages       = eligibility_module.get("stdAges", [])
        brief_summary  = ps.get("descriptionModule", {}).get("briefSummary", "")
        inclusion_criteria, exclusion_criteria = _split_eligibility_criteria(criteria_text)

    # ── Common prompt and OpenAI call ────────────────────────────────────────
    prompt = (
        f"PATIENT INFORMATION:\n{patient_info}\n\n"
        f"TRIAL SUMMARY:\n{brief_summary}\n\n"
        f"AGE REQUIREMENTS: minimum {min_age}, maximum {max_age}"
        + (f", standard age groups: {', '.join(std_ages)}" if std_ages else "") + "\n"
        f"GENDER REQUIREMENT: {gender}\n\n"
        f"INCLUSION CRITERIA:\n{inclusion_criteria if inclusion_criteria else 'Not specified'}\n\n"
        f"EXCLUSION CRITERIA:\n{exclusion_criteria if exclusion_criteria else 'None specified'}\n\n"
        "Evaluate the patient's eligibility for this clinical trial by analyzing each criterion independently.\n"
        "Rules:\n"
        "- A patient is eligible only if ALL inclusion criteria are clearly met AND no exclusion criterion is violated.\n"
        "- If a criterion cannot be evaluated due to missing patient data, mark it as 'uncertain'.\n"
        "- The overall result is 'yes' only if there are no violated exclusion criteria and no uncertain inclusion criteria.\n"
        "- The overall result is 'no' if any exclusion criterion is violated or any inclusion criterion is clearly not met.\n"
        "- The overall result is 'unknown' if no criterion is violated but some inclusion criteria are uncertain.\n\n"
        "Respond with a JSON object with these keys:\n"
        "  'result': one of 'yes', 'no', or 'unknown'\n"
        "  'explanation': a concise overall motivation\n"
        "  'inclusion_criteria_met': list of inclusion criteria the patient clearly satisfies\n"
        "  'exclusion_criteria_violated': list of exclusion criteria the patient violates\n"
        "  'uncertain_criteria': list of criteria that could not be evaluated due to missing patient data"
    )

    try:
        response = openai_client.chat.completions.create(
            model="o3-mini",
            reasoning_effort="medium",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical trial eligibility specialist. "
                        "Analyze each inclusion and exclusion criterion independently against the patient profile. "
                        "If a criterion cannot be evaluated due to missing patient data, mark it as uncertain. "
                        "A patient is eligible only if ALL inclusion criteria are clearly met and NO exclusion criteria are violated. "
                        "Respond strictly in JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content
    except Exception as exc:
        return {"error": f"OpenAI API error: {exc}"}

    try:
        eligibility_result = json.loads(answer)
        if "result" not in eligibility_result or "explanation" not in eligibility_result:
            raise ValueError("Missing expected keys in OpenAI response")
    except Exception as exc:
        return {
            "error": "Failed to parse OpenAI response",
            "raw_response": answer,
            "parse_error": str(exc),
        }

    return {"nctId": nct_id, "eligibility": eligibility_result}


# ---------------------------------------------------------------------------
# Bearer Authentication Middleware (for SSE transport)
# ---------------------------------------------------------------------------

def _create_authenticated_sse_app(host: str, port: int):
    """Create an SSE server with Bearer token authentication."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Route, Mount
    from mcp.server.sse import SseServerTransport

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path == "/health":
                return await call_next(request)

            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    {"error": "Missing or invalid Authorization header. Use: Bearer <token>"},
                    status_code=401,
                )

            token = auth_header.replace("Bearer ", "").strip()
            if not ALLOWED_BEARER_TOKENS:
                return JSONResponse(
                    {"error": "No tokens configured. Set ALLOWED_API_KEYS in .env"},
                    status_code=500,
                )

            if token not in ALLOWED_BEARER_TOKENS:
                return JSONResponse(
                    {"error": "Invalid Bearer token"},
                    status_code=401,
                )

            return await call_next(request)

    async def health_check(request):
        return JSONResponse({"status": "ok", "service": "Clinical Trials MCP"})

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0], streams[1], mcp._mcp_server.create_initialization_options()
            )

    app = Starlette(
        routes=[
            Route("/health", health_check, methods=["GET"]),
            Route("/sse", handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ],
        middleware=[Middleware(BearerAuthMiddleware)],
    )

    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clinical Trials MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clinicaltrials_mcp.py                           # stdio mode (local)
  python clinicaltrials_mcp.py --transport sse           # SSE on 0.0.0.0:8080
  python clinicaltrials_mcp.py -t sse -p 9000 -H 127.0.0.1

For SSE mode, authenticate with:
  Authorization: Bearer <your-api-key>

The Bearer tokens are read from the ALLOWED_API_KEYS environment variable.
        """,
    )
    parser.add_argument(
        "-t", "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (default) or sse (HTTP/SSE for remote access)",
    )
    parser.add_argument(
        "-H", "--host",
        default="0.0.0.0",
        help="Host to bind to in SSE mode (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8080,
        help="Port to bind to in SSE mode (default: 8080)",
    )

    args = parser.parse_args()

    if args.transport == "sse":
        print(f"Starting MCP server in SSE mode on http://{args.host}:{args.port}")
        print("Bearer authentication is ENABLED")
        print(f"Configured tokens: {len(ALLOWED_BEARER_TOKENS)} token(s) from ALLOWED_API_KEYS")
        print("\nEndpoints:")
        print(f"  SSE:    http://{args.host}:{args.port}/sse")
        print(f"  Health: http://{args.host}:{args.port}/health")
        print("\nUse header: Authorization: Bearer <your-token>")
        print("-" * 50)

        _create_authenticated_sse_app(args.host, args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
