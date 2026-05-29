from flask import Flask, request, jsonify
import requests
import pandas as pd
from fuzzywuzzy import fuzz
import hashlib
import json
import os
import re
import time
from datetime import datetime
from dotenv import load_dotenv
from io import StringIO
import argparse
import openai  # This line is required to use the openai module
from openai import OpenAI
from typing import Optional

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Import ChatCompletion directly per migration guidelines

app = Flask(__name__)

# Caching setup
API_CACHE_FILE = 'api_cache.json'  # Cache for API calls to clinicaltrials.gov
RESPONSE_CACHE_FILE = 'response_cache_flask.json'  # Cache for responses to microservice endpoints
CTIS_CACHE_FILE = 'ctis_cache.json'  # Cache for CTIS API calls
CACHE_TIMEOUT = 86400  # Cache expiry time in seconds (24 hours)

# Load caches if they exist
if os.path.exists(API_CACHE_FILE):
    with open(API_CACHE_FILE, 'r') as f:
        api_cache = json.load(f)
else:
    api_cache = {}

if os.path.exists(RESPONSE_CACHE_FILE):
    with open(RESPONSE_CACHE_FILE, 'r') as f:
        response_cache = json.load(f)
else:
    response_cache = {}

if os.path.exists(CTIS_CACHE_FILE):
    with open(CTIS_CACHE_FILE, 'r') as f:
        try:
            ctis_cache = json.load(f)
        except json.JSONDecodeError:
            ctis_cache = {}
else:
    ctis_cache = {}

API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Load allowed API keys from .env file
ALLOWED_API_KEYS = os.getenv('ALLOWED_API_KEYS', '').split(',')


def get_api_cache_key(disease):
    return hashlib.md5(disease.encode('utf-8')).hexdigest()

def get_response_cache_key(endpoint, disease, country):
    return hashlib.md5(f"{endpoint}_{disease}_{country}".encode('utf-8')).hexdigest()

def is_cache_valid(entry):
    return (datetime.now().timestamp() - entry['timestamp']) < CACHE_TIMEOUT

def get_cached_api_data(disease):
    cache_key = get_api_cache_key(disease)
    if cache_key in api_cache and is_cache_valid(api_cache[cache_key]):
        return api_cache[cache_key]['data']
    return None

def set_api_cache_data(disease, data):
    cache_key = get_api_cache_key(disease)
    api_cache[cache_key] = {
        'timestamp': datetime.now().timestamp(),
        'data': data
    }
    with open(API_CACHE_FILE, 'w') as f:
        json.dump(api_cache, f)

def get_trial_data(disease):
    cached_data = get_cached_api_data(disease)
    if cached_data:
        print(f"Getting from cache disease {disease}, trials {len(cached_data['studies'])}")
        return cached_data

    # Pagination parameters
    page_size = 512
    page_token = None
    all_studies = []

    # Fetching data with pagination
    while True:
        params = {
            "query.cond": disease,
            "filter.overallStatus": 'ACTIVE_NOT_RECRUITING,COMPLETED,ENROLLING_BY_INVITATION,NOT_YET_RECRUITING,RECRUITING,APPROVED_FOR_MARKETING',
            "pageSize": page_size
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(API_BASE_URL, params=params)
        if response.status_code != 200:
            return None

        data = response.json()
        studies = data.get('studies', [])
        all_studies.extend(studies)

        # Check if there's a next page
        page_token = data.get('nextPageToken')
        if not page_token:
            break

    print(f"Getting from API Call disease {disease}, trials {len(all_studies)}")
    # Cache the fetched data
    set_api_cache_data(disease, {'studies': all_studies})
    return {'studies': all_studies}


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
    if entry and is_cache_valid(entry):
        return entry["data"]

    try:
        r = requests.get(f"{_CTIS_RETRIEVE}/{euct_code}",
                         headers=_CTIS_HEADERS, timeout=30)
        r.raise_for_status()
        raw = r.json()
    except requests.RequestException:
        return None

    ctis_cache[cache_key] = {"timestamp": datetime.now().timestamp(), "data": raw}
    with open(CTIS_CACHE_FILE, 'w') as f:
        json.dump(ctis_cache, f)
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
    Each trial detail is cached individually; the full parsed list is cached
    per (disease, status) combination. Returns list of parsed trial dicts.
    """
    list_key = f"ctis_list_{hashlib.md5(f'{disease}_{ctis_status}'.encode()).hexdigest()}"
    entry = ctis_cache.get(list_key)
    if entry and is_cache_valid(entry):
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
    with open(CTIS_CACHE_FILE, 'w') as f:
        json.dump(ctis_cache, f)
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


@app.route('/current_trials', methods=['GET'])
def current_trials():
    api_key = request.headers.get('x-api-key')
    if api_key not in ALLOWED_API_KEYS:
        return jsonify({'error': 'Unauthorized'}), 401

    disease = request.args.get('disease')
    country = request.args.get('country')
    endpoint = 'current_trials'

    # Check response cache
    cached_response = get_cached_response(endpoint, disease, country)
    if cached_response and len(cached_response)>0:
        return jsonify({'trials': cached_response})

    # Get trial data
    trial_data = get_trial_data(disease)
    if not trial_data:
        return jsonify({'error': 'No data found'}), 404

    valid_statuses = ["RECRUITING"]
    current_trials_list = []
    for study in trial_data.get('studies', []):
        try:
            overall_status = study.get('protocolSection', {}).get('statusModule', {}).get('overallStatus')
            if overall_status in valid_statuses:
                locations = study.get('protocolSection', {}).get('contactsLocationsModule', {}).get('locations', [])
                for location in locations:
                    if location.get('country') == country:
                        simplified_study = {
                            'NCTId': study.get('protocolSection', {}).get('identificationModule', {}).get('nctId'),
                            'BriefTitle': study.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle'),
                            'StudyUrl': f"https://clinicaltrials.gov/study/{study.get('protocolSection', {}).get('identificationModule', {}).get('nctId')}",
                            'BriefSummary': study.get('protocolSection', {}).get('descriptionModule', {}).get('briefSummary'),
                            'InterventionType': [intervention.get('type') for intervention in study.get('protocolSection', {}).get('armsInterventionsModule', {}).get('interventions', [])],
                            'InterventionName': [intervention.get('name') for intervention in study.get('protocolSection', {}).get('armsInterventionsModule', {}).get('interventions', [])],
                            'CompletionDate': study.get('protocolSection', {}).get('statusModule', {}).get('completionDateStruct', {}).get('date'),
                            'Locations': [{'facility': loc.get('facility'), 'city': loc.get('city'), 'state': loc.get('state'), 'country': loc.get('country')} for loc in locations],
                            'Phases': study.get('protocolSection', {}).get('designModule', {}).get('phases', []),
                            'StudyType': study.get('protocolSection', {}).get('designModule', {}).get('studyType'),
                            'EligibilityModule': study.get('protocolSection', {}).get('eligibilityModule', {})
                        }
                        current_trials_list.append(simplified_study)
                        break
        except Exception as e:
            print(f"Error processing study {study}: {e}")

    # CTIS supplemental trials (status "ongoing" ≈ RECRUITING)
    try:
        ctis_parsed = _ctis_fetch_parsed(disease, "ongoing")
        ctis_unique = _ctis_dedup(current_trials_list, ctis_parsed)
        country_iso = _CTIS_COUNTRY_TO_ISO.get(country)
        if country_iso:
            ctis_unique = [
                t for t in ctis_unique
                if any(s.get("country") == country_iso for s in t.get("sites", []))
            ]
        current_trials_list.extend(_ctis_normalize(t) for t in ctis_unique)
    except Exception as e:
        print(f"[CTIS] current_trials error: {e}")

    # Cache the response
    set_response_cache_response(endpoint, disease, country, current_trials_list)

    print(f"Current trials for {disease} in {country} count: {len(current_trials_list)}")

    return jsonify({'trials': current_trials_list})

@app.route('/all_trials', methods=['GET'])
def all_trials():
    api_key = request.headers.get('x-api-key')
    if api_key not in ALLOWED_API_KEYS:
        return jsonify({'error': 'Unauthorized'}), 401

    disease = request.args.get('disease')
    if not disease:
        return jsonify({'error': 'Disease parameter is required'}), 400

    country = request.args.get('country')
    cache_country = country if country else 'None'
    endpoint = 'all_trials'

    cached_response = get_cached_response(endpoint, disease, cache_country)
    if cached_response and len(cached_response) > 0:
        return jsonify({'trials': cached_response})

    trial_data = get_trial_data(disease)
    if not trial_data:
        return jsonify({'error': 'No data found'}), 404

    all_trials_list = []
    for study in trial_data.get('studies', []):
        try:
            locations = study.get('protocolSection', {}).get('contactsLocationsModule', {}).get('locations', [])
            if country:
                if not any(loc.get('country') == country for loc in locations):
                    continue

            simplified_study = {
                'NCTId': study.get('protocolSection', {}).get('identificationModule', {}).get('nctId'),
                'BriefTitle': study.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle'),
                'StudyUrl': f"https://clinicaltrials.gov/study/{study.get('protocolSection', {}).get('identificationModule', {}).get('nctId')}",
                'BriefSummary': study.get('protocolSection', {}).get('descriptionModule', {}).get('briefSummary'),
                'InterventionType': [intervention.get('type') for intervention in study.get('protocolSection', {}).get('armsInterventionsModule', {}).get('interventions', [])],
                'InterventionName': [intervention.get('name') for intervention in study.get('protocolSection', {}).get('armsInterventionsModule', {}).get('interventions', [])],
                'CompletionDate': study.get('protocolSection', {}).get('statusModule', {}).get('completionDateStruct', {}).get('date'),
                'Locations': [{'facility': loc.get('facility'), 'city': loc.get('city'), 'state': loc.get('state'), 'country': loc.get('country')} for loc in locations],
                'Phases': study.get('protocolSection', {}).get('designModule', {}).get('phases', []),
                'StudyType': study.get('protocolSection', {}).get('designModule', {}).get('studyType'),
                'EligibilityModule': study.get('protocolSection', {}).get('eligibilityModule', {})
            }
            all_trials_list.append(simplified_study)
        except Exception as e:
            print(f"Error processing study {study}: {e}")

    # CTIS supplemental trials (all statuses)
    try:
        ctis_parsed = _ctis_fetch_parsed(disease, "all")
        ctis_unique = _ctis_dedup(all_trials_list, ctis_parsed)
        if country:
            country_iso = _CTIS_COUNTRY_TO_ISO.get(country)
            if country_iso:
                ctis_unique = [
                    t for t in ctis_unique
                    if any(s.get("country") == country_iso for s in t.get("sites", []))
                ]
            else:
                ctis_unique = []
        all_trials_list.extend(_ctis_normalize(t) for t in ctis_unique)
    except Exception as e:
        print(f"[CTIS] all_trials error: {e}")

    set_response_cache_response(endpoint, disease, cache_country, all_trials_list)

    return jsonify({'trials': all_trials_list})

@app.route('/specialized_centers', methods=['GET'])
def specialized_centers():
    api_key = request.headers.get('x-api-key')
    if api_key not in ALLOWED_API_KEYS:
        return jsonify({'error': 'Unauthorized'}), 401

    disease = request.args.get('disease')
    country = request.args.get('country')
    endpoint = 'specialized_centers'

    # Check response cache
    cached_response = get_cached_response(endpoint, disease, country)
    if cached_response and len(cached_response) > 0:
        return jsonify({'centers': cached_response})

    # Get trial data
    trial_data = get_trial_data(disease)
    if not trial_data:
        return jsonify({'error': 'No data found'}), 404

    specialized_centers_data = {}
    similarity_threshold = 50
    ctgov_nct_ids: set = set()

    for study in trial_data.get('studies', []):
        try:
            ps = study.get('protocolSection', {})
            nct = ps.get('identificationModule', {}).get('nctId', '')
            if nct:
                ctgov_nct_ids.add(nct.upper())
            locations = ps.get('contactsLocationsModule', {}).get('locations', [])
            for loc in locations:
                if country and loc.get('country') != country:
                    continue
                facility = loc.get('facility')
                city = loc.get('city')

                matched_facility = None
                for existing_facility in specialized_centers_data.keys():
                    if fuzz.ratio(facility, existing_facility) >= similarity_threshold:
                        matched_facility = existing_facility
                        break

                if matched_facility:
                    facility = matched_facility
                else:
                    specialized_centers_data[facility] = {
                        'city': city,
                        'count': 0,
                        'interventions': set()
                    }

                specialized_centers_data[facility]['count'] += 1
                specialized_centers_data[facility]['interventions'].update(
                    [intervention.get('name') for intervention in ps.get('armsInterventionsModule', {}).get('interventions', [])]
                )
        except Exception as e:
            print(f"Error processing study {study}: {e}")

    # CTIS supplemental sites
    try:
        country_iso = _CTIS_COUNTRY_TO_ISO.get(country) if country else None
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

                matched_facility = None
                for existing_facility in specialized_centers_data.keys():
                    if fuzz.ratio(facility, existing_facility) >= similarity_threshold:
                        matched_facility = existing_facility
                        break

                if matched_facility:
                    facility = matched_facility
                else:
                    specialized_centers_data[facility] = {
                        'city': city,
                        'count': 0,
                        'interventions': set()
                    }

                specialized_centers_data[facility]['count'] += 1
                specialized_centers_data[facility]['interventions'].update(intervention_names)
    except Exception as e:
        print(f"[CTIS] specialized_centers error: {e}")

    response_data = sorted(
        [
            {
                'facility': facility,
                'city': data['city'],
                'trialCount': data['count'],
                'interventions': list(data['interventions'])
            }
            for facility, data in specialized_centers_data.items() if data['count'] > 4
        ],
        key=lambda x: x['trialCount'],
        reverse=True
    )

    # Cache the response
    set_response_cache_response(endpoint, disease, country, response_data)

    print(f"Specialized centers for {disease} in {country} count: {len(response_data)}")

    return jsonify({'centers': response_data})

@app.route('/available_treatments', methods=['GET'])
def available_treatments():
    api_key = request.headers.get('x-api-key')
    if api_key not in ALLOWED_API_KEYS:
        return jsonify({'error': 'Unauthorized'}), 401

    disease = request.args.get('disease')
    endpoint = 'available_treatments'

    # Check response cache
    cached_response = get_cached_response(endpoint, disease, 'None')
    if cached_response and len(cached_response) > 0:
        return jsonify({'treatments': cached_response})

    # Get trial data
    trial_data = get_trial_data(disease)
    if not trial_data:
        return jsonify({'error': 'No data found'}), 404

    available_treatments_data = {}
    similarity_threshold = 50
    ctgov_nct_ids: set = set()

    for study in trial_data.get('studies', []):
        ps = study.get('protocolSection', {})
        nct = ps.get('identificationModule', {}).get('nctId', '')
        if nct:
            ctgov_nct_ids.add(nct.upper())
        if ps.get('designModule', {}).get('studyType') != 'INTERVENTIONAL':
            continue
        try:
            interventions = ps.get('armsInterventionsModule', {}).get('interventions', [])
            for intervention in interventions:
                intervention_name = intervention.get('name')
                intervention_type = intervention.get('type')

                matched_intervention = None
                for existing_intervention in available_treatments_data.keys():
                    if fuzz.ratio(intervention_name, existing_intervention) >= similarity_threshold:
                        matched_intervention = existing_intervention
                        break

                if matched_intervention:
                    intervention_name = matched_intervention
                else:
                    available_treatments_data[intervention_name] = {
                        'type': intervention_type,
                        'count': 0
                    }

                available_treatments_data[intervention_name]['count'] += 1
        except Exception as e:
            print(f"Error processing study {study}: {e}")

    # CTIS supplemental interventions (all CTIS trials are IMP trials)
    try:
        ctis_parsed = _ctis_fetch_parsed(disease, "all")
        ctis_unique = [
            t for t in ctis_parsed
            if not (t.get("nct_number") and t["nct_number"].upper() in ctgov_nct_ids)
        ]
        for trial in ctis_unique:
            for intervention in trial.get("interventions", []):
                intervention_name = intervention.get("name")
                if not intervention_name:
                    continue
                matched_intervention = None
                for existing_intervention in available_treatments_data.keys():
                    if fuzz.ratio(intervention_name, existing_intervention) >= similarity_threshold:
                        matched_intervention = existing_intervention
                        break
                if matched_intervention:
                    intervention_name = matched_intervention
                else:
                    available_treatments_data[intervention_name] = {'type': 'DRUG', 'count': 0}
                available_treatments_data[intervention_name]['count'] += 1
    except Exception as e:
        print(f"[CTIS] available_treatments error: {e}")

    response_data = sorted(
        [
            {
                'interventionName': intervention,
                'interventionType': data['type'],
                'trialCount': data['count']
            }
            for intervention, data in available_treatments_data.items() if data['count'] >= 2
        ],
        key=lambda x: x['trialCount'],
        reverse=True
    )

    # Cache the response
    set_response_cache_response(endpoint, disease, 'None', response_data)

    print(f"Current treatment for {disease} count: {len(response_data)}")

    return jsonify({'treatments': response_data})

# Additional caching functions

def get_cached_response(endpoint, disease, country):
    cache_key = get_response_cache_key(endpoint, disease, country)
    if cache_key in response_cache and is_cache_valid(response_cache[cache_key]):
        print(f'Fetched from response cache for endpoint: {endpoint}, disease: {disease}, country: {country}')
        return response_cache[cache_key]['data']
    return None

def set_response_cache_response(endpoint, disease, country, data):
    cache_key = get_response_cache_key(endpoint, disease, country)
    response_cache[cache_key] = {
        'timestamp': datetime.now().timestamp(),
        'data': data
    }
    with open(RESPONSE_CACHE_FILE, 'w') as f:
        json.dump(response_cache, f)


def split_eligibility_criteria(criteria_text):
    """Split raw criteria text into inclusion and exclusion lists."""
    inclusion, exclusion = "", ""
    if "Inclusion Criteria:" in criteria_text:
        parts = criteria_text.split("Exclusion Criteria:")
        inclusion = parts[0].replace("Inclusion Criteria:", "").strip()
        exclusion = parts[1].strip() if len(parts) > 1 else ""
    else:
        inclusion = criteria_text.strip()
    return inclusion, exclusion


@app.route('/check_eligibility', methods=['POST'])
def check_eligibility():
    # Authenticate API key
    api_key = request.headers.get('x-api-key')
    if api_key not in ALLOWED_API_KEYS:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON payload'}), 400

    nct_id = data.get('nctId')
    disease = data.get('disease')
    patient_info = data.get('patient_info')
    if not nct_id or not disease or not patient_info:
        return jsonify({'error': 'nctId, disease, and patient_info are required parameters'}), 400

    # ── CTIS path (EUCT ID) ──────────────────────────────────────────────────
    if _is_euct_id(nct_id):
        raw = _ctis_get_detail(nct_id)
        if not raw:
            return jsonify({'error': f'CTIS trial {nct_id} not found'}), 404
        parsed = _ctis_parse_trial(raw)
        inclusion_criteria = parsed.get("inclusion_criteria", "")
        exclusion_criteria = parsed.get("exclusion_criteria", "")
        min_age       = _ctis_age_str(parsed.get("min_age_months"))
        max_age       = _ctis_age_str(parsed.get("max_age_months"))
        gender        = parsed.get("gender", "not specified")
        std_ages: list = []
        brief_summary = parsed.get("lay_summary") or parsed.get("primary_objective", "")

    # ── ClinicalTrials.gov path (NCT ID) ─────────────────────────────────────
    else:
        trial_data = get_trial_data(disease)
        if not trial_data:
            return jsonify({'error': 'No trial data found for disease'}), 404

        trial = None
        for study in trial_data.get('studies', []):
            if study.get('protocolSection', {}).get('identificationModule', {}).get('nctId') == nct_id:
                trial = study
                break

        if not trial:
            return jsonify({'error': 'Trial with specified NCTId not found'}), 404

        eligibility_module = trial.get('protocolSection', {}).get('eligibilityModule', {})
        criteria_text = eligibility_module.get('criteria', 'No criteria provided')
        min_age   = eligibility_module.get('minimumAge', 'not specified')
        max_age   = eligibility_module.get('maximumAge', 'not specified')
        gender    = eligibility_module.get('gender', 'not specified')
        std_ages  = eligibility_module.get('stdAges', [])
        brief_summary = trial.get('protocolSection', {}).get('descriptionModule', {}).get('briefSummary', '')
        inclusion_criteria, exclusion_criteria = split_eligibility_criteria(criteria_text)

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
        response = client.chat.completions.create(
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
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        return jsonify({'error': f'Error calling OpenAI API: {str(e)}'}), 500

    try:
        eligibility_result = json.loads(answer)
        if 'result' not in eligibility_result or 'explanation' not in eligibility_result:
            raise ValueError("Missing expected keys")
    except Exception as e:
        return jsonify({
            'error': 'Failed to parse OpenAI response into JSON. Raw response:',
            'raw_response': answer,
            'parse_error': str(e)
        }), 500

    return jsonify({'nctId': nct_id, 'eligibility': eligibility_result})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the clinical trial service Flask app.")
    parser.add_argument('--port', type=int, default=5000, help="Port to run the server on (default: 5000)")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=True)
