"""
Trial records shared by the MCP server and the Flask API.

Pure functions only (no network, no cache): they turn raw ClinicalTrials.gov
studies and raw CTIS trials into the flat records returned by the tools.
"""

import json
import re
from typing import Optional

REGISTRY_CTGOV = "clinicaltrials.gov"
REGISTRY_CTIS = "ctis"

CTIS_ISO_TO_COUNTRY: dict = {
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
CTIS_COUNTRY_TO_ISO: dict = {v: k for k, v in CTIS_ISO_TO_COUNTRY.items()}

# CTIS publishes the overall trial status as a numeric code. Labels measured
# against the public retrieve endpoint (2026-09-23): codes 2-5 all read
# "Authorised" there; the portal's finer labels could not be matched to them.
CTIS_STATUS_LABELS: dict = {
    1: "Under evaluation",
    2: "Authorised", 3: "Authorised", 4: "Authorised", 5: "Authorised",
    6: "Halted",
    7: "Suspended",
    8: "Ended",
    9: "Expired",
    10: "Revoked",
    11: "Not authorised",
}

_EUCT_RE = re.compile(r"^\d{4}-\d{6}-\d{2}-\d{2}$")
_NCT_RE = re.compile(r"^NCT\d{8}$", re.I)


def is_euct_id(trial_id: str) -> bool:
    """Return True if trial_id matches the EUCT format (e.g. '2023-505701-14-00')."""
    return bool(_EUCT_RE.match(trial_id.strip()))


def is_nct_id(trial_id: str) -> bool:
    """Return True if trial_id matches the NCT format (e.g. 'NCT04512345')."""
    return bool(_NCT_RE.match(trial_id.strip()))


# ---------------------------------------------------------------------------
# ClinicalTrials.gov
# ---------------------------------------------------------------------------

def simplify_ctgov(study: dict) -> dict:
    """Flat record for a raw ClinicalTrials.gov study."""
    ps = study.get("protocolSection", {})
    id_mod = ps.get("identificationModule", {})
    status_mod = ps.get("statusModule", {})
    design_mod = ps.get("designModule", {})
    nct_id = id_mod.get("nctId")
    locations = ps.get("contactsLocationsModule", {}).get("locations", [])
    interventions = ps.get("armsInterventionsModule", {}).get("interventions", [])
    return {
        "NCTId": nct_id,
        "Registry": REGISTRY_CTGOV,
        "BriefTitle": id_mod.get("briefTitle"),
        "StudyUrl": f"https://clinicaltrials.gov/study/{nct_id}",
        "BriefSummary": ps.get("descriptionModule", {}).get("briefSummary"),
        "OverallStatus": status_mod.get("overallStatus"),
        "StartDate": status_mod.get("startDateStruct", {}).get("date"),
        "CompletionDate": status_mod.get("completionDateStruct", {}).get("date"),
        "LeadSponsor": ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name"),
        "EnrollmentCount": design_mod.get("enrollmentInfo", {}).get("count"),
        "InterventionType": [i.get("type") for i in interventions],
        "InterventionName": [i.get("name") for i in interventions],
        "Locations": [
            {
                "facility": loc.get("facility"),
                "city": loc.get("city"),
                "state": loc.get("state"),
                "country": loc.get("country"),
            }
            for loc in locations
        ],
        "Phases": design_mod.get("phases", []),
        "StudyType": design_mod.get("studyType"),
        "EligibilityModule": ps.get("eligibilityModule", {}),
    }


def ctgov_details(study: dict) -> dict:
    """Full protocol of a ClinicalTrials.gov study (results are left out)."""
    return {
        "protocolSection": study.get("protocolSection", {}),
        "hasResults": bool(study.get("hasResults")),
    }


# ---------------------------------------------------------------------------
# CTIS
# ---------------------------------------------------------------------------

def _texts(items: list, key: str) -> list:
    return [i[key].strip() for i in items or [] if isinstance(i.get(key), str) and i[key].strip()]


def _ctis_find_nct(raw: dict, identifiers: dict) -> str:
    """NCT cross-reference: the dedicated field, else the first NCT number in the JSON."""
    number = identifiers.get("secondaryIdentifyingNumbers", {}).get("nctNumber", {}).get("number", "")
    if number and re.match(r"NCT\d{8}", str(number), re.I):
        return number.upper()
    matches = re.findall(r"NCT\d{8}", json.dumps(raw), re.I)
    return matches[0].upper() if matches else ""


def _ctis_gender(overview_gender: str, population: dict) -> str:
    """CT.gov-style gender (ALL / FEMALE / MALE)."""
    text = (overview_gender or "").lower()
    female = population.get("isFemaleSubjects", "female" in text)
    male = population.get("isMaleSubjects", bool(re.search(r"\bmale\b", text)))
    if female and male:
        return "ALL"
    if female:
        return "FEMALE"
    if male:
        return "MALE"
    return ""


def _to_int(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def ctis_status_label(raw: dict, overview: Optional[dict] = None) -> str:
    """Overall CTIS status as text (e.g. 'Authorised', 'Ended')."""
    label = raw.get("ctStatus")
    if isinstance(label, str) and label:
        return label
    code = raw.get("ctPublicStatusCode") or (overview or {}).get("ctStatus")
    return CTIS_STATUS_LABELS.get(_to_int(code), "")


def status_enum(label: str) -> str:
    """'Not authorised' -> 'NOT_AUTHORISED', matching CT.gov's enum style."""
    return re.sub(r"[^A-Z0-9]+", "_", label.upper()).strip("_")


def ctis_parse_trial(raw: dict, overview: Optional[dict] = None) -> dict:
    """
    Normalize a CTIS trial into a flat dict.

    *raw* is the JSON from the retrieve endpoint; *overview* (optional) is the
    trial's row from the search endpoint, which carries phase, sponsor and
    enrollment as text.
    """
    overview = overview or {}
    app = raw.get("authorizedApplication", {})
    part1 = app.get("authorizedPartI", {})
    details = part1.get("trialDetails", {})
    identifiers = details.get("clinicalTrialIdentifiers", {})
    info = details.get("trialInformation", {})
    objective = info.get("trialObjective", {})
    eligibility = info.get("eligibilityCriteria", {})
    endpoints = info.get("endPoint", {})
    duration = info.get("trialDuration", {})
    population = info.get("populationOfTrialSubjects", {})
    euct = raw.get("ctNumber") or overview.get("ctNumber", "")

    sponsors = part1.get("sponsors", [])
    primary_sponsor = next((s for s in sponsors if s.get("primary")), sponsors[0] if sponsors else {})
    public_contact = (primary_sponsor.get("publicContacts") or [{}])[0]

    status_label = ctis_status_label(raw, overview)
    result = {
        "euct_number":          euct,
        "ctis_status":          status_label,
        "ctis_status_code":     raw.get("ctPublicStatusCode") or _to_int(overview.get("ctStatus")),
        "ctis_url":             f"https://euclinicaltrials.eu/ctis-public/search#{euct}",
        "title":                identifiers.get("publicTitle") or overview.get("ctTitle", ""),
        "official_title":       identifiers.get("fullTitle", ""),
        "short_title":          identifiers.get("shortTitle") or overview.get("shortTitle", ""),
        "conditions":           _texts(part1.get("medicalConditions"), "medicalCondition"),
        "primary_objective":    (objective.get("mainObjective") or "").strip(),
        "secondary_objectives": _texts(objective.get("secondaryObjectives"), "secondaryObjective"),
        "primary_endpoints":    _texts(endpoints.get("primaryEndPoints"), "endPoint"),
        "secondary_endpoints":  _texts(endpoints.get("secondaryEndPoints"), "endPoint"),
        "trial_phase":          overview.get("trialPhase", ""),
        "gender":               _ctis_gender(overview.get("gender", ""), population),
        "age_groups":           [a.strip() for a in overview.get("ageGroup", "").split(",") if a.strip()],
        "inclusion_criteria":   "\n".join(_texts(eligibility.get("principalInclusionCriteria"),
                                                 "principalInclusionCriteria")),
        "exclusion_criteria":   "\n".join(_texts(eligibility.get("principalExclusionCriteria"),
                                                 "principalExclusionCriteria")),
        "sponsor":              primary_sponsor.get("organisation", {}).get("name") or overview.get("sponsor", ""),
        "sponsor_type":         overview.get("sponsorType") or primary_sponsor.get("commercial", ""),
        "sponsor_contact": {
            k: v for k, v in {
                "name":  public_contact.get("functionalName", ""),
                "email": public_contact.get("functionalEmailAddress", ""),
                "phone": public_contact.get("telephone", ""),
            }.items() if v
        },
        "start_date":           raw.get("startDateEU") or duration.get("estimatedRecruitmentStartDate", ""),
        "estimated_end_date":   duration.get("estimatedGlobalEndDate") or duration.get("estimatedEndDate", ""),
        "decision_date":        (raw.get("decisionDate") or "")[:10],
    }

    result["interventions"] = [
        {k: v for k, v in {
            "name": p.get("productDictionaryInfo", {}).get("prodName") or p.get("productName", ""),
            "active_substance": p.get("productDictionaryInfo", {}).get("activeSubstanceName", ""),
            "pharmaceutical_form": p.get("pharmaceuticalFormDisplay", ""),
            "routes": p.get("routes", []),
        }.items() if v}
        for p in part1.get("products", [])
        if p.get("productDictionaryInfo", {}).get("prodName") or p.get("productName")
    ]

    sites, countries = [], []
    for part2 in app.get("authorizedPartsII", []):
        msc = part2.get("mscInfo", {})
        country_name = msc.get("countryName") or msc.get("mscName", "")
        countries.append({k: v for k, v in {
            "country":                country_name,
            "status":                 msc.get("trialStatus") or msc.get("reportingStatusCode", ""),
            "recruitment_started":    msc.get("hasRecruitmentStarted"),
            "recruitment_start_date": (msc.get("activeTrialRecruitmentPeriod") or {}).get("recruitmentStartDate", ""),
            "planned_subjects":       part2.get("recruitmentSubjectCount"),
        }.items() if v is not None and v != ""})
        for site in part2.get("trialSites", []):
            org_info = site.get("organisationAddressInfo", {})
            address = org_info.get("address", {})
            site_country = address.get("countryName") or country_name
            person = site.get("personInfo", {})
            sites.append({k: v for k, v in {
                "country":      CTIS_COUNTRY_TO_ISO.get(site_country, site_country),
                "country_name": site_country,
                "name":         org_info.get("organisation", {}).get("name", ""),
                "department":   site.get("departmentName", ""),
                "city":         address.get("city", ""),
                "pi":           " ".join(p for p in (person.get("firstName"), person.get("lastName")) if p),
            }.items() if v})
    result["country_status"] = countries
    result["trial_countries"] = [c["country"] for c in countries if c.get("country")]
    started = [c["recruitment_started"] for c in countries if "recruitment_started" in c]
    result["recruitment_started"] = any(started) if started else None
    result["sites"] = sites
    result["n_sites"] = len(sites)

    enrolled = _to_int(overview.get("totalNumberEnrolled"))
    if enrolled is None:
        planned = [c["planned_subjects"] for c in countries if isinstance(c.get("planned_subjects"), int)]
        if planned:
            enrolled = sum(planned) + (part1.get("rowSubjectCount") or 0)
    result["enrollment"] = enrolled

    result["nct_number"] = _ctis_find_nct(raw, identifiers)

    return {k: v for k, v in result.items()
            if v is not None and v != "" and v != [] and v != {}}


def ctis_normalize(parsed: dict) -> dict:
    """Convert a parsed CTIS trial to the CT.gov simplified study format."""
    euct = parsed.get("euct_number", "")
    interventions = parsed.get("interventions", [])
    sites = parsed.get("sites", [])

    eligibility: dict = {
        "minimumAge": "not specified",
        "maximumAge": "not specified",
        "gender":     parsed.get("gender") or "ALL",
    }
    if parsed.get("age_groups"):
        eligibility["stdAges"] = parsed["age_groups"]
    parts = []
    if parsed.get("inclusion_criteria"):
        parts.append(f"Inclusion Criteria:\n{parsed['inclusion_criteria']}")
    if parsed.get("exclusion_criteria"):
        parts.append(f"Exclusion Criteria:\n{parsed['exclusion_criteria']}")
    if parts:
        # ClinicalTrials.gov names this field eligibilityCriteria; "criteria"
        # is kept for clients that read the older name.
        eligibility["eligibilityCriteria"] = eligibility["criteria"] = "\n\n".join(parts)

    return {
        "NCTId": euct,
        "Registry": REGISTRY_CTIS,
        "BriefTitle": parsed.get("title") or parsed.get("official_title") or parsed.get("short_title", ""),
        "StudyUrl": parsed.get("ctis_url",
                               f"https://euclinicaltrials.eu/ctis-public/search#{euct}"),
        "BriefSummary": parsed.get("primary_objective", ""),
        "OverallStatus": status_enum(parsed.get("ctis_status", "")) or None,
        "RecruitmentStarted": parsed.get("recruitment_started"),
        "StartDate": parsed.get("start_date"),
        "CompletionDate": parsed.get("estimated_end_date"),
        "LeadSponsor": parsed.get("sponsor"),
        "EnrollmentCount": parsed.get("enrollment"),
        "InterventionType": ["DRUG"] * len(interventions),
        "InterventionName": [i.get("name", "") for i in interventions],
        "Locations": [
            {
                "facility": s.get("name") or s.get("department", ""),
                "city":     s.get("city", ""),
                "state":    None,
                "country":  CTIS_ISO_TO_COUNTRY.get(s.get("country", ""), s.get("country_name", "")),
            }
            for s in sites
        ],
        "Phases":          [parsed["trial_phase"]] if parsed.get("trial_phase") else [],
        "StudyType":       "INTERVENTIONAL",
        "EligibilityModule": eligibility,
        "_source":         "ctis",
    }
