# Clinical Trials MCP Server

MCP server and REST API for clinical trial discovery and AI-powered patient eligibility checking. Aggregates data from ClinicalTrials.gov and CTIS (EU Clinical Trials Register).

## Features

- Query recruiting and completed clinical trials by disease and country
- Aggregate data from two registries: **ClinicalTrials.gov** (NCT IDs) and **CTIS / euclinicaltrials.eu** (EUCT IDs)
- Automatic deduplication of trials present in both registries
- Identify specialized treatment centers ranked by trial volume
- Discover available investigational treatments
- AI-powered patient eligibility evaluation via OpenAI (supports both NCT and EUCT trial IDs)
- Dual caching system with 24-hour TTL to minimize external API calls

## Architecture

Two server implementations sharing the same data layer:

| Server | File | Transport | Authentication |
|--------|------|-----------|----------------|
| Flask REST API | `clinicaltrialservice.py` | HTTP/HTTPS | `x-api-key` header |
| MCP Server | `clinicaltrials_mcp.py` | stdio or SSE | Bearer token (SSE mode) |

### Data Sources

- **ClinicalTrials.gov** — paginated via `https://clinicaltrials.gov/api/v2/studies`
- **CTIS** — queried via `https://euclinicaltrials.eu/ctis-public-api/`

### Caching

| File | Content | Key |
|------|---------|-----|
| `api_cache.json` | Raw ClinicalTrials.gov responses | MD5 of disease name |
| `response_cache_flask.json` | Processed Flask responses | MD5 of endpoint+disease+country |
| `response_cache_mcp.json` | Processed MCP responses | MD5 of endpoint+disease+country |
| `ctis_cache.json` | CTIS trial details and lists | MD5 of EUCT code or query |

## Installation

```bash
git clone https://github.com/rmpugliese/clinicaltrials-mcp.git
cd clinicaltrials-mcp
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file (see `.env.example`):

```env
ALLOWED_API_KEYS=your-api-key-here,optional-second-key
OPENAI_API_KEY=sk-...
```

Generate a new API key:
```bash
python gen_new_api_key.py
```

## Usage

### Flask REST API

```bash
python clinicaltrialservice.py --port 5000
```

### MCP Server — stdio (local clients)

```bash
python clinicaltrials_mcp.py
```

### MCP Server — SSE (remote clients)

```bash
python clinicaltrials_mcp.py --transport sse --port 8080
```

## API Reference

### `GET /current_trials`

Recruiting trials with at least one active site in the specified country.

| Parameter | Type | Required |
|-----------|------|----------|
| `disease` | string | yes |
| `country` | string | yes |

```bash
curl -H "x-api-key: <key>" \
  "http://localhost:5000/current_trials?disease=glioblastoma&country=Italy"
```

### `GET /all_trials`

All trials for a disease, with optional country filter.

| Parameter | Type | Required |
|-----------|------|----------|
| `disease` | string | yes |
| `country` | string | no |

### `GET /specialized_centers`

Treatment centers with more than 4 trials, ranked by trial count. Similar facility names are deduplicated via fuzzy matching.

| Parameter | Type | Required |
|-----------|------|----------|
| `disease` | string | yes |
| `country` | string | yes |

### `GET /available_treatments`

Investigational treatments appearing in at least 2 trials (interventional studies only).

| Parameter | Type | Required |
|-----------|------|----------|
| `disease` | string | yes |

### `POST /check_eligibility`

AI-powered eligibility evaluation for a specific trial. Supports both NCT IDs (ClinicalTrials.gov) and EUCT IDs (CTIS).

```json
{
  "nctId": "NCT04512345",
  "disease": "glioblastoma",
  "patient_info": "Male, 55 years, glioblastoma IDH-wildtype, MGMT not methylated, Karnofsky 80, partial resection"
}
```

Response:
```json
{
  "nctId": "NCT04512345",
  "eligibility": {
    "result": "yes | no | unknown",
    "explanation": "...",
    "inclusion_criteria_met": ["..."],
    "exclusion_criteria_violated": ["..."],
    "uncertain_criteria": ["..."]
  }
}
```

## MCP Tools

When used as an MCP server, the same functionality is available as tools:

| Tool | Equivalent endpoint |
|------|-------------------|
| `get_current_trials` | `GET /current_trials` |
| `get_all_trials` | `GET /all_trials` |
| `get_specialized_centers` | `GET /specialized_centers` |
| `get_available_treatments` | `GET /available_treatments` |
| `check_eligibility` | `POST /check_eligibility` |

### MCP Client Configuration (Claude Desktop)

```json
{
  "mcpServers": {
    "clinical-trials": {
      "command": "python",
      "args": ["/path/to/clinicaltrials_mcp.py"]
    }
  }
}
```

### MCP SSE Authentication

```
Authorization: Bearer <your-api-key>
```

## Testing

Unit tests (no server required):
```bash
pytest test_clinicaltrialservice.py -v
```

Interactive test against a running Flask server:
```bash
python test_flask_api.py --host localhost --port 5000 --api-key <key>
# Remote with HTTPS:
python test_flask_api.py --host clinicaltrials.example.com --port 443 --api-key <key> --https
```

Interactive test against a running MCP SSE server:
```bash
python test_mcp_sse.py --host localhost --port 8080 --token <key>
# Remote with HTTPS:
python test_mcp_sse.py --host mcp.example.com --port 443 --token <key> --https
```

## Dependencies

- `flask` — REST API
- `mcp[cli]` / FastMCP — MCP server
- `openai` — eligibility checking
- `fuzzywuzzy` / `python-Levenshtein` — facility and treatment name deduplication
- `starlette` / `uvicorn` — ASGI server for SSE transport
- `requests`, `python-dotenv`
