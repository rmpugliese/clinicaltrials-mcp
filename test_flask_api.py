"""
Interactive test client for Clinical Trials Flask REST API.

Usage:
    Local:  python test_flask_api.py
    Remote: python test_flask_api.py --host 167.86.115.64 --port 5000
"""

import argparse
import json
import requests

parser = argparse.ArgumentParser(description="Flask REST API interactive test client")
parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
parser.add_argument("--api-key", default="b350e4e379b6da7ff0e0a4c432750c9f", help="API key for x-api-key header")
parser.add_argument("--https", action="store_true", help="Use HTTPS instead of HTTP")
args = parser.parse_args()

scheme = "https" if args.https else "http"
BASE_URL = f"{scheme}://{args.host}:{args.port}"
API_KEY = args.api_key

ENDPOINTS_CONFIG = {
    "1": {
        "name": "current_trials",
        "method": "GET",
        "path": "/current_trials",
        "description": "Recruiting trials by disease and country",
        "params": [
            {"name": "disease", "prompt": "Disease (e.g., lung cancer): "},
            {"name": "country", "prompt": "Country (e.g., Italy): "},
        ]
    },
    "2": {
        "name": "all_trials",
        "method": "GET",
        "path": "/all_trials",
        "description": "All trials by disease (optional country)",
        "params": [
            {"name": "disease", "prompt": "Disease (e.g., breast cancer): "},
            {"name": "country", "prompt": "Country (leave empty for all): ", "optional": True},
        ]
    },
    "3": {
        "name": "specialized_centers",
        "method": "GET",
        "path": "/specialized_centers",
        "description": "Treatment centers ranked by trial count",
        "params": [
            {"name": "disease", "prompt": "Disease (e.g., melanoma): "},
            {"name": "country", "prompt": "Country (e.g., Germany): "},
        ]
    },
    "4": {
        "name": "available_treatments",
        "method": "GET",
        "path": "/available_treatments",
        "description": "Interventions from interventional studies",
        "params": [
            {"name": "disease", "prompt": "Disease (e.g., diabetes): "},
        ]
    },
    "5": {
        "name": "check_eligibility",
        "method": "POST",
        "path": "/check_eligibility",
        "description": "AI-powered eligibility check (requires OpenAI)",
        "params": [
            {"name": "nctId", "prompt": "NCT ID (e.g., NCT04512345): "},
            {"name": "disease", "prompt": "Disease: "},
            {"name": "patient_info", "prompt": "Patient info (age, gender, conditions): "},
        ]
    },
}


def print_menu():
    print("\n" + "=" * 60)
    print("  CLINICAL TRIALS REST API - Interactive Test Client")
    print("=" * 60)
    print("\nAvailable endpoints:\n")
    for key, ep in ENDPOINTS_CONFIG.items():
        print(f"  [{key}] [{ep['method']}] {ep['path']}")
        print(f"      {ep['description']}\n")
    print("  [0] Exit")
    print("-" * 60)


def get_arguments(endpoint_config):
    arguments = {}
    print(f"\nEnter parameters for {endpoint_config['path']}:\n")
    for param in endpoint_config["params"]:
        value = input(f"   {param['prompt']}").strip()
        if value:
            arguments[param["name"]] = value
    return arguments


def display_results(data, endpoint_name):
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if "error" in data:
        print(f"\nError: {data['error']}")
        return

    if endpoint_name in ("current_trials", "all_trials"):
        trials = data.get("trials", [])
        print(f"\nFound {len(trials)} trials:\n")
        for i, trial in enumerate(trials[:5], 1):
            print(f"{i}. [{trial.get('NCTId')}] {trial.get('BriefTitle', 'N/A')[:60]}...")
            print(f"   Type: {trial.get('StudyType')} | Phases: {trial.get('Phases')}")
            print(f"   URL: {trial.get('StudyUrl')}")
            print()
        if len(trials) > 5:
            print(f"   ... and {len(trials) - 5} more trials")

    elif endpoint_name == "specialized_centers":
        centers = data.get("centers", [])
        print(f"\nFound {len(centers)} specialized centers:\n")
        for i, center in enumerate(centers[:10], 1):
            print(f"{i}. {center.get('facility', 'N/A')}")
            print(f"   City: {center.get('city')} | Trials: {center.get('trialCount')}")
            print()
        if len(centers) > 10:
            print(f"   ... and {len(centers) - 10} more centers")

    elif endpoint_name == "available_treatments":
        treatments = data.get("treatments", [])
        print(f"\nFound {len(treatments)} treatments:\n")
        for i, treatment in enumerate(treatments[:10], 1):
            print(f"{i}. {treatment.get('interventionName', 'N/A')}")
            print(f"   Type: {treatment.get('interventionType')} | Trials: {treatment.get('trialCount')}")
            print()
        if len(treatments) > 10:
            print(f"   ... and {len(treatments) - 10} more treatments")

    elif endpoint_name == "check_eligibility":
        print(f"\nTrial: {data.get('nctId')}")
        eligibility = data.get("eligibility", {})
        result = eligibility.get("result", "unknown").upper()
        print(f"\nEligibility: {result}")
        print(f"\nExplanation:\n   {eligibility.get('explanation', 'N/A')}")

        met = eligibility.get("inclusion_criteria_met", [])
        if met:
            print(f"\nInclusion criteria met ({len(met)}):")
            for c in met:
                print(f"   - {c}")

        violated = eligibility.get("exclusion_criteria_violated", [])
        if violated:
            print(f"\nExclusion criteria violated ({len(violated)}):")
            for c in violated:
                print(f"   - {c}")

        uncertain = eligibility.get("uncertain_criteria", [])
        if uncertain:
            print(f"\nUncertain criteria - missing patient data ({len(uncertain)}):")
            for c in uncertain:
                print(f"   - {c}")

    else:
        print(json.dumps(data, indent=2)[:1000])


def call_endpoint(endpoint_config, arguments):
    url = BASE_URL + endpoint_config["path"]
    headers = {"x-api-key": API_KEY}

    if endpoint_config["method"] == "GET":
        response = requests.get(url, params=arguments, headers=headers, timeout=60)
    else:
        response = requests.post(url, json=arguments, headers=headers, timeout=60)

    return response


def interactive_session():
    print("\nConnecting to Flask REST API...")
    print(f"   Base URL: {BASE_URL}")
    print(f"   API key: {API_KEY[:8]}...")

    # Quick connectivity check
    try:
        r = requests.get(BASE_URL + "/current_trials", headers={"x-api-key": API_KEY},
                         params={"disease": "_ping_", "country": "_ping_"}, timeout=5)
        print("Connected successfully!\n")
    except requests.exceptions.ConnectionError:
        print(f"\nConnection refused. Is the server running?")
        print(f"   Start it with: python clinicaltrialservice.py --port {args.port}")
        return
    except requests.exceptions.Timeout:
        print("\nConnection timed out.")
        return

    while True:
        print_menu()
        choice = input("\nSelect endpoint [0-5]: ").strip()

        if choice == "0":
            print("\nGoodbye!\n")
            break

        if choice not in ENDPOINTS_CONFIG:
            print("\nInvalid choice. Please try again.")
            continue

        endpoint_config = ENDPOINTS_CONFIG[choice]
        arguments = get_arguments(endpoint_config)

        print(f"\nCalling [{endpoint_config['method']}] {endpoint_config['path']}...")
        print(f"   Arguments: {arguments}")

        try:
            response = call_endpoint(endpoint_config, arguments)
            print(f"   Status: {response.status_code}")
            data = response.json()
            display_results(data, endpoint_config["name"])
        except requests.exceptions.Timeout:
            print("\nRequest timed out.")
        except Exception as e:
            print(f"\nError: {e}")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    interactive_session()
