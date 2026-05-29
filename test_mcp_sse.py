"""
Interactive test client for MCP SSE server with Bearer authentication.

Usage:
    Local:  python test_mcp_sse.py
    Remote: python test_mcp_sse.py --host 167.86.115.64 --port 5050
"""

import argparse
import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client

parser = argparse.ArgumentParser(description="MCP SSE interactive test client")
parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
parser.add_argument("--token", default="b350e4e379b6da7ff0e0a4c432750c9f", help="Bearer token")
parser.add_argument("--https", action="store_true", help="Use HTTPS instead of HTTP")
args = parser.parse_args()

scheme = "https" if args.https else "http"
SERVER_URL = f"{scheme}://{args.host}:{args.port}/sse"
BEARER_TOKEN = args.token

# Tool definitions with required parameters
TOOLS_CONFIG = {
    "1": {
        "name": "get_current_trials",
        "description": "Recruiting trials by disease and country",
        "params": [
            {"name": "disease", "prompt": "Disease (e.g., lung cancer): "},
            {"name": "country", "prompt": "Country (e.g., Italy): "},
        ]
    },
    "2": {
        "name": "get_all_trials",
        "description": "All trials by disease (optional country)",
        "params": [
            {"name": "disease", "prompt": "Disease (e.g., breast cancer): "},
            {"name": "country", "prompt": "Country (leave empty for all): ", "optional": True},
        ]
    },
    "3": {
        "name": "get_specialized_centers",
        "description": "Treatment centers ranked by trial count",
        "params": [
            {"name": "disease", "prompt": "Disease (e.g., melanoma): "},
            {"name": "country", "prompt": "Country (e.g., Germany): "},
        ]
    },
    "4": {
        "name": "get_available_treatments",
        "description": "Interventions from interventional studies",
        "params": [
            {"name": "disease", "prompt": "Disease (e.g., diabetes): "},
        ]
    },
    "5": {
        "name": "check_eligibility",
        "description": "AI-powered eligibility check (requires OpenAI)",
        "params": [
            {"name": "nct_id", "prompt": "NCT ID (e.g., NCT04512345): "},
            {"name": "disease", "prompt": "Disease: "},
            {"name": "patient_info", "prompt": "Patient info (age, gender, conditions): "},
        ]
    },
}


def print_menu():
    """Print the tool selection menu."""
    print("\n" + "=" * 60)
    print("  CLINICAL TRIALS MCP - Interactive Test Client")
    print("=" * 60)
    print("\nAvailable tools:\n")
    for key, tool in TOOLS_CONFIG.items():
        print(f"  [{key}] {tool['name']}")
        print(f"      {tool['description']}\n")
    print("  [0] Exit")
    print("-" * 60)


def get_tool_arguments(tool_config):
    """Prompt user for tool arguments."""
    arguments = {}
    print(f"\n📝 Enter parameters for {tool_config['name']}:\n")

    for param in tool_config["params"]:
        value = input(f"   {param['prompt']}").strip()
        if value or not param.get("optional"):
            if value:  # Only add non-empty values
                arguments[param["name"]] = value

    return arguments


def display_results(data, tool_name):
    """Display results based on tool type."""
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)

    if "error" in data:
        print(f"\n❌ Error: {data['error']}")
        return

    if tool_name == "get_current_trials" or tool_name == "get_all_trials":
        trials = data.get("trials", [])
        print(f"\n✅ Found {len(trials)} trials:\n")
        for i, trial in enumerate(trials[:5], 1):
            print(f"{i}. [{trial.get('NCTId')}] {trial.get('BriefTitle', 'N/A')[:60]}...")
            print(f"   Type: {trial.get('StudyType')} | Phases: {trial.get('Phases')}")
            print(f"   URL: {trial.get('StudyUrl')}")
            print()
        if len(trials) > 5:
            print(f"   ... and {len(trials) - 5} more trials")

    elif tool_name == "get_specialized_centers":
        centers = data.get("centers", [])
        print(f"\n✅ Found {len(centers)} specialized centers:\n")
        for i, center in enumerate(centers[:10], 1):
            print(f"{i}. {center.get('facility', 'N/A')}")
            print(f"   City: {center.get('city')} | Trials: {center.get('trialCount')}")
            print()
        if len(centers) > 10:
            print(f"   ... and {len(centers) - 10} more centers")

    elif tool_name == "get_available_treatments":
        treatments = data.get("treatments", [])
        print(f"\n✅ Found {len(treatments)} treatments:\n")
        for i, treatment in enumerate(treatments[:10], 1):
            print(f"{i}. {treatment.get('interventionName', 'N/A')}")
            print(f"   Type: {treatment.get('interventionType')} | Trials: {treatment.get('trialCount')}")
            print()
        if len(treatments) > 10:
            print(f"   ... and {len(treatments) - 10} more treatments")

    elif tool_name == "check_eligibility":
        print(f"\n🏥 Trial: {data.get('nctId')}")
        eligibility = data.get("eligibility", {})
        result = eligibility.get("result", "unknown").upper()
        emoji = "✅" if result == "YES" else "❌" if result == "NO" else "❓"
        print(f"\n{emoji} Eligibility: {result}")
        print(f"\n📋 Explanation:\n   {eligibility.get('explanation', 'N/A')}")

        met = eligibility.get("inclusion_criteria_met", [])
        if met:
            print(f"\n✅ Inclusion criteria met ({len(met)}):")
            for c in met:
                print(f"   • {c}")

        violated = eligibility.get("exclusion_criteria_violated", [])
        if violated:
            print(f"\n❌ Exclusion criteria violated ({len(violated)}):")
            for c in violated:
                print(f"   • {c}")

        uncertain = eligibility.get("uncertain_criteria", [])
        if uncertain:
            print(f"\n❓ Uncertain criteria - missing patient data ({len(uncertain)}):")
            for c in uncertain:
                print(f"   • {c}")

    else:
        # Generic JSON output
        print(json.dumps(data, indent=2)[:1000])


async def interactive_session():
    """Run interactive MCP client session."""

    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    print("\n🔌 Connecting to MCP server...")
    print(f"   URL: {SERVER_URL}")
    print(f"   Token: {BEARER_TOKEN[:8]}...")

    try:
        async with sse_client(SERVER_URL, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ Connected successfully!\n")

                # Verify tools
                tools = await session.list_tools()
                print(f"📦 Server has {len(tools.tools)} tools available")

                while True:
                    print_menu()
                    choice = input("\nSelect tool [0-5]: ").strip()

                    if choice == "0":
                        print("\n👋 Goodbye!\n")
                        break

                    if choice not in TOOLS_CONFIG:
                        print("\n⚠️  Invalid choice. Please try again.")
                        continue

                    tool_config = TOOLS_CONFIG[choice]
                    arguments = get_tool_arguments(tool_config)

                    print(f"\n⏳ Calling {tool_config['name']}...")
                    print(f"   Arguments: {arguments}")

                    try:
                        result = await session.call_tool(
                            tool_config["name"],
                            arguments=arguments
                        )

                        if result.content:
                            for content in result.content:
                                if hasattr(content, 'text'):
                                    data = json.loads(content.text)
                                    display_results(data, tool_config["name"])

                    except Exception as e:
                        print(f"\n❌ Error calling tool: {e}")

                    input("\nPress Enter to continue...")

    except ConnectionRefusedError:
        print("\n❌ Connection refused. Is the server running?")
        print("   Start it with: python clinicaltrials_mcp.py --transport sse --port 8080")
    except Exception as e:
        print(f"\n❌ Connection error: {e}")


if __name__ == "__main__":
    asyncio.run(interactive_session())
