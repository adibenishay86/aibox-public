#!/usr/bin/env python3
"""
Create a Google API Key programmatically.

Requirements:
  pip install google-api-python-client google-auth-httplib2 google-auth

Usage:
  python create_google_api_key.py --sa PATH/TO/service-account.json --project YOUR_PROJECT_ID --name "My AI Box Key"

Notes:
  - The service account must have `roles/serviceusage.serviceUsageAdmin` or equivalent
    and permission to manage API keys for the project.
  - The API Keys API must be enabled for the project: `apikeys.googleapis.com`.
  - This script creates a key and polls the long-running operation until completion,
    then prints the created key resource (may include `keyString`).
"""
import argparse
import time
import json
import sys
from google.oauth2 import service_account
from googleapiclient.discovery import build


def create_key(sa_file, project, display_name):
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    creds = service_account.Credentials.from_service_account_file(sa_file, scopes=scopes)

    service = build("apikeys", "v2", credentials=creds, cache_discovery=False)

    parent = f"projects/{project}/locations/global"
    body = {"displayName": display_name}

    print("Creating API key...")
    op = service.projects().locations().keys().create(parent=parent, body=body).execute()
    op_name = op.get("name")
    if not op_name:
        print("Unexpected response creating key:", json.dumps(op, indent=2))
        return 1

    print(f"Operation started: {op_name}. Polling until done...")
    while True:
        op_status = service.operations().get(name=op_name).execute()
        if op_status.get("done"):
            if "response" in op_status:
                resp = op_status["response"]
                # The API may return the created key in the response
                print("Operation completed. Response:")
                print(json.dumps(resp, indent=2))
                # Try to extract a human-usable key string if present
                key = resp.get("key") or resp.get("apiKey") or resp
                if isinstance(key, dict) and key.get("keyString"):
                    print("Created API key string:", key.get("keyString"))
                else:
                    print("Created key resource returned above. If `keyString` is not shown, check the Cloud Console.")
                return 0
            else:
                print("Operation completed but no response found:")
                print(json.dumps(op_status, indent=2))
                return 1
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sa", required=True, help="Path to service account JSON file")
    parser.add_argument("--project", required=True, help="GCP project id")
    parser.add_argument("--name", default="generated-key", help="Display name for the key")
    args = parser.parse_args()

    try:
        rc = create_key(args.sa, args.project, args.name)
        sys.exit(rc)
    except Exception as e:
        print("Error:", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
