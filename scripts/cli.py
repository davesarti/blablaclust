"""
Conversational Clustering — interactive CLI

Usage:
    python scripts/cli.py
    python scripts/cli.py --dataset amazon_reviews_train
    python scripts/cli.py --session <existing-session-id>
    python scripts/cli.py --url http://localhost:8000

Type your feedback at each prompt. Type 'quit' or 'exit' to stop.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(base_url: str, path: str, body: dict) -> dict:
    """Send a POST request to the backend and return the parsed JSON response."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(base_url: str, path: str) -> dict:
    """Send a GET request to the backend and return the parsed JSON response."""
    with urllib.request.urlopen(f"{base_url}{path}") as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def create_session(base_url: str, dataset_name: str) -> str:
    """Create a new session and return its ID."""
    result = _post(base_url, "/sessions", {"dataset_name": dataset_name})
    return result["id"]


def send_turn(base_url: str, session_id: str, text: str) -> dict:
    """Send oracle feedback and return the system response."""
    body = {
        "session_id": session_id,
        "raw_text": text,
        "feedback_type": "global",
        "target_cluster_id": None,
        "target_point_ids": [],
        "metadata": {},
    }
    return _post(base_url, f"/sessions/{session_id}/turns", body)


def print_response(turn: dict) -> None:
    """Print the system response in a readable format."""
    display = turn.get("display", {})
    content = display.get("content", str(turn))
    action = turn.get("action", "")
    print(f"\n[Sistema — {action}]: {content}\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Conversational Clustering CLI")
    parser.add_argument("--dataset", default="amazon_reviews_train", help="Dataset name")
    parser.add_argument("--session", default=None, help="Resume an existing session ID")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    args = parser.parse_args()

    base_url = args.url

    print("=== Conversational Clustering CLI ===")
    print(f"Backend: {base_url}")

    # Check that the backend is reachable before doing anything else.
    try:
        _get(base_url, "/sessions")
    except urllib.error.URLError:
        print(f"\nErrore: impossibile connettersi al backend su {base_url}")
        print("Assicurati che il server sia avviato con: uvicorn backend.main:app --reload")
        sys.exit(1)

    # Create a new session or resume an existing one.
    if args.session:
        session_id = args.session
        print(f"Ripresa sessione: {session_id}")
    else:
        print(f"Dataset: {args.dataset}")
        print("Creazione sessione...", end=" ", flush=True)
        try:
            session_id = create_session(base_url, args.dataset)
            print(f"OK — Session ID: {session_id}")
        except Exception as e:
            print(f"\nErrore nella creazione della sessione: {e}")
            sys.exit(1)

    print("\nScrivi il tuo feedback ad ogni turno. 'quit' per uscire.\n")

    turn_number = 1
    while True:
        try:
            text = input(f"Turn {turn_number} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUscita.")
            break

        if not text:
            continue

        if text.lower() in {"quit", "exit", "q"}:
            print("Sessione terminata.")
            break

        try:
            response = send_turn(base_url, session_id, text)
            print_response(response)

            # Stop the loop if the system decided the session is done.
            if response.get("action") == "stop":
                print("Il sistema ha raggiunto la convergenza. Sessione chiusa.")
                break

        except urllib.error.HTTPError as e:
            # /turns endpoint not yet implemented — inform the user clearly.
            if e.code == 404:
                print("\n[Attenzione]: L'endpoint /turns non è ancora implementato da P1.")
                print(f"Session ID corrente: {session_id}\n")
            else:
                print(f"\nErrore HTTP {e.code}: {e.reason}\n")
        except urllib.error.URLError as e:
            print(f"\nErrore di connessione: {e.reason}\n")

        turn_number += 1


if __name__ == "__main__":
    main()
