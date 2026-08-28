"""
Devin agent driver for SREGym.

Orchestrates the two-phase benchmark flow (diagnosis + mitigation) by calling
the Devin API and submitting results to the SREGym conductor.

Devin runs as an external service -- the agent investigates via MCP tools
(kubectl, prometheus, jaeger, loki) that are pre-configured on the Devin org
and reach the SREGym MCP server directly (security-group allowlisted).

One Devin session is reused across both phases: the mitigation prompt is sent
as a follow-up message, so the diagnosis context carries over automatically.

Env config:
  DEVIN_API_URL       e.g. https://api.devin.ai
  DEVIN_ORG_API_KEY   org API key
  DEVIN_ORG_ID        org id
  DEVIN_ONCALL_MODE   "true" to request oncall mode on the session
"""

import json
import logging
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from urllib.parse import quote

import requests

from clients.harness.problem_id import resolve_problem_id

# Add SREGym root to path
sregym_root = Path(__file__).resolve().parents[2]
if str(sregym_root) not in sys.path:
    sys.path.insert(0, str(sregym_root))

from logger import init_logger  # noqa: E402

init_logger()

logger = logging.getLogger("all.devin.driver")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

API_HOSTNAME = os.getenv("API_HOSTNAME", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
CONDUCTOR_URL = f"http://{API_HOSTNAME}:{API_PORT}"

DEVIN_API_URL = os.environ.get("DEVIN_API_URL", "https://api.devin.ai")
DEVIN_ORG_API_KEY = os.environ.get("DEVIN_ORG_API_KEY", "")
DEVIN_ORG_ID = os.environ.get("DEVIN_ORG_ID", "")
DEVIN_ONCALL_MODE = os.environ.get("DEVIN_ONCALL_MODE", "false").lower() == "true"

AGENT_LOGS_DIR = os.environ.get("AGENT_LOGS_DIR", "./logs/devin")

POLL_INTERVAL = int(os.environ.get("DEVIN_POLL_INTERVAL", "15"))
PHASE_TIMEOUT = int(os.environ.get("DEVIN_PHASE_TIMEOUT", "1800"))

# Experiment key: groups all sessions of a benchmark run under one tag so they
# can be listed/compared together (e.g. "oncall-ab-2026-08").
EXPERIMENT_KEY = os.environ.get("EXPERIMENT_KEY", "")
# Extra session tags, comma-separated.
EXTRA_TAGS = [t for t in os.environ.get("DEVIN_SESSION_TAGS_EXTRA", "").split(",") if t]
# If set, ask Devin to timebox its investigation (minutes).
TIMEBOX_MINUTES = os.environ.get("DEVIN_TIMEBOX_MINUTES", "")
# Devin agent mode for the session (e.g. normal, fast, ultra). Empty = org default.
DEVIN_MODE = os.environ.get("DEVIN_MODE", "")
# Optionally pin Devin's action model by prepending a model flag to the
# prompt (honored on Devin staging/dev environments; production ignores it).
DEVIN_ACTION_MODEL = os.environ.get("DEVIN_ACTION_MODEL", "")

SESSIONS_URL = f"{DEVIN_API_URL}/v3/organizations/{DEVIN_ORG_ID}/sessions"

# Session statuses that mean "Devin has stopped working and is waiting on us"
PHASE_DONE_STATUSES = {"blocked", "finished", "stopped", "expired", "suspended"}


def devin_headers() -> dict:
    return {
        "Authorization": f"Bearer {DEVIN_ORG_API_KEY}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Devin API helpers
# ---------------------------------------------------------------------------


def create_session(prompt: str, problem_id: str) -> str:
    """Create a Devin session. Returns session_id."""
    tags = ["sregym", problem_id, "oncall" if DEVIN_ONCALL_MODE else "no-oncall"]
    if EXPERIMENT_KEY:
        tags.append(f"exp:{EXPERIMENT_KEY}")
    tags.extend(EXTRA_TAGS)
    if DEVIN_ACTION_MODEL:
        prompt = f"--action-model {DEVIN_ACTION_MODEL}\n{prompt}"
        tags.append(f"model:{DEVIN_ACTION_MODEL}")
    payload: dict = {"prompt": prompt, "tags": tags}
    if DEVIN_MODE:
        payload["devin_mode"] = DEVIN_MODE
        tags.append(f"mode:{DEVIN_MODE}")
    if DEVIN_ONCALL_MODE:
        payload["additional_args"] = {"oncall_mode": True}

    logger.info(f"Creating Devin session (oncall={DEVIN_ONCALL_MODE})")
    logger.info(f"Prompt ({len(prompt)} chars): {prompt[:200]}...")

    resp = requests.post(SESSIONS_URL, json=payload, headers=devin_headers(), timeout=30)
    if not resp.ok:
        logger.error(f"Devin API error: {resp.status_code} {resp.text}")
    resp.raise_for_status()
    data = resp.json()

    session_id = data["session_id"]
    logger.info(f"Session created: {session_id} (status={data.get('status')})")
    return session_id


def send_message(session_id: str, message: str) -> None:
    """Send a follow-up message to an existing session (wakes it if suspended)."""
    url = f"{SESSIONS_URL}/{session_id}/messages"
    logger.info(f"Sending follow-up message to {session_id} ({len(message)} chars)")
    resp = requests.post(url, json={"message": message}, headers=devin_headers(), timeout=30)
    if not resp.ok:
        logger.error(f"Send message failed: {resp.status_code} {resp.text}")
    resp.raise_for_status()


def get_session(session_id: str) -> dict:
    resp = requests.get(f"{SESSIONS_URL}/{session_id}", headers=devin_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_devin_attachments(session_id: str, max_files: int = 3) -> str:
    """Download Devin-attached text files (.md/.txt) and return their contents
    concatenated. Devin (especially in oncall mode) often puts the full writeup
    in an attached file; the grader only sees what we submit."""
    try:
        resp = requests.get(
            f"{SESSIONS_URL}/{session_id}/attachments", headers=devin_headers(), timeout=30
        )
        resp.raise_for_status()
        items = resp.json()
    except Exception as e:
        logger.warning(f"Attachment listing failed: {e}")
        return ""
    parts = []
    for item in items:
        if item.get("source") != "devin":
            continue
        name = item.get("name") or ""
        if not name.lower().endswith((".md", ".txt")):
            continue
        try:
            url = (
                f"{DEVIN_API_URL}/v3/organizations/{DEVIN_ORG_ID}/attachments/"
                f"{item['attachment_id']}/{quote(name, safe='')}"
            )
            # The endpoint 307s to a presigned S3 URL; S3 rejects requests that
            # carry an Authorization header alongside presigned query auth, so
            # follow the redirect manually without our bearer token.
            r = requests.get(url, headers=devin_headers(), allow_redirects=False, timeout=60)
            if r.status_code in (301, 302, 303, 307, 308):
                r = requests.get(r.headers["Location"], timeout=60)
            r.raise_for_status()
            parts.append(f"\n\n--- Attached file: {name} ---\n{r.text}")
            logger.info(f"Fetched attachment {name} ({len(r.text)} chars)")
        except Exception as e:
            logger.warning(f"Attachment download failed for {name}: {e}")
        if len(parts) >= max_files:
            break
    return "".join(parts)


def last_devin_message(session_id: str) -> str:
    """Fetch the most recent message authored by Devin."""
    url = f"{SESSIONS_URL}/{session_id}/messages"
    resp = requests.get(url, headers=devin_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    messages = data if isinstance(data, list) else data.get("messages", data.get("items", []))
    for msg in reversed(messages):
        who = (
            msg.get("source") or msg.get("type") or msg.get("role") or msg.get("author") or ""
        ).lower()
        if "devin" in who or who == "assistant":
            return msg.get("message") or msg.get("content") or msg.get("text") or ""
    return ""


def poll_phase(session_id: str, min_wait: int = 30) -> str:
    """Poll until the session stops working (blocked/finished/etc).

    Returns the phase result: structured_output if present, else the last
    Devin message.
    """
    start = time.time()
    time.sleep(min_wait)
    consecutive_errors = 0

    while time.time() - start < PHASE_TIMEOUT:
        try:
            session = get_session(session_id)
            consecutive_errors = 0
            status = (session.get("status") or "").lower()
            status_detail = (session.get("status_detail") or "").lower()
            elapsed = int(time.time() - start)
            logger.info(
                f"Session {session_id}: status={status}, detail={status_detail}, elapsed={elapsed}s"
            )

            if status in PHASE_DONE_STATUSES or status_detail == "waiting_for_user":
                structured = session.get("structured_output")
                if structured:
                    return json.dumps(structured, indent=2)
                result = last_devin_message(session_id)
                if result:
                    # Devin (esp. oncall mode) often attaches the full writeup
                    # as a file; the judge only reads what we submit. Strip the
                    # raw ATTACHMENT:"url" markers (noise/dead links for the
                    # judge) and append the fetched file contents instead.
                    result = re.sub(
                        r'\n?ATTACHMENT:("[^"]*"|\{[^\n]*\})', "", result
                    ).strip()
                    return result + fetch_devin_attachments(session_id)
                logger.warning("Phase looks done but no Devin message yet; continuing to poll")
        except requests.RequestException as e:
            consecutive_errors += 1
            logger.warning(f"Poll error (attempt {consecutive_errors}, will retry): {e}")
            if consecutive_errors > 10:
                logger.error(f"Too many consecutive errors polling {session_id}")
                return ""

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Session {session_id} phase timed out after {PHASE_TIMEOUT}s")


# ---------------------------------------------------------------------------
# SREGym conductor helpers
# ---------------------------------------------------------------------------


def get_app_info(max_retries: int = 6, backoff: int = 5) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(f"{CONDUCTOR_URL}/get_app", timeout=10)
            resp.raise_for_status()
            info = resp.json()
            logger.info(f"App info: {info}")
            return info
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"get_app attempt {attempt}/{max_retries} failed: {e}")
                time.sleep(backoff)
            else:
                raise


def wait_for_stage(target_stages: set[str], timeout: int = 300) -> str:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{CONDUCTOR_URL}/status", timeout=10)
            resp.raise_for_status()
            stage = resp.json().get("stage", "")
            if stage in target_stages:
                logger.info(f"Conductor reached stage: {stage}")
                return stage
        except Exception as e:
            logger.debug(f"Status poll error: {e}")
        time.sleep(2)

    raise TimeoutError(f"Conductor did not reach {target_stages} within {timeout}s")


def submit_to_conductor(solution: str) -> None:
    logger.info(f"Submitting to conductor ({len(solution)} chars)")
    resp = requests.post(f"{CONDUCTOR_URL}/submit", json={"solution": solution}, timeout=30)
    if not resp.ok:
        logger.error(f"Submit failed: {resp.status_code} {resp.text}")
    resp.raise_for_status()
    logger.info(f"Submit response: {resp.json()}")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DIAGNOSIS_PROMPT = """You are investigating a Kubernetes application failure.

Application: {app_name}
Namespace: {namespace}
Description: {descriptions}

Your task is to diagnose the root cause of the failure in this application.

Use your available MCP tools (kubectl, prometheus, jaeger, loki) to investigate.
Do not attempt to fix anything yet.

Provide a clear, concise diagnosis identifying:
- What component is failing and how
- The root cause of the failure
- Supporting evidence from your investigation

Be specific and technical. State the root cause clearly. When you are done,
post your diagnosis as a message and stop.{timebox}"""

TIMEBOX_SUFFIX = """

Aim to conclude your investigation within about {minutes} minutes; prioritize
the most likely causes first."""

MITIGATION_PROMPT = """Your diagnosis above is confirmed. Now FIX the issue using kubectl commands
via the kubectl MCP tool.

1. Based on your diagnosis, determine the appropriate fix
2. Apply the fix using kubectl (patch, scale, rollout, apply, etc.)
3. Verify the fix: pods Running, containers Ready, services responding

After applying the fix, post a summary of what you changed and how you verified
it, then stop."""


# ---------------------------------------------------------------------------
# Main driver loop
# ---------------------------------------------------------------------------


def main():
    logs_dir = Path(AGENT_LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / "driver.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("Devin driver starting")
    logger.info(f"Conductor: {CONDUCTOR_URL}")
    logger.info(f"Devin API: {DEVIN_API_URL} (org {DEVIN_ORG_ID}, oncall={DEVIN_ONCALL_MODE})")
    logger.info("=" * 60)

    if not DEVIN_ORG_API_KEY or not DEVIN_ORG_ID:
        logger.error("DEVIN_ORG_API_KEY / DEVIN_ORG_ID not set")
        sys.exit(1)

    problem_id = resolve_problem_id()

    try:
        wait_for_stage({"diagnosis"}, timeout=300)
    except TimeoutError:
        logger.error("Timed out waiting for conductor to reach diagnosis stage")
        sys.exit(1)

    app_info = get_app_info()
    params = {
        "app_name": app_info.get("app_name", "unknown"),
        "namespace": app_info.get("namespace", "default"),
        "descriptions": app_info.get("descriptions", ""),
        "timebox": (
            TIMEBOX_SUFFIX.format(minutes=TIMEBOX_MINUTES) if TIMEBOX_MINUTES else ""
        ),
    }

    # ---- PHASE 1: DIAGNOSIS ----
    logger.info("=" * 40 + " DIAGNOSIS " + "=" * 40)
    session_id = None
    diagnosis_text = ""
    try:
        session_id = create_session(DIAGNOSIS_PROMPT.format(**params), problem_id)
        diagnosis_text = poll_phase(session_id)
        logger.info(f"Diagnosis ({len(diagnosis_text)} chars): {diagnosis_text[:300]}...")
    except Exception as e:
        logger.error(f"Diagnosis phase failed: {e}")
        diagnosis_text = "Unable to complete diagnosis"

    submit_to_conductor(diagnosis_text)
    _save_stage_result(logs_dir, "diagnosis", session_id, diagnosis_text)

    # ---- Wait for mitigation stage ----
    try:
        stage = wait_for_stage({"mitigation", "done"}, timeout=300)
    except TimeoutError:
        logger.error("Timed out waiting for mitigation stage")
        sys.exit(1)

    if stage == "done":
        logger.info("Conductor went straight to done after diagnosis")
        _finish(logs_dir, problem_id, session_id)
        return

    # ---- PHASE 2: MITIGATION (same session, follow-up message) ----
    logger.info("=" * 40 + " MITIGATION " + "=" * 39)
    mitigation_text = ""
    try:
        if session_id is None:
            session_id = create_session(
                DIAGNOSIS_PROMPT.format(**params) + "\n\n" + MITIGATION_PROMPT, problem_id
            )
        else:
            send_message(session_id, MITIGATION_PROMPT)
        mitigation_text = poll_phase(session_id)
        logger.info(f"Mitigation ({len(mitigation_text)} chars): {mitigation_text[:300]}...")
    except Exception as e:
        logger.error(f"Mitigation phase failed: {e}")

    # Fix has been applied via kubectl; conductor validates cluster state itself
    submit_to_conductor("")
    _save_stage_result(logs_dir, "mitigation", session_id, mitigation_text)

    # ---- Resolution / done ----
    try:
        stage = wait_for_stage({"resolution", "done", "tearing_down"}, timeout=300)
        if stage == "resolution":
            logger.info("Resolution stage reached, submitting empty string")
            submit_to_conductor("")
            wait_for_stage({"done", "tearing_down"}, timeout=300)
    except TimeoutError:
        logger.warning("Timed out waiting for done stage")

    _finish(logs_dir, problem_id, session_id)


def _save_stage_result(logs_dir: Path, stage: str, session_id: str | None, content: str) -> None:
    result_file = logs_dir / f"{stage}_result.json"
    with open(result_file, "w") as f:
        json.dump(
            {
                "stage": stage,
                "session_id": session_id,
                "oncall_mode": DEVIN_ONCALL_MODE,
                "content_length": len(content),
                "content": content,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            f,
            indent=2,
        )
    logger.info(f"Saved {stage} result to {result_file}")


def _finish(logs_dir: Path, problem_id: str, session_id: str | None) -> None:
    summary = {
        "problem_id": problem_id,
        "driver": "devin",
        "session_id": session_id,
        "oncall_mode": DEVIN_ONCALL_MODE,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with open(logs_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Devin driver finished")


if __name__ == "__main__":
    main()
