from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from unidiff import PatchSet

OUTPUT_JSONL = str(Path.cwd() / "data" / "train" / "matplotlib_all_prs_train_base_commit.jsonl")
OWNER = "matplotlib"
REPO = "matplotlib"
REPO_SLUG = f"{OWNER}/{REPO}"

MAX_PRS = 50_000
PAGE_SIZE = 30
FILES_FIRST = 100
MAX_WORKERS = 10

SKIP_IF_DIFF_TOO_LARGE = False
KEEP_TEST_CHANGES = True
SLEEP_BETWEEN_GRAPHQL_PAGES = 0.5

GRAPHQL_URL = "https://api.github.com/graphql"
REST_PR_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{{num}}"

FILE_LOCK = threading.Lock()


class TokenRotator:
    def __init__(self, tokens: list[str]):
        self.token_pool = {token: {"reset_at": 0} for token in tokens}
        self.lock = threading.Lock()

    def get_token(self) -> str:
        while True:
            now = time.time()
            with self.lock:
                available_tokens = [
                    token for token, info in self.token_pool.items() if now >= info["reset_at"]
                ]
                if available_tokens:
                    return random.choice(available_tokens)

                earliest_reset = min(info["reset_at"] for info in self.token_pool.values())
                wait_time = max(earliest_reset - now, 0) + 10

            print(
                f"All GitHub tokens are exhausted. Sleeping for {int(wait_time)} seconds "
                f"until {time.ctime(earliest_reset)}..."
            )
            time.sleep(wait_time)

    def mark_exhausted(self, token: str, reset_timestamp: float) -> None:
        with self.lock:
            if token in self.token_pool and reset_timestamp > self.token_pool[token]["reset_at"]:
                self.token_pool[token]["reset_at"] = reset_timestamp
                print(f"Token {token[-6:]} exhausted until {time.ctime(reset_timestamp)}")


token_rotator: TokenRotator | None = None


def get_token_rotator() -> TokenRotator:
    global token_rotator
    if token_rotator is None:
        tokens = [token.strip() for token in os.environ.get("GITHUB_TOKEN", "").strip().split(",") if token.strip()]
        if not tokens:
            raise RuntimeError("Please set `GITHUB_TOKEN` with one or more comma-separated tokens.")
        token_rotator = TokenRotator(tokens)
    return token_rotator


def load_existing_instance_ids(output_jsonl: str) -> set[str]:
    ids: set[str] = set()
    path = Path(output_jsonl)
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    obj = json.loads(line)
                    ids.add(obj.get("instance_id"))
                except Exception:
                    pass
    return ids


def write_jsonl_safe(path: str, obj: dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with FILE_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)


def request_with_retry(
    method: str,
    url: str,
    json_body: dict[str, Any] | None = None,
    use_diff: bool = False,
    max_retries: int = 10,
):
    for attempt in range(max_retries):
        current_token = get_token_rotator().get_token()
        headers = {
            "Authorization": f"Bearer {current_token}",
            "Accept": "application/vnd.github.diff" if use_diff else "application/vnd.github+json",
        }
        try:
            response = requests.request(method, url, headers=headers, json=json_body, timeout=30)

            if response.status_code in (403, 429):
                body_text = response.text.lower()
                if response.headers.get("x-ratelimit-remaining") == "0":
                    reset_time = int(response.headers.get("x-ratelimit-reset", time.time() + 3600))
                    get_token_rotator().mark_exhausted(current_token, reset_time)
                    continue
                if "retry-after" in response.headers:
                    wait_time = int(response.headers["retry-after"])
                    print(f"Secondary rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                if "rate limit" in body_text:
                    get_token_rotator().mark_exhausted(current_token, time.time() + 3600)
                    continue

            if 200 <= response.status_code < 300:
                return response

            if response.status_code >= 500:
                time.sleep(2**attempt)
                continue
        except Exception as exc:
            print(f"Network exception on attempt {attempt}: {exc}")
            time.sleep(2)

    raise RuntimeError(f"Failed to complete request {url} after {max_retries} retries")


def graphql_fetch_pr_page(after_cursor: str | None) -> dict[str, Any]:
    after_clause = f', after: "{after_cursor}"' if after_cursor else ""
    query = f"""
    {{
      repository(owner: "{OWNER}", name: "{REPO}") {{
        pullRequests(
          states: MERGED,
          first: {PAGE_SIZE}{after_clause},
          orderBy: {{field: UPDATED_AT, direction: DESC}}
        ) {{
          edges {{
            node {{
              number
              title
              url
              body
              merged
              mergedAt
              closedAt
              updatedAt
              baseRefOid
              headRefOid
              commits(first: 1) {{
                nodes {{
                  commit {{
                    parents(first: 1) {{
                      nodes {{
                        oid
                      }}
                    }}
                  }}
                }}
              }}
              files(first: {FILES_FIRST}) {{
                edges {{
                  node {{
                    path
                  }}
                }}
              }}
            }}
          }}
          pageInfo {{
            hasNextPage
            endCursor
          }}
        }}
      }}
    }}
    """
    response = request_with_retry("POST", GRAPHQL_URL, json_body={"query": query})
    data = response.json()
    if "errors" in data:
        print(f"GraphQL error: {data['errors'][0].get('message')}")
        return {}
    return data["data"]["repository"]["pullRequests"]


def process_single_pr(node: dict[str, Any], existing_ids: set[str]):
    pr_number = node["number"]
    instance_id = f"{OWNER}__{REPO}-{pr_number}"
    if instance_id in existing_ids:
        return None

    real_base_commit = None
    try:
        commits_nodes = node.get("commits", {}).get("nodes", [])
        if commits_nodes:
            parents = commits_nodes[0]["commit"].get("parents", {}).get("nodes", [])
            if parents:
                real_base_commit = parents[0]["oid"]
    except Exception:
        pass

    if not real_base_commit:
        real_base_commit = node.get("baseRefOid") or ""

    patch = ""
    diff_too_large = False
    try:
        url = REST_PR_URL.format(num=pr_number)
        response = request_with_retry("GET", url, use_diff=True)
        diff_text = response.text
        if KEEP_TEST_CHANGES:
            patch = diff_text
        else:
            if len(diff_text) > 1_000_000:
                diff_too_large = True
            else:
                try:
                    patch_set = PatchSet(diff_text)
                    patch_filtered = ""
                    for changed_file in patch_set:
                        path = getattr(changed_file, "path", "") or ""
                        is_test = any(word in path for word in ["test", "tests", "e2e", "testing"])
                        if not is_test:
                            patch_filtered += str(changed_file)
                    patch = patch_filtered
                except Exception:
                    patch = diff_text
    except Exception as exc:
        if "406" in str(exc) or "too large" in str(exc):
            diff_too_large = True

    return {
        "repo": REPO_SLUG,
        "instance_id": instance_id,
        "pr_number": pr_number,
        "base_commit": real_base_commit,
        "head_commit": node.get("headRefOid") or "",
        "url": node["url"],
        "title": node.get("title") or "",
        "problem_statement": node.get("body") or "",
        "patch": patch,
        "diff_too_large": diff_too_large,
        "merged": bool(node.get("merged")),
        "merged_at": node.get("mergedAt"),
        "closed_at": node.get("closedAt"),
        "updated_at": node.get("updatedAt"),
        "files_preview": [item["node"]["path"] for item in (node.get("files", {}).get("edges") or [])],
    }


def collect_pr_data(repo_name: str, owner: str, output_path: str) -> None:
    global OUTPUT_JSONL, REPO, OWNER, REPO_SLUG, REST_PR_URL

    OUTPUT_JSONL = output_path
    REPO = repo_name
    OWNER = owner
    REPO_SLUG = f"{OWNER}/{REPO}"
    REST_PR_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{{num}}"

    existing_ids = load_existing_instance_ids(OUTPUT_JSONL)
    print(f"[INFO] Loaded {len(existing_ids)} existing PR records")

    total_scanned = 0
    total_written = 0
    cursor = None
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    while total_scanned < MAX_PRS:
        print(f"Fetching GraphQL page (cursor={cursor})...")
        try:
            pr_data = graphql_fetch_pr_page(cursor)
        except Exception as exc:
            print(f"GraphQL page failed: {exc}")
            time.sleep(5)
            continue

        if not pr_data:
            break

        edges = pr_data.get("edges") or []
        if not edges:
            break

        page_info = pr_data.get("pageInfo") or {}
        cursor = page_info.get("endCursor")
        total_scanned += len(edges)

        futures = []
        for edge in edges:
            node = edge["node"]
            instance_id = f"{OWNER}__{REPO}-{node['number']}"
            if instance_id in existing_ids:
                continue
            futures.append(executor.submit(process_single_pr, node, existing_ids))

        for future in as_completed(futures):
            result = future.result()
            if result:
                write_jsonl_safe(OUTPUT_JSONL, result)
                existing_ids.add(result["instance_id"])
                total_written += 1
                if total_written % 10 == 0:
                    print(f"Progress: {total_written} written (scanned {total_scanned})")

        if not page_info.get("hasNextPage"):
            break
        time.sleep(SLEEP_BETWEEN_GRAPHQL_PAGES)

    executor.shutdown()
    print(f"Done. Total scanned: {total_scanned}, written: {total_written}")
