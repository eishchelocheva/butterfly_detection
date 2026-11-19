#!/usr/bin/env python3
"""Read Yandex Tasks pools and prepare review tasks without modifying the service."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlencode
from urllib.request import Request
from urllib.error import HTTPError, URLError

REVIEW = Path(__file__).resolve().parent
BBOX = REVIEW.parent / "bbox"
sys.path.insert(0, str(BBOX))
import tools as bbox_tools
from tasks_api import API_BASE, ApiClient, ApiError, ProvisioningError

ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
STATUSES = {"SUBMITTED", "ACCEPTED"}


class ReadClient(ApiClient):
    """Use bbox authentication and transport; allow only documented list GETs."""
    def get_list(self, endpoint, params):
        if endpoint not in ("/pools", "/assignments"):
            raise ProvisioningError("Unsupported list endpoint")
        query = urlencode(params)
        req = Request(API_BASE + endpoint + "?" + query, method="GET", headers={
            "Authorization": self._scheme + " " + self._token,
            "Accept": "application/json",
        })
        try:
            with self._opener.open(req, timeout=self._timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ApiError("API response exceeds size limit")
            result = json.loads(raw)
        except HTTPError as e:
            status = e.code
            e.close()
            raise ApiError("Tasks API rejected list request", status=status) from None
        except (URLError, OSError):
            raise ApiError("Tasks API transport failed") from None
        except (ValueError, UnicodeError):
            raise ApiError("Tasks API returned invalid JSON") from None
        if not isinstance(result, dict):
            raise ApiError("API response must be an object")
        return result


def pages(client, endpoint, params):
    cursor = None
    while True:
        query = {**params, "sort": "id", "limit": 100}
        if cursor is not None:
            query["id_gt"] = cursor
        page = client.get_list(endpoint, query)
        items, more = page.get("items"), page.get("has_more")
        if not isinstance(items, list) or type(more) is not bool:
            raise ValueError("Invalid API list response")
        previous = cursor
        for item in items:
            identifier = item.get("id") if isinstance(item, dict) else None
            if not isinstance(identifier, str) or not ID.fullmatch(identifier):
                raise ValueError("Invalid ID in API response")
            if previous is not None and identifier <= previous:
                raise ValueError("API returned non-increasing IDs")
            previous = identifier
            yield item
        if not more:
            return
        if not items:
            raise ValueError("API pagination cannot advance")
        cursor = previous


def submitted_time(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Completed assignment has no submitted timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("Invalid submitted timestamp") from None
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def assignment_records(assignment, expected_pool):
    if assignment.get("status") not in STATUSES:
        return []
    if assignment.get("pool_id") != expected_pool:
        raise ValueError("Assignment pool differs from requested pool")
    assignment_id, worker = assignment.get("id"), assignment.get("user_id")
    if not isinstance(assignment_id, str) or not ID.fullmatch(assignment_id):
        raise ValueError("Assignment ID missing or invalid")
    if not isinstance(worker, str) or not worker.strip():
        raise ValueError("Assignment worker ID missing")
    when = submitted_time(assignment.get("submitted"))
    tasks, solutions = assignment.get("tasks"), assignment.get("solutions")
    if not isinstance(tasks, list) or not isinstance(solutions, list) or len(tasks) != len(solutions):
        raise ValueError("Assignment tasks and solutions are missing or have different lengths")
    records = []
    for task, solution in zip(tasks, solutions):
        if not isinstance(task, dict) or not isinstance(solution, dict):
            raise ValueError("Invalid task or solution")
        if solution.get("task_id") is not None and solution["task_id"] != task.get("id"):
            raise ValueError("Solution task_id does not match task order")
        inputs, outputs = task.get("input_values"), solution.get("output_values")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise ValueError("Missing task input_values or output_values")
        records.append({
            "pool_id": expected_pool, "assignment_id": assignment_id,
            "worker_id": worker, "submitted": assignment["submitted"],
            "_when": when, "task_id": task.get("id"),
            "input_values": inputs, "output_values": outputs,
        })
    return records


def fetch_assignments(client, pool_ids):
    all_records = []
    for pool_id in dict.fromkeys(pool_ids):
        if not ID.fullmatch(pool_id):
            raise ValueError("Invalid pool ID")
        for item in pages(client, "/assignments", {
            "pool_id": pool_id, "status": "SUBMITTED,ACCEPTED"
        }):
            all_records.extend(assignment_records(item, pool_id))
    return all_records


def serial(record):
    return {key: value for key, value in record.items() if key != "_when"}


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def publish_directory(out, build):
    if out.exists() or out.is_symlink():
        raise ValueError("Destination already exists: " + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".review-export-", dir=out.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        build(stage)
        out.mkdir()
        try:
            os.replace(stage, out)
        except OSError:
            out.rmdir()
            raise


def bbox_records(records, canonical):
    seen = set()
    result = []
    for r in records:
        inp, out = r["input_values"], r["output_values"]
        identifier = bbox_tools.photo_id(inp.get("photo_id"))
        if identifier not in canonical or inp.get("image_url") != canonical[identifier]["image_url"]:
            raise ValueError("Input photo ID or URL differs from canonical dataset: " + identifier)
        answer = {
            "photo_id": identifier, "worker_id": r["worker_id"],
            "assignment_id": r["assignment_id"], "status": out.get("status"),
            "boxes": [] if out.get("boxes") is None else out["boxes"],
            "comment": "" if out.get("comment") is None else out["comment"],
        }
        normalized = bbox_tools.validate_answer(answer, canonical, seen)
        result.append({**normalized, "pool_id": r["pool_id"], "submitted": r["submitted"],
                       "_when": r["_when"], "task_id": r["task_id"]})
    return result


def pick_first(rows):
    selected = {}
    for row in rows:
        key = row["photo_id"]
        order = (row["_when"], row["assignment_id"], row["pool_id"], str(row["task_id"] or ""))
        if key not in selected or order < selected[key][0]:
            selected[key] = (order, row)
    return [selected[k][1] for k in sorted(selected, key=int)]


def run_pools(args, client):
    params = {}
    if args.project_id:
        if not ID.fullmatch(args.project_id):
            raise ValueError("Invalid project ID")
        params["project_id"] = args.project_id
    for pool in pages(client, "/pools", params):
        print(json.dumps(pool, ensure_ascii=False, allow_nan=False))


def run_bbox(args, client):
    canonical, _ = bbox_tools.canonical_input(args.input)
    records = fetch_assignments(client, args.pool_id)
    rows = bbox_records(records, canonical)
    if not rows:
        raise ValueError("No submitted or accepted answers found")
    chosen = pick_first(rows)
    blocked = [r for r in chosen if r["status"] == "load_error"]
    if args.review_pool_id and not ID.fullmatch(args.review_pool_id):
        raise ValueError("Invalid review pool ID")
    authors = {}
    for row in rows:
        authors.setdefault(row["photo_id"], set()).add(row["worker_id"])

    def build(stage):
        write_jsonl(stage / "bbox_answers.jsonl", (serial(r) for r in rows))
        write_jsonl(stage / "selected_answers.jsonl", (serial(r) for r in chosen))
        report = {
            "selection": "earliest submitted, then assignment_id, pool_id, task_id",
            "statuses_included": sorted(STATUSES),
            "answers": len(rows), "photos": len(chosen),
            "load_error_photo_ids": [r["photo_id"] for r in blocked],
            "uncertain_photo_ids": [r["photo_id"] for r in chosen if r["status"] == "uncertain"],
            "self_review_prevention": "unavailable_for in review_tasks.jsonl; TSV alone does not enforce",
            "review_pool_id": args.review_pool_id,
            "excluded_authors_scope": "authors of submitted or accepted bbox answers in requested pools",
        }
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not blocked:
            tasks = []
            for row in chosen:
                boxes = [dict(box, shape="rectangle") for box in row["boxes"]]
                task = {
                    "input_values": {
                        "photo_id": row["photo_id"],
                        "image_url": row["image_url"],
                        "boxes": boxes,
                    },
                    "unavailable_for": sorted(authors[row["photo_id"]]),
                }
                if args.review_pool_id:
                    task["pool_id"] = args.review_pool_id
                tasks.append(task)
            write_jsonl(stage / "review_tasks.jsonl", tasks)
            with (stage / "review_tasks.tsv").open("x", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t", lineterminator="\n")
                writer.writerow(["INPUT:photo_id", "INPUT:image_url", "INPUT:boxes"])
                for row in chosen:
                    boxes = [dict(box, shape="rectangle") for box in row["boxes"]]
                    writer.writerow([row["photo_id"], row["image_url"],
                                     json.dumps(boxes, ensure_ascii=False, separators=(",", ":"), allow_nan=False)])
    publish_directory(args.out, build)
    print(f"Saved {len(rows)} answers; selected {len(chosen)} photos in {args.out}")
    if blocked:
        print(f"TSV not created: {len(blocked)} selected answers have load_error; resolve them first.", file=sys.stderr)
        return 2
    return 0


def run_review(args, client):
    rows = fetch_assignments(client, args.pool_id)
    if not rows:
        raise ValueError("No submitted or accepted review answers found")
    def build(stage):
        write_jsonl(stage / "review_answers.jsonl", (serial(r) for r in rows))
    publish_directory(args.out, build)
    print(f"Saved {len(rows)} review answers in {args.out}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REVIEW / "configs/api.json")
    parser.add_argument("--token", help="API token; overrides api_token from config")
    parser.add_argument("--auth-scheme", choices=("OAuth", "ApiKey"), default="OAuth")
    commands = parser.add_subparsers(dest="command", required=True)
    pools = commands.add_parser("pools", help="List pools as JSONL")
    pools.add_argument("--project-id")
    pools.set_defaults(run=run_pools)
    bbox = commands.add_parser("prepare", help="Download bbox answers and prepare review TSV")
    bbox.add_argument("--pool-id", action="append", required=True)
    bbox.add_argument("--input", type=Path, default=bbox_tools.INPUT)
    bbox.add_argument("--review-pool-id", help="Target review pool ID for API task payloads")
    bbox.add_argument("--out", type=Path, required=True)
    bbox.set_defaults(run=run_bbox)
    review = commands.add_parser("download-review", help="Download review answers as JSONL")
    review.add_argument("--pool-id", action="append", required=True)
    review.add_argument("--out", type=Path, required=True)
    review.set_defaults(run=run_review)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        token = args.token if args.token is not None else config.get("api_token")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Provide --token or set api_token in the config before calling the API")
        client = ReadClient(token.strip(), auth_scheme=args.auth_scheme)
        return args.run(args, client) or 0
    except (OSError, ValueError, csv.Error, ApiError, ProvisioningError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
