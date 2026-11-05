#!/usr/bin/env python3
"""Monitor Hugging Face and news feeds for image/video generation and editing releases.

Usage:
    ./latestai.py                    fetch, record new items, rebuild the digest
    ./latestai.py --open             same, then open the digest in a browser
    ./latestai.py install-timer      run automatically via a systemd user timer
    ./latestai.py uninstall-timer    remove that timer
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "state.db"
DIGEST_PATH = DATA_DIR / "digest.html"

USER_AGENT = "LatestAI-monitor/0.1"
MAX_RESPONSE_BYTES = 20_000_000
HF_API = "https://huggingface.co/api"

PIPELINE_SECTIONS = {
    "text-to-image": "Image generation",
    "image-to-image": "Image editing",
    "text-to-video": "Video generation",
    "image-to-video": "Video generation",
    "video-to-video": "Video editing",
}
SECTION_ORDER = [
    "Image generation",
    "Image editing",
    "Video generation",
    "Video editing",
    "Research",
    "News and tool updates",
]
DERIVATIVE_LABELS = {
    "adapter": "LoRA/adapter",
    "quantized": "quantized",
    "finetune": "finetune",
    "merge": "merge",
}
SERVICE_NAME = "latestai"


# ---------------------------------------------------------------- helpers

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_date(value: str | None) -> datetime | None:
    """Parse the RFC 822 and ISO 8601 dates used by feeds and the HF API."""
    if not value:
        return None
    value = value.strip()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def strip_html(text: str | None) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def is_web_url(url: str | None) -> bool:
    return bool(url) and urllib.parse.urlparse(url).scheme in ("http", "https")


def fetch(url: str, retries: int = 1) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError("response too large")
            # Some servers send gzip even though the request did not ask for it.
            return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
        except urllib.error.URLError:
            if attempt == retries:
                raise
        time.sleep(5)
    raise AssertionError("unreachable")


def fetch_json(url: str):
    return json.loads(fetch(url))


# ---------------------------------------------------------------- Hugging Face

def derivative_kind(tags: list[str]) -> str | None:
    """Return how a repo derives from another model, or None if it is original."""
    for tag in tags:
        if tag.startswith("base_model:"):
            kind = tag.split(":")[1]
            if kind in DERIVATIVE_LABELS:
                return kind
    return "adapter" if "lora" in tags else None


def merge_models(models: list[dict]) -> list[dict]:
    """Combine duplicate repos; only some API sorts report every metric."""
    merged: dict[str, dict] = {}
    for model in models:
        existing = merged.get(model["id"])
        if existing is None:
            merged[model["id"]] = dict(model)
            continue
        for key, value in model.items():
            if existing.get(key) in (None, "", 0):
                existing[key] = value
    return list(merged.values())


def keep_model(model: dict, rules: dict, watched: set[str], now: datetime) -> bool:
    """Decide whether a repo is a release worth showing, rather than noise."""
    pipeline = model.get("pipeline_tag")
    if pipeline not in PIPELINE_SECTIONS:
        return False
    tags = model.get("tags") or []
    if "not-for-all-audiences" in tags or re.search(rules["exclude_name"], model["id"]):
        return False
    created = parse_date(model.get("createdAt"))
    if created is None or now - created > timedelta(days=rules["max_age_days"]):
        return False

    org = model["id"].split("/")[0].lower()
    tier = "derivative" if derivative_kind(tags) else "original"
    if org in watched:
        if tier == "original":
            return True
        thresholds = rules["watched_derivative"]
    else:
        thresholds = rules[tier]
    return (
        (model.get("trendingScore") or 0) >= thresholds["trending"]
        or (model.get("likes") or 0) >= thresholds["likes"]
    )


def model_to_item(model: dict) -> dict:
    tags = model.get("tags") or []
    meta = [model["pipeline_tag"]]
    kind = derivative_kind(tags)
    if kind:
        meta.append(DERIVATIVE_LABELS[kind])
    meta += [t.split(":", 1)[1] for t in tags if t.startswith("license:")]
    return {
        "id": f"hf:{model['id']}",
        "section": PIPELINE_SECTIONS[model["pipeline_tag"]],
        "title": model["id"],
        "url": f"https://huggingface.co/{model['id']}",
        "source": "Hugging Face",
        "date": model.get("createdAt"),
        "trending": model.get("trendingScore") or 0,
        "likes": model.get("likes") or 0,
        "meta": meta,
        "summary": "",
    }


def fetch_hf_models(config: dict, pool: ThreadPoolExecutor) -> tuple[list[dict], list[str]]:
    rules = config["huggingface"]
    urls = {}
    for pipeline in PIPELINE_SECTIONS:
        for sort in ("trendingScore", "createdAt"):
            query = urllib.parse.urlencode(
                {"pipeline_tag": pipeline, "sort": sort, "direction": -1, "limit": 100}
            )
            urls[f"HF {pipeline} by {sort}"] = f"{HF_API}/models?{query}"
    for org in rules["watched_orgs"]:
        query = urllib.parse.urlencode(
            {"author": org, "sort": "createdAt", "direction": -1, "limit": 30}
        )
        urls[f"HF org {org}"] = f"{HF_API}/models?{query}"

    futures = {name: pool.submit(fetch_json, url) for name, url in urls.items()}
    models, errors = [], []
    for name, future in futures.items():
        try:
            models += future.result()
        except Exception as err:
            errors.append(f"{name}: {err}")

    watched = {org.lower() for org in rules["watched_orgs"]}
    now = now_utc()
    kept = [m for m in merge_models(models) if keep_model(m, rules, watched, now)]
    return [model_to_item(m) for m in kept], errors


def fetch_hf_papers(config: dict) -> list[dict]:
    rules = config["papers"]
    media = re.compile(config["filters"]["media"])
    items = []
    for entry in fetch_json(f"{HF_API}/daily_papers?limit=100"):
        paper = entry.get("paper") or {}
        title = paper.get("title") or entry.get("title") or ""
        # Abstracts mention image and video work in passing, so match only the subject.
        text = f"{title} {' '.join(paper.get('ai_keywords') or [])}"
        if (paper.get("upvotes") or 0) < rules["min_upvotes"] or not media.search(text):
            continue
        items.append({
            "id": f"paper:{paper['id']}",
            "section": "Research",
            "title": title,
            "url": f"https://huggingface.co/papers/{paper['id']}",
            "source": "HF Daily Papers",
            "date": entry.get("publishedAt"),
            "meta": [f"{paper.get('upvotes')} upvotes"],
            "summary": strip_html(paper.get("ai_summary") or entry.get("summary"))[:300],
        })
    return items


# ---------------------------------------------------------------- ways to try a model

GENERIC_NAME_TOKENS = {
    "image", "images", "video", "videos", "comfy", "comfyui", "base", "dev", "pro", "turbo",
    "lora", "diffusers", "model", "the", "fast", "lite", "hd", "2k", "4k",
}
ZEROGPU_DOCS = "https://huggingface.co/docs/hub/spaces-zerogpu"
PROVIDER_DOCS = "https://huggingface.co/docs/inference-providers/pricing"


def name_tokens(repo_id: str) -> set[str]:
    """Words in a repo name that can identify it, ignoring versions and common words."""
    words = re.split(r"[^a-z0-9]+", repo_id.split("/")[-1].lower())
    return {w for w in words if len(w) > 2 and not w.isdigit() and w not in GENERIC_NAME_TOKENS}


def rank_spaces(model_id: str, spaces: list[str], rules: dict) -> list[str]:
    """Pick the Spaces that look like demos of this model, most likely first.

    The Hub lists every Space that references a model, including calculators and
    all-in-one tools, so a Space must share the model's owner or a name word.
    """
    owner = model_id.split("/")[0].lower()
    tokens = name_tokens(model_id)
    scored = []
    for space in spaces:
        if re.search(rules["exclude_space"], space):
            continue
        same_owner = space.split("/")[0].lower() == owner
        overlap = len(tokens & name_tokens(space))
        if same_owner or overlap:
            scored.append((same_owner, overlap, space))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [space for _, _, space in scored]


def describe_space(info: dict, model_id: str) -> dict | None:
    """Summarize a Space as a demo link, or return None if it cannot be used now."""
    runtime = info.get("runtime") or {}
    stage = runtime.get("stage") or ""
    usable = stage.startswith("RUNNING") or stage == "SLEEPING"
    if not usable or info.get("private") or info.get("disabled") or info.get("gated"):
        return None
    hardware = runtime.get("hardware") or {}
    hardware = hardware.get("current") or hardware.get("requested") or ""
    if (info.get("cardData") or {}).get("hf_oauth"):
        access = "May ask for a free Hugging Face login"
    elif hardware.startswith("zero"):
        access = "No signup needed, limited free GPU time"
    elif hardware.startswith("cpu"):
        access = "No signup needed, but CPU only, so it may rely on an outside service"
    else:
        access = "No signup needed"
    if stage == "SLEEPING":
        access += "; asleep, so the first load is slow"
    return {
        "id": info["id"],
        "url": f"https://huggingface.co/spaces/{info['id']}",
        "access": access,
        "official": info["id"].split("/")[0].lower() == model_id.split("/")[0].lower(),
    }


def live_providers(info: dict) -> list[str]:
    mapping = info.get("inferenceProviderMapping") or {}
    if isinstance(mapping, dict):
        entries = [{"provider": name, **details} for name, details in mapping.items()]
    else:
        entries = mapping
    return [e["provider"] for e in entries if e.get("status") == "live"]


def find_try_options(item: dict, rules: dict, fetch_space) -> dict:
    """Work out how an item's model can be tried. Network errors propagate to the caller."""
    model_id = item["title"]
    query = (
        "expand[]=spaces&expand[]=inferenceProviderMapping&expand[]=gated"
        "&expand[]=trendingScore&expand[]=likes"
    )
    info = fetch_json(f"{HF_API}/models/{model_id}?{query}")
    demos = []
    for space in rank_spaces(model_id, info.get("spaces") or [], rules)[: rules["max_candidates"]]:
        try:
            demo = describe_space(fetch_space(space), model_id)
        except urllib.error.HTTPError:
            continue  # A deleted or private Space is simply not an option.
        if demo:
            demos.append(demo)
    return {
        "demos": demos[: rules["max_demos"]],
        "providers": live_providers(info),
        "gated": bool(info.get("gated")),
        "trending": info.get("trendingScore") or 0,
        "likes": info.get("likes") or 0,
    }


def add_try_options(items: list[dict], config: dict) -> list[str]:
    """Attach try-it options and live stats to Hugging Face items. Returns any errors."""
    rules = config["try"]
    models = [i for i in items if i["id"].startswith("hf:")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        # Many models share Spaces; dedupe lookups so each is requested once per run.
        space_jobs: dict[str, object] = {}

        def fetch_space(space: str) -> dict:
            job = space_jobs.setdefault(
                space, pool.submit(fetch_json, f"{HF_API}/spaces/{space}")
            )
            return job.result()

        # Model lookups use their own pool so waiting on Spaces cannot starve this one.
        with ThreadPoolExecutor(max_workers=8) as model_pool:
            futures = {
                i["title"]: (i, model_pool.submit(find_try_options, i, rules, fetch_space))
                for i in models
            }
            errors = []
            for title, (item, future) in futures.items():
                try:
                    item["try"] = future.result()
                    item["trending"] = item["try"]["trending"]
                    item["likes"] = item["try"]["likes"]
                except Exception as err:
                    errors.append(f"Try links for {title}: {err}")
    return errors


# ---------------------------------------------------------------- feeds

def parse_feed(body: bytes) -> list[dict]:
    """Parse RSS or Atom into {title, url, date, summary} dictionaries."""
    root = ET.fromstring(body)
    entries = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] not in ("item", "entry"):
            continue
        fields: dict[str, str] = {}
        url = None
        for child in element:
            name = child.tag.rsplit("}", 1)[-1]
            if name == "link":
                # Atom keeps the URL in href; RSS keeps it in the element text.
                href = child.get("href")
                if href and child.get("rel", "alternate") == "alternate":
                    url = url or href
                elif child.text and child.text.strip():
                    url = url or child.text.strip()
            elif child.text and name not in fields:
                fields[name] = child.text
        if url is None and fields.get("guid", "").startswith("http"):
            url = fields["guid"].strip()
        entries.append({
            "title": strip_html(fields.get("title")),
            "url": url,
            "date": fields.get("pubDate") or fields.get("published")
                or fields.get("updated") or fields.get("date"),
            "summary": strip_html(
                fields.get("description") or fields.get("summary") or fields.get("content")
            ),
        })
    return entries


def select_feed_entries(entries: list[dict], feed: dict, config: dict, now: datetime) -> list[dict]:
    """Apply the feed's keyword filter and age limit, and build digest items."""
    pattern = config["filters"].get(feed.get("filter", "all"))
    cutoff = now - timedelta(days=config["feeds_max_age_days"])
    items = []
    for entry in entries:
        date = parse_date(entry["date"])
        if not entry["title"] or not is_web_url(entry["url"]) or date is None or date < cutoff:
            continue
        if pattern and not re.search(pattern, f"{entry['title']} {entry['summary'][:500]}"):
            continue
        prefix = feed.get("title_prefix", "")
        title = entry["title"] if entry["title"].startswith(prefix) else prefix + entry["title"]
        items.append({
            "id": f"feed:{entry['url']}",
            "section": "News and tool updates",
            "title": title,
            "url": entry["url"],
            "source": feed["name"],
            "date": date.isoformat(),
            "meta": [],
            "summary": entry["summary"][:300],
        })
    return items


def fetch_feed(feed: dict, config: dict) -> list[dict]:
    entries = parse_feed(fetch(feed["url"]))
    if not entries:
        raise ValueError("no entries found; the feed may have moved")
    return select_feed_entries(entries, feed, config, now_utc())


# ---------------------------------------------------------------- state

def open_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE IF NOT EXISTS items ("
        "id TEXT PRIMARY KEY, first_seen TEXT NOT NULL, data TEXT NOT NULL)"
    )
    return db


def record_items(db: sqlite3.Connection, items: list[dict], now: datetime) -> list[dict]:
    """Store items and return the ones not seen before. Known items get fresh metrics."""
    new = []
    with db:
        for item in items:
            payload = json.dumps(item)
            cursor = db.execute(
                "INSERT OR IGNORE INTO items (id, first_seen, data) VALUES (?, ?, ?)",
                (item["id"], now.isoformat(), payload),
            )
            if cursor.rowcount:
                new.append(item)
            else:
                db.execute("UPDATE items SET data = ? WHERE id = ?", (payload, item["id"]))
    return new


def recent_items(db: sqlite3.Connection, days: int, now: datetime) -> list[dict]:
    cutoff = (now - timedelta(days=days)).isoformat()
    rows = db.execute(
        "SELECT data, first_seen FROM items WHERE first_seen >= ?", (cutoff,)
    ).fetchall()
    return [{**json.loads(data), "first_seen": first_seen} for data, first_seen in rows]


# ---------------------------------------------------------------- digest

STYLE = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--accent:#a8431a;--badge:#fbe9df}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecebe6;--muted:#a09f98;
--line:#33332f;--accent:#f0935f;--badge:#43281a}}
*{box-sizing:border-box}
body{margin:0 auto;max-width:980px;padding:32px 16px 64px;background:var(--bg);color:var(--fg);
font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
h1{font-size:2.5rem;line-height:1.2;margin:0}
h2{font-size:2rem;line-height:1.25;color:var(--accent);margin:56px 0 6px;padding-bottom:8px;
border-bottom:3px solid var(--accent)}
h2 small{font-size:1rem;font-weight:700;background:var(--badge);padding:2px 8px;border-radius:10px;
margin-left:6px;vertical-align:middle}
.sub{color:var(--muted);font-size:1.2rem;margin:4px 0 0}
li>a{font-size:1.35rem}
ul{list-style:none;margin:0;padding:0}
li{padding:10px 0;border-bottom:1px solid var(--line)}
a{color:var(--fg);font-weight:600;text-decoration:none;overflow-wrap:anywhere}
a:hover{color:var(--accent);text-decoration:underline}
.meta{color:var(--muted);font-size:1.1rem}
.summary{font-size:1.2rem;margin:2px 0 0}
.new{background:var(--badge);color:var(--accent);font-size:.9rem;font-weight:700;
padding:1px 6px;border-radius:4px;margin-right:6px}
.try{margin-top:6px;font-size:1.2rem;color:var(--muted)}
.try div{margin-top:3px}
.try a.btn{display:inline-block;padding:1px 9px;margin-right:6px;border:1px solid var(--accent);
border-radius:5px;color:var(--accent);font-weight:600}
.try a.btn:hover{background:var(--badge);text-decoration:none}
.try a.plain{color:inherit;font-weight:400;text-decoration:underline}
.errors{margin-top:20px;padding:10px 14px;border:1px solid var(--accent);border-radius:6px;font-size:1.1rem}
.errors ul{margin-top:4px}.errors li{padding:2px 0;border:0}
"""


def link(url: str, text: str, css: str) -> str:
    return (
        f'<a class="{css}" href="{html.escape(url, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">{html.escape(text)}</a>'
    )


def render_try(options: dict | None, model_url: str) -> str:
    """Show how to try a model and what access each route needs."""
    if options is None:
        return ""
    rows = []
    for demo in options["demos"]:
        label = "Try official demo" if demo["official"] else "Try community demo"
        rows.append(
            f'<div>{link(demo["url"], label, "btn")}{html.escape(demo["id"])}: '
            f'{html.escape(demo["access"])}</div>'
        )
    if not options["demos"]:
        rows.append(
            "<div>No live demo found. "
            f'{link(model_url + "/tree/main", "Download the files", "plain")} to run it yourself.</div>'
        )
    if options["providers"]:
        names = ", ".join(options["providers"])
        rows.append(
            f"<div>Hosted API via {html.escape(names)}: "
            f'{link(PROVIDER_DOCS, "needs a Hugging Face account and paid credits", "plain")}.</div>'
        )
    if options["gated"]:
        rows.append("<div>Downloading the files needs a Hugging Face login and accepting the terms.</div>")
    return f'<div class="try">{"".join(rows)}</div>'


def render_item(item: dict, new_ids: set[str]) -> str:
    date = parse_date(item.get("date"))
    stats = []
    if "trending" in item:
        likes = item["likes"]
        stats = [f"trending {item['trending']:g}", f"{likes} like{'' if likes == 1 else 's'}"]
    meta = stats + [item["source"]] + ([date.strftime("%b %-d")] if date else []) + item["meta"]
    badge = '<span class="new">NEW</span>' if item["id"] in new_ids else ""
    summary = (
        f'<p class="summary">{html.escape(item["summary"])}</p>' if item["summary"] else ""
    )
    return (
        f'<li>{badge}<a href="{html.escape(item["url"], quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">{html.escape(item["title"])}</a>'
        f'<div class="meta">{html.escape(" · ".join(meta))}</div>{summary}'
        f'{render_try(item.get("try"), item["url"])}</li>'
    )


def sort_key(item: dict, new_ids: set[str]) -> tuple:
    """Rank models by trending score; papers and news have none, so newest come first."""
    date = parse_date(item.get("date"))
    timestamp = date.timestamp() if date else 0
    if "trending" in item:
        return (item["trending"], item["likes"], timestamp)
    return (item["id"] in new_ids, timestamp)


def render_digest(items: list[dict], new_ids: set[str], errors: list[str],
                  days: int, now: datetime) -> str:
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>Latest AI</title><style>{STYLE}</style></head><body>",
        "<h1>Latest image and video AI</h1>",
        f'<p class="sub">Updated {now.astimezone():%b %-d, %-I:%M %p} · '
        f"{len(new_ids)} new · first seen in the last {days} days</p>",
    ]
    checked = [i["try"] for i in items if i.get("try")]
    if checked:
        with_demo = sum(1 for options in checked if options["demos"])
        parts.append(
            f'<p class="sub">Models are ranked by Hugging Face trending score within each section. '
            f'{with_demo} of {len(checked)} have a live demo. Most demos run on '
            f'{link(ZEROGPU_DOCS, "ZeroGPU", "plain")}: no account needed, about 2 minutes of GPU '
            "time a day, or 5 with a free account.</p>"
        )
    if errors:
        rows = "".join(f"<li>{html.escape(e)}</li>" for e in errors)
        parts.append(
            f'<div class="errors"><strong>{len(errors)} source(s) failed on this run, '
            f"so this page may be missing items:</strong><ul>{rows}</ul></div>"
        )
    for section in SECTION_ORDER:
        rows = sorted((i for i in items if i["section"] == section), key=lambda i: sort_key(i, new_ids), reverse=True)
        if rows:
            body = "".join(render_item(i, new_ids) for i in rows)
            parts.append(f"<h2>{section} <small>{len(rows)}</small></h2><ul>{body}</ul>")
    parts.append("</body></html>")
    return "".join(parts)


def write_atomic(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
    Path(handle.name).replace(path)


def notify(new_items: list[dict]) -> None:
    if not new_items or not shutil.which("notify-send"):
        return
    # Model releases are what the monitor exists for, so list them ahead of news.
    ranked = sorted(new_items, key=lambda i: SECTION_ORDER.index(i["section"]))
    body = "\n".join(i["title"][:80] for i in ranked[:5])
    subprocess.run(
        ["notify-send", "--", f"Latest AI: {len(new_items)} new", body],
        check=False, capture_output=True,
    )


# ---------------------------------------------------------------- run

def collect(config: dict) -> tuple[list[dict], list[str], int]:
    """Return (items, errors, number of sources attempted)."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        papers = pool.submit(fetch_hf_papers, config)
        feeds = {f["name"]: pool.submit(fetch_feed, f, config) for f in config["feeds"]}
        items, errors = fetch_hf_models(config, pool)
        attempted = len(config["huggingface"]["watched_orgs"]) + 2 * len(PIPELINE_SECTIONS)
        for name, future in [("HF Daily Papers", papers), *feeds.items()]:
            attempted += 1
            try:
                items += future.result()
            except Exception as err:
                errors.append(f"{name}: {err}")
    return items, errors, attempted


def command_run(args: argparse.Namespace) -> int:
    config = json.loads(CONFIG_PATH.read_text())
    items, errors, attempted = collect(config)
    now = now_utc()

    db = open_db()
    new_items = record_items(db, items, now)
    days = config["digest_days"]
    new_ids = {i["id"] for i in new_items}
    digest_items = recent_items(db, days, now)
    db.close()
    errors += add_try_options(digest_items, config)
    write_atomic(DIGEST_PATH, render_digest(digest_items, new_ids, errors, days, now))
    options_by_id = {i["id"]: i.get("try") for i in digest_items}

    new_items.sort(key=lambda i: sort_key(i, new_ids), reverse=True)
    for section in SECTION_ORDER:
        rows = [i for i in new_items if i["section"] == section]
        if rows:
            print(f"\n{section} ({len(rows)} new)")
            for item in rows:
                print(f"  {item['title'][:90]}\n    {item['url']}")
                for demo in (options_by_id.get(item["id"]) or {}).get("demos", []):
                    print(f"    try: {demo['url']}")
    print(f"\n{len(new_items)} new, {len(items)} matched. Digest: {DIGEST_PATH}")
    for error in errors:
        print(f"warning: {error}", file=sys.stderr)

    if not args.no_notify:
        notify(new_items)
    if args.open:
        webbrowser.open(DIGEST_PATH.as_uri())
    if len(errors) >= attempted:
        print("error: every source failed", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------- scheduling

def unit_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def render_units(hours: int) -> dict[str, str]:
    return {
        f"{SERVICE_NAME}.service": (
            "[Unit]\nDescription=Check for new image and video AI releases\n"
            "Wants=network-online.target\nAfter=network-online.target\n\n"
            f'[Service]\nType=oneshot\nExecStart="{sys.executable}" "{ROOT / "latestai.py"}"\n'
        ),
        f"{SERVICE_NAME}.timer": (
            "[Unit]\nDescription=Check for new image and video AI releases on a schedule\n\n"
            f"[Timer]\nOnCalendar=*-*-* 00/{hours}:00:00\nPersistent=true\n"
            "RandomizedDelaySec=5min\n\n[Install]\nWantedBy=timers.target\n"
        ),
    }


def command_install_timer(args: argparse.Namespace) -> int:
    if not 1 <= args.every_hours <= 23:
        print("error: --every-hours must be between 1 and 23", file=sys.stderr)
        return 2
    directory = unit_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in render_units(args.every_hours).items():
        (directory / name).write_text(text)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", f"{SERVICE_NAME}.timer"], check=True
    )
    print(f"Checking every {args.every_hours} hours. Logs: journalctl --user -u {SERVICE_NAME}")
    return 0


def command_uninstall_timer(_args: argparse.Namespace) -> int:
    subprocess.run(
        ["systemctl", "--user", "disable", "--now", f"{SERVICE_NAME}.timer"], check=False
    )
    for suffix in ("service", "timer"):
        (unit_dir() / f"{SERVICE_NAME}.{suffix}").unlink(missing_ok=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    print("Timer removed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-notify", action="store_true", help="skip the desktop notification")
    parser.add_argument("--open", action="store_true", help="open the digest in a browser")
    parser.set_defaults(handler=command_run)
    commands = parser.add_subparsers()
    install = commands.add_parser("install-timer", help="run automatically on a schedule")
    install.add_argument("--every-hours", type=int, default=6)
    install.set_defaults(handler=command_install_timer)
    uninstall = commands.add_parser("uninstall-timer", help="remove the schedule")
    uninstall.set_defaults(handler=command_uninstall_timer)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
