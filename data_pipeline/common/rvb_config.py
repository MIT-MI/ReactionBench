"""Shared configuration for the ReactionBench data-construction pipeline.

Every script in data_pipeline/ imports this module first. It

  * loads ``data_pipeline/.env`` (or the file named by ``RVB_ENV_FILE``) into ``os.environ``
    without overriding variables that are already set,
  * puts ``common/`` and every stage directory (``01_collect`` ... ``07_export``) on
    ``sys.path`` so cross-stage helpers (e.g. ``build_pilot_frame.default_data_root``) import
    by bare module name, and
  * exposes the filesystem roots the scripts use as argparse defaults.

No path, credential or endpoint is hardcoded here: everything comes from the environment,
with repo-relative fallbacks so ``--help`` works on a fresh checkout. See ``.env.example``.

Roots
-----
RVB_RAW_ROOT    raw downloads + per-domain pipeline outputs (``raw/``, ``face/``, ``stimuli/`` ...)
RVB_DATA_ROOT   the consolidated clip pool (``face_origin/``, ``stimuli/``, ``_meta/`` ...)
RVB_RECUT_ROOT  re-aligned (re-cut) stimuli for desynchronised Patreon clips
RVB_WORK_DIR    intermediate tables, nominations, selections, task exports
RVB_LOG_DIR     structured JSON run logs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = PIPELINE_ROOT / "common"
STAGE_DIRS = sorted(p for p in PIPELINE_ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit())


def _load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE parser (no dependency on python-dotenv). Existing env wins."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and value:
            os.environ.setdefault(key, value)


_load_dotenv(Path(os.environ.get("RVB_ENV_FILE", PIPELINE_ROOT / ".env")))

for _d in [COMMON_DIR, *STAGE_DIRS, PIPELINE_ROOT / "02_align" / "talking_head"]:
    if str(_d) not in sys.path:
        sys.path.append(str(_d))


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


RAW_ROOT = _env_path("RVB_RAW_ROOT", PIPELINE_ROOT / "data" / "human_reaction_data")
DATA_ROOT = _env_path("RVB_DATA_ROOT", PIPELINE_ROOT / "data" / "clean_human_reaction_data")
RECUT_ROOT = _env_path("RVB_RECUT_ROOT", DATA_ROOT.parent / f"{DATA_ROOT.name}_recut")
WORK_DIR = _env_path("RVB_WORK_DIR", PIPELINE_ROOT / "work")
LOG_DIR = _env_path("RVB_LOG_DIR", WORK_DIR / "logs")
META_DIR = DATA_ROOT / "_meta"

# Cloudflare R2 (clip hosting for the annotation sites). Empty until configured.
R2_BUCKET = os.environ.get("R2_BUCKET", "")
_r2_pub = os.environ.get("R2_PUBLIC_BASE_URL", "")
R2_PUBLIC = (_r2_pub.rstrip("/") + "/") if _r2_pub else ""


def meta_file(name: str) -> Path:
    """Pool-level metadata table (mapping.csv, mapping_lineage_part1.csv, youtuber_id_map.json...).

    build_reaction_hallo3.py / split_clean_ids_part1.py write these at the pool root; the
    downstream stages read them from ``_meta/``. Prefer ``_meta/``, fall back to the root.
    """
    for cand in (META_DIR / name, DATA_ROOT / name):
        if cand.exists():
            return cand
    return META_DIR / name


def work(*parts: str) -> Path:
    """Path under the work directory (``RVB_WORK_DIR``)."""
    return WORK_DIR.joinpath(*parts)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"[config] environment variable {name} is not set; copy data_pipeline/.env.example "
                 f"to data_pipeline/.env and fill it in")
    return value


def supabase(write: bool = False) -> Tuple[str, str]:
    """(url, key). Reads prefer the anon key; writes require the service-role secret key."""
    url = (os.environ.get("SUPABASE_URL") or os.environ.get("REACTIONBENCH_SUPABASE_URL")
           or require_env("SUPABASE_URL")).rstrip("/")
    if write:
        return url, require_env("SUPABASE_SECRET_KEY")
    key = (os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_ANON_KEY")
           or os.environ.get("REACTIONBENCH_SUPABASE_ANON_KEY"))
    if not key:
        sys.exit("[config] set SUPABASE_ANON_KEY (read-only) or SUPABASE_SECRET_KEY")
    return url, key


def r2_public_base() -> str:
    base = require_env("R2_PUBLIC_BASE_URL")
    return base if base.endswith("/") else base + "/"


def r2_client():
    """boto3 S3 client for the Cloudflare R2 bucket (write access)."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=require_env("R2_S3_API"),
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID") or require_env("R2_ACCESS_SECRET_ID"),
        aws_secret_access_key=require_env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def r2_bucket() -> str:
    return require_env("R2_BUCKET")


def gcp_project() -> Optional[str]:
    return os.environ.get("GOOGLE_CLOUD_PROJECT")


def gcp_location(default: str = "global") -> str:
    return os.environ.get("GOOGLE_CLOUD_LOCATION", default)
