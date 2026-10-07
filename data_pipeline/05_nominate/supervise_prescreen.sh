#!/bin/bash
# Supervise the full-scale pre-screen across ADC reauth expiries.
#
# Why this exists: an ADC `authorized_user` credential can have a reauth window of
# roughly an hour, while the pre-screen is upload-bandwidth-bound and takes ~6.5 h. Where
# service-account keys are disabled by org policy
# (constraints/iam.managed.disableServiceAccountKeyCreation), the credential WILL
# expire mid-run and only a human can renew it.
#
# When it expires, requests fail in milliseconds, so an unsupervised runner burns
# thousands of ids into error records in minutes. Those ids are retried on resume and
# cost nothing, but the run stalls until someone relaunches it.
#
# This supervisor:
#   1. probes auth with a single cheap call before each launch;
#   2. kills the runner the moment errors burst (expiry detected in <=30 s), so no
#      further ids are burned;
#   3. waits, re-probing, until a human has reauthed -- then relaunches automatically.
#
# Net effect: the operator's only job is `gcloud auth application-default login`
# whenever convenient. Progress resumes on its own, whenever that happens.
#
# Safe to run repeatedly: the underlying runner is resumable and skips completed ids.
# Usage: data_pipeline/05_nominate/supervise_prescreen.sh [total_expected]
# Env:   RVB_WORK_DIR / RVB_LOG_DIR (default data_pipeline/work, work/logs), PYTHON,
#        PRESCREEN_WORKERS (default 4; 8 workers hit the Vertex concurrency quota in the
#        original run and the tail was finished at 2).

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PIPE="$(dirname "$HERE")"
if [ -f "$PIPE/.env" ]; then set -a; . "$PIPE/.env"; set +a; fi

TOTAL=${1:-14582}
PY=${PYTHON:-python}
RUNNER=$HERE/run_gemini_variant.py
WORK=${RVB_WORK_DIR:-$PIPE/work}
LOGS=${RVB_LOG_DIR:-$WORK/logs}
WORKERS=${PRESCREEN_WORKERS:-4}
OUT_DIR=$WORK/prescreen
JSONL=$OUT_DIR/prescreen_flash_low.jsonl
LOG=$LOGS/prescreen_flash_low_run.log
PROBE_DIR=${TMPDIR:-/tmp}/prescreen_probe
BURST_THRESHOLD=20      # new error records per poll that means "auth is gone"
POLL_S=30
REPROBE_S=120
# Requests carry no client-side timeout, so a wedged connection blocks its worker
# thread forever: no error, no progress, and the burst check above never fires.
# Throughput is genuinely bursty (zero-progress windows of ~2 min are normal when
# several large clips are uploading at once), so this threshold is deliberately far
# above any lull observed in practice -- it catches a wedge, not a slow patch.
STALL_POLLS=20          # consecutive polls with zero completions => restart (20*30s = 10 min)

mkdir -p "$OUT_DIR" "$LOGS" "$PROBE_DIR"

ok_count() { grep -c '"error": null' "$JSONL" 2>/dev/null || echo 0; }
err_count() {
  local tot; tot=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
  echo $(( tot - $(ok_count) ))
}

# One clip, throwaway output dir, so a dead credential is detected for ~half a cent
# instead of by burning several thousand ids.
probe_auth() {
  rm -f "$PROBE_DIR"/authprobe.jsonl
  timeout 180 $PY "$RUNNER" --variant authprobe \
      --model gemini-3.6-flash --media_resolution LOW --fps 1 \
      --ids_file "$WORK/prescreen_ids_excl_pilot.txt" \
      --sample_csv "$WORK/prescreen_pool.csv" \
      --out_dir "$PROBE_DIR" --log_dir "$PROBE_DIR" \
      --workers 1 --top_k 5 --limit 1 >"$PROBE_DIR"/probe.log 2>&1
  grep -q '"error": null' "$PROBE_DIR"/authprobe.jsonl 2>/dev/null
}

echo "[supervisor] start $(date -u +%FT%TZ), $(ok_count)/$TOTAL done"

while true; do
  done_n=$(ok_count)
  if [ "$done_n" -ge "$TOTAL" ]; then
    echo "[supervisor] COMPLETE $done_n/$TOTAL at $(date -u +%FT%TZ)"
    exit 0
  fi

  if ! probe_auth; then
    echo "[supervisor] auth probe FAILED at $(date -u +%FT%TZ) ($done_n/$TOTAL done) -- \
needs: gcloud auth application-default login --no-launch-browser. Re-probing in ${REPROBE_S}s."
    sleep "$REPROBE_S"
    continue
  fi

  echo "[supervisor] auth ok, launching runner at $done_n/$TOTAL"
  $PY "$RUNNER" --variant prescreen_flash_low \
      --model gemini-3.6-flash --media_resolution LOW --fps 1 \
      --ids_file "$WORK/prescreen_ids_excl_pilot.txt" \
      --sample_csv "$WORK/prescreen_pool.csv" \
      --out_dir "$OUT_DIR" --log_dir "$LOGS" --workers "$WORKERS" --top_k 5 >>"$LOG" 2>&1 &
  pid=$!

  prev_err=$(err_count)
  prev_ok=$(ok_count)
  flat=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_S"
    cur_err=$(err_count)
    new=$(( cur_err - prev_err ))
    prev_err=$cur_err
    if [ "$new" -gt "$BURST_THRESHOLD" ]; then
      echo "[supervisor] error burst ($new in ${POLL_S}s) at $(date -u +%FT%TZ) -- \
killing runner at $(ok_count)/$TOTAL, credential is almost certainly expired"
      kill "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      break
    fi

    cur_ok=$(ok_count)
    if [ "$cur_ok" -gt "$prev_ok" ]; then
      flat=0
    else
      flat=$(( flat + 1 ))
    fi
    prev_ok=$cur_ok
    if [ "$flat" -ge "$STALL_POLLS" ]; then
      echo "[supervisor] STALL: no completions for $(( STALL_POLLS * POLL_S ))s at \
$(date -u +%FT%TZ) -- restarting runner at $cur_ok/$TOTAL"
      kill "$pid" 2>/dev/null
      sleep 5
      kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      break
    fi
  done
  wait "$pid" 2>/dev/null
  echo "[supervisor] runner stopped at $(ok_count)/$TOTAL, $(date -u +%FT%TZ)"
  sleep 5
done
