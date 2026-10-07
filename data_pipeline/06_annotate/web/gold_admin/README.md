# Gold annotation — admin dashboard

Read-only monitoring for the Stage A (and optional Stage B ranking) gold annotation. Static site, vanilla JS,
reads the `gold_*` tables and views with the anon key (RLS allows select). Deploy as its
own Netlify site, separate from the two worker sites.

## What it shows

- **Overview cards** — candidates, how many reached the rater target, scores collected,
  videos passing Stage A, rankable clips, clips ranked.
- **Stage A** — raters-per-candidate histogram (progress), human score distribution, and
  inter-rater agreement (within-1, exact, pass/fail agreement) over candidates with ≥2
  raters.
- **Annotators** — per-worker count, mean, score spread, and mean absolute deviation from
  the peer median. Low-effort workers are flagged red (no score variance, or high
  deviation) so you can reject or exclude them.
- **Stage B** (optional within-video ranking stage, not used for the released benchmark) —
  passed / rankable / ranked counts and "no clear strongest" rate.
- **Spot-check** — enter a video id to play the clip and see each candidate's per-rater
  scores next to Gemini's own score.

## Run / deploy

Serve over HTTP (not `file://`):

```bash
cp manifest.example.json manifest.json   # fill in Supabase URL / anon key / R2 base
python -m http.server 8080              # then open http://localhost:8080/
```

Deploy the folder to Netlify like the worker sites. It uses only the publishable (anon)
key, so it exposes nothing the annotation sites don't already; still, treat the URL as
internal — keep it off the Prolific studies.

The dashboard pages through all rows (PostgREST caps single requests at 1000), so counts
are exact. Click **Refresh** to re-pull.
