# Opt-out and Takedown

ReactionBench is built from publicly posted reaction videos. Being public does not mean a
creator agreed to be in a research benchmark, so any creator (or rights holder) can opt out.
Annotators can also ask for their contributions to be removed.

**Contact:** Yibo Zhao ([yibozhao@mit.edu](mailto:yibozhao@mit.edu)) and Ao Qu
([qua@mit.edu](mailto:qua@mit.edu)). Please use the subject line "ReactionBench takedown".

## Who can ask

* **Creators** who appear in, or published, a reaction recording used in the benchmark.
* **Rights holders** of stimulus content shown in a clip.
* **Annotators** who contributed ratings or rationale references (identify yourself through the
  platform you participated on; we hold the mapping from platform ID to release pseudonym
  privately and never publish it).

## What to send

1. Who you are and how you relate to the content (creator, authorized representative, rights
   holder, annotator).
2. What to remove: a channel or video URL, or ReactionBench identifiers (`video_id`,
   `candidate_id`, clip key such as `face/46052.mp4`). If you do not know the identifiers, a
   channel URL is enough; we look them up.
3. The scope: specific clips, all clips from a channel, or your annotations.

You do not need to give a reason.

## What we do

| Step | Action |
|---|---|
| 1 | Acknowledge the request and confirm the scope. |
| 2 | Disable access to the affected media in the gated store immediately (before verification is finished). |
| 3 | Remove the items from the task files and annotation files in the next data release, record the removal (identifiers only, no reason) in the changelog on the dataset page, and bump the dataset version. |
| 4 | Remove the creator from internal construction metadata and source manifests, so the content is not re-collected. |
| 5 | Report back to the requester. |

Results in already published papers cannot be withdrawn, but scoring scripts report the
release version, and we publish per-version numbers for the reference models so that later
work can compare on the reduced set.

## For dataset users

If an item you hold is removed, delete your copies of its media and do not report new results
on it (see [DATA_LICENSE.md](DATA_LICENSE.md), §3.6).
