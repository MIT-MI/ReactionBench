/* Phase 0 blind review — vanilla JS, no CDN, no cloud.
 *
 * Candidates are grouped by clip rather than shown in isolation. Judging "is this a clear
 * reaction" requires the reviewer's baseline for *this person* — some creators are animated
 * throughout and some are deadpan — so the clip plays from the start with the candidate
 * marked on a timeline, and the reviewer sees every candidate of a clip together.
 *
 * Grouping costs nothing in blinding: what must stay hidden is which variant proposed a
 * candidate, not which clip it came from. Clip order is still the seeded shuffle.
 *
 * Every judgement is POSTed the moment it is made, so closing the tab loses nothing.
 */

const state = {
  reviewer: null,
  clipOrder: [],       // clip_id[], in the pool's shuffled clip order
  byClip: new Map(),   // clip_id -> candidate[] sorted by start_s
  scores: new Map(),
  sweeps: new Map(),
  clipIndex: 0,
  candIndex: 0,
  contextReplays: 0,
  cueType: null,
  marks: [],
  mode: "score",
  stopAt: null,
};

const $ = (id) => document.getElementById(id);
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

async function boot() {
  const [pool, saved] = await Promise.all([
    fetch("/api/pool").then((r) => r.json()),
    fetch("/api/state").then((r) => r.json()),
  ]);
  state.reviewer = pool.reviewer;
  saved.scores.forEach((r) => state.scores.set(r.candidate_id, r));
  saved.sweeps.forEach((r) => state.sweeps.set(r.clip_id, r));

  // Preserve the seeded shuffle at the clip level; order candidates within a clip by time.
  pool.candidates.forEach((c) => {
    if (!state.byClip.has(c.clip_id)) {
      state.byClip.set(c.clip_id, []);
      state.clipOrder.push(c.clip_id);
    }
    state.byClip.get(c.clip_id).push(c);
  });
  state.byClip.forEach((list) => list.sort((a, b) => a.start_s - b.start_s));

  $("subtitle").textContent =
    `reviewer ${state.reviewer} · ${pool.candidates.length} candidates · ${state.clipOrder.length} clips`;

  seekToFirstUnscored();
  seekToFirstUnswept();
  renderScore();
  renderSweep();
  wire();
}

// Resume the sweep at the first clip with no saved record (marked or empty). A saved
// empty is "done" -- that is a real answer -- so it is skipped on resume too.
function seekToFirstUnswept() {
  const i = state.clipOrder.findIndex((c) => !state.sweeps.has(c));
  state.sweepIndex = i === -1 ? Math.max(0, state.clipOrder.length - 1) : i;
}

function seekToFirstUnscored() {
  for (let i = 0; i < state.clipOrder.length; i += 1) {
    const cands = state.byClip.get(state.clipOrder[i]);
    for (let j = 0; j < cands.length; j += 1) {
      if (!state.scores.has(cands[j].candidate_id)) {
        state.clipIndex = i;
        state.candIndex = j;
        return;
      }
    }
  }
  state.clipIndex = state.clipOrder.length - 1;
  state.candIndex = state.byClip.get(state.clipOrder[state.clipIndex]).length - 1;
}

const currentClip = () => state.clipOrder[state.clipIndex];
const currentCands = () => state.byClip.get(currentClip()) || [];
const currentCand = () => currentCands()[state.candIndex];

/* ------------------------------------------------------------------ playback */

function play(video, from, until) {
  if (state.stopAt) video.removeEventListener("timeupdate", state.stopAt);
  const guard = () => {
    if (until != null && video.currentTime >= until) {
      video.pause();
      video.removeEventListener("timeupdate", guard);
      state.stopAt = null;
    }
  };
  state.stopAt = until == null ? null : guard;
  const begin = () => {
    video.currentTime = Math.max(0, from);
    if (until != null) video.addEventListener("timeupdate", guard);
    video.play().catch(() => {});
  };
  if (video.readyState >= 1) begin();
  else video.addEventListener("loadedmetadata", begin, { once: true });
}

/* ------------------------------------------------------------------- scoring */

function renderTimeline() {
  const cand = currentCand();
  const strip = $("timeline");
  strip.innerHTML = "";
  currentCands().forEach((c, i) => {
    const el = document.createElement("div");
    el.className = "seg" + (i === state.candIndex ? " current" : "") +
      (state.scores.has(c.candidate_id) ? " scored" : "");
    el.style.left = `${(100 * c.start_s) / cand.clip_duration_s}%`;
    el.style.width = `${(100 * (c.end_s - c.start_s)) / cand.clip_duration_s}%`;
    el.title = `${fmt(c.start_s)}–${fmt(c.end_s)}`;
    el.onclick = () => { state.candIndex = i; renderScore(); };
    strip.appendChild(el);
  });
  const head = document.createElement("div");
  head.className = "playhead";
  head.id = "playhead";
  strip.appendChild(head);
}

function renderScore() {
  const cand = currentCand();
  if (!cand) return;
  const done = state.scores.size;
  const total = [...state.byClip.values()].reduce((n, l) => n + l.length, 0);
  $("scoreBar").style.width = `${(100 * done) / total}%`;
  $("scoreProgress").textContent =
    `${done} / ${total} scored · clip ${state.clipIndex + 1}/${state.clipOrder.length} · ` +
    `candidate ${state.candIndex + 1}/${currentCands().length} in this clip`;
  $("scoreMeta").textContent =
    `candidate ${fmt(cand.start_s)}–${fmt(cand.end_s)} (${(cand.end_s - cand.start_s).toFixed(1)}s) ` +
    `· clip is ${cand.clip_duration_s.toFixed(1)}s long`;

  const existing = state.scores.get(cand.candidate_id);
  $("scoreNote").value = existing?.note || "";
  state.cueType = existing?.cue_type || null;
  $("scoreSaved").textContent = existing ? `saved: ${existing.score}` : "";
  document.querySelectorAll(".score").forEach((b) =>
    b.classList.toggle("chosen", !!existing && Number(b.dataset.score) === existing.score));
  document.querySelectorAll(".cue").forEach((b) =>
    b.classList.toggle("chosen", b.dataset.cue === state.cueType));

  state.contextReplays = 0;
  const video = $("scoreVideo");
  const src = `/video/${cand.clip_id}.mp4`;
  const isNewClip = !video.src.endsWith(src);
  if (isNewClip) video.src = src;
  renderTimeline();

  // A new clip plays from the top so the reviewer sees this person's baseline before
  // being asked to call anything strong. Later candidates in the same clip start a few
  // seconds early, since the baseline is already established.
  if (isNewClip) play(video, 0, cand.end_s);
  else play(video, Math.max(0, cand.start_s - 4), cand.end_s);
}

async function submitScore(score) {
  const cand = currentCand();
  if (!cand) return;
  const body = {
    candidate_id: cand.candidate_id,
    score,
    cue_type: state.cueType,
    note: $("scoreNote").value,
    context_replays: state.contextReplays,
  };
  const res = await fetch("/api/score", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }).then((r) => r.json());
  if (res.error) { $("scoreSaved").textContent = `NOT SAVED: ${res.error}`; return; }
  state.scores.set(cand.candidate_id, { ...body, clip_id: cand.clip_id });
  advance();
}

function advance() {
  if (state.candIndex < currentCands().length - 1) state.candIndex += 1;
  else if (state.clipIndex < state.clipOrder.length - 1) { state.clipIndex += 1; state.candIndex = 0; }
  renderScore();
}

function retreat() {
  if (state.candIndex > 0) state.candIndex -= 1;
  else if (state.clipIndex > 0) {
    state.clipIndex -= 1;
    state.candIndex = currentCands().length - 1;
  }
  renderScore();
}

/* --------------------------------------------------------------------- sweep */

function renderSweep() {
  const clip = state.clipOrder[state.sweepIndex ?? 0];
  if (!clip) return;
  const done = state.sweeps.size;
  const skipped = [...state.sweeps.values()].filter((s) => s.skip_reason).length;
  $("sweepBar").style.width = `${(100 * done) / state.clipOrder.length}%`;
  $("sweepProgress").textContent =
    `${done} / ${state.clipOrder.length} clips done (${skipped} skipped) · showing #${(state.sweepIndex ?? 0) + 1}`;
  $("sweepMeta").textContent = `clip ${clip}`;

  const existing = state.sweeps.get(clip);
  state.marks = existing && existing.marks ? existing.marks.map((m) => ({ ...m })) : [];
  $("sweepSaved").textContent = existing
    ? (existing.skip_reason ? `saved: skipped (${existing.skip_reason})` : `saved: ${existing.marks.length} mark(s)`)
    : "";
  renderMarks();

  const video = $("sweepVideo");
  const src = `/video/${clip}.mp4`;
  if (!video.src.endsWith(src)) video.src = src;
  video.playbackRate = Number($("btnSpeed").dataset.rate || 2);
  video.currentTime = 0;
}

function renderMarks() {
  const ul = $("sweepMarks");
  ul.innerHTML = "";
  state.marks.slice().sort((a, b) => a.t_s - b.t_s).forEach((m) => {
    const li = document.createElement("li");
    const when = m.end_s != null ? `${fmt(m.t_s)}–${fmt(m.end_s)}` : fmt(m.t_s);
    li.innerHTML = `<span>${when}</span> <b>score ≥${m.score}</b>`;
    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "remove";
    del.onclick = () => { state.marks = state.marks.filter((x) => x !== m); renderMarks(); };
    li.appendChild(del);
    ul.appendChild(li);
  });
  if (!state.marks.length) ul.innerHTML = '<li class="empty">no marks — saving empty is a real answer</li>';
}

async function saveSweep() {
  const clip = state.clipOrder[state.sweepIndex ?? 0];
  const res = await fetch("/api/sweep", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clip_id: clip, marks: state.marks }),
  }).then((r) => r.json());
  if (res.error) { $("sweepSaved").textContent = `NOT SAVED: ${res.error}`; return; }
  state.sweeps.set(clip, { clip_id: clip, marks: state.marks.map((m) => ({ ...m })) });
  if ((state.sweepIndex ?? 0) < state.clipOrder.length - 1) state.sweepIndex = (state.sweepIndex ?? 0) + 1;
  renderSweep();
}

/* -------------------------------------------------------------------- wiring */

function setMode(mode) {
  state.mode = mode;
  $("scorePane").classList.toggle("hidden", mode !== "score");
  $("sweepPane").classList.toggle("hidden", mode !== "sweep");
  $("tabScore").classList.toggle("active", mode === "score");
  $("tabSweep").classList.toggle("active", mode === "sweep");
  $(mode === "score" ? "sweepVideo" : "scoreVideo").pause();
  // Show the pane at its resume position. Sweep seeks to the first un-swept clip
  // (fixing the bug where it showed clip #1 after 19 were saved); scoring keeps its
  // current place (already resumed on load) and just re-renders.
  if (mode === "sweep") { seekToFirstUnswept(); renderSweep(); }
  else { renderScore(); }
}

function togglePlay(video) { video.paused ? video.play().catch(() => {}) : video.pause(); }

// Record a clip as "not a real-time reaction" and advance. Keeps the non-realtime clips
// (intro/outro, no visible stimulus) from blocking resume, and records why they were
// skipped so recall can exclude them cleanly.
async function skipNotRealtime() {
  const clip = state.clipOrder[state.sweepIndex ?? 0];
  const res = await fetch("/api/sweep", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clip_id: clip, marks: [], skip_reason: "not_realtime" }),
  }).then((r) => r.json());
  if (res.error) { $("sweepSaved").textContent = `NOT SAVED: ${res.error}`; return; }
  state.sweeps.set(clip, { clip_id: clip, marks: [], skip_reason: "not_realtime" });
  if ((state.sweepIndex ?? 0) < state.clipOrder.length - 1) state.sweepIndex += 1;
  renderSweep();
}

function wire() {
  seekToFirstUnswept();

  $("tabScore").onclick = () => setMode("score");
  $("tabSweep").onclick = () => setMode("sweep");
  $("btnSkipNotRealtime").onclick = skipNotRealtime;

  document.querySelectorAll(".score").forEach((b) => {
    b.onclick = () => submitScore(Number(b.dataset.score));
  });
  document.querySelectorAll(".cue").forEach((b) => {
    b.onclick = () => {
      state.cueType = state.cueType === b.dataset.cue ? null : b.dataset.cue;
      document.querySelectorAll(".cue").forEach((x) =>
        x.classList.toggle("chosen", x.dataset.cue === state.cueType));
    };
  });

  $("btnReplay").onclick = () => {
    const c = currentCand();
    play($("scoreVideo"), c.start_s, c.end_s);
  };
  $("btnFromStart").onclick = () => {
    state.contextReplays += 1;
    play($("scoreVideo"), 0, currentCand().end_s);
  };
  $("btnFree").onclick = () => {
    const v = $("scoreVideo");
    if (state.stopAt) { v.removeEventListener("timeupdate", state.stopAt); state.stopAt = null; }
    v.play().catch(() => {});
  };
  $("btnPrev").onclick = retreat;
  $("btnNext").onclick = advance;

  $("scoreVideo").addEventListener("timeupdate", () => {
    const c = currentCand();
    const head = $("playhead");
    if (c && head) head.style.left = `${(100 * $("scoreVideo").currentTime) / c.clip_duration_s}%`;
  });

  $("btnSpeed").dataset.rate = "2";
  $("btnSpeed").onclick = () => {
    const rates = [1, 1.5, 2, 3];
    const next = rates[(rates.indexOf(Number($("btnSpeed").dataset.rate)) + 1) % rates.length];
    $("btnSpeed").dataset.rate = String(next);
    $("btnSpeed").textContent = `Speed: ${next}×`;
    $("sweepVideo").playbackRate = next;
  };
  $("btnMark1").onclick = () => addMark(1);
  $("btnMark2").onclick = () => addMark(2);
  $("btnMark3").onclick = () => addMark(3);
  $("btnExtend").onclick = extendLastMark;
  $("btnSweepSave").onclick = saveSweep;
  $("btnSweepPrev").onclick = () => {
    if ((state.sweepIndex ?? 0) > 0) state.sweepIndex -= 1;
    renderSweep();
  };

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    const scoring = state.mode === "score";
    if (e.key === " ") {
      e.preventDefault();
      togglePlay($(scoring ? "scoreVideo" : "sweepVideo"));
      return;
    }
    if (scoring) {
      if (["0", "1", "2", "3"].includes(e.key)) { e.preventDefault(); submitScore(Number(e.key)); }
      if (e.key.toLowerCase() === "r") $("btnReplay").click();
      if (e.key.toLowerCase() === "f") $("btnFromStart").click();
      if (e.key.toLowerCase() === "g") $("btnFree").click();
      if (e.key.toLowerCase() === "v") document.querySelector('.cue[data-cue="visual"]').click();
      if (e.key.toLowerCase() === "a") document.querySelector('.cue[data-cue="audio"]').click();
      if (e.key.toLowerCase() === "b") document.querySelector('.cue[data-cue="both"]').click();
      if (e.key === "ArrowRight") { e.preventDefault(); advance(); }
      if (e.key === "ArrowLeft") { e.preventDefault(); retreat(); }
    } else {
      if (["1", "2", "3"].includes(e.key)) { e.preventDefault(); addMark(Number(e.key)); }
      if (e.key === "Enter") { e.preventDefault(); saveSweep(); }
      if (e.key.toLowerCase() === "n") { e.preventDefault(); skipNotRealtime(); }
      if (e.key.toLowerCase() === "e") { e.preventDefault(); extendLastMark(); }
    }
  });
}

const now = () => Math.round($("sweepVideo").currentTime * 1000) / 1000;

function addMark(score) {
  state.marks.push({ t_s: now(), score });
  renderMarks();
}

// Turn the most recent point mark into a span ending at the current playhead. Use it for
// a long reaction: mark its start with 1/2/3, let it play, press E at its end. A variant
// that caught any part of the span then counts as covering it -- the fix for a point at
// one edge being spuriously "missed" when the variant caught the other edge.
function extendLastMark() {
  if (!state.marks.length) return;
  const m = state.marks[state.marks.length - 1];
  const t = now();
  if (t > m.t_s) { m.end_s = t; renderMarks(); }
}

boot();
