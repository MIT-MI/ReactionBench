/* ReactionBench gold annotation — crowd frontend.
 *
 * Mode-locked: manifest.mode is "A" (score) or "B" (rank). Stage A and Stage B are deployed
 * as separate sites; Stage B is an optional within-video ranking stage that was not used for
 * the released benchmark. One annotator per id does the WHOLE set: entering an id resumes their
 * progress, shows how much of the total they've done, and the completion screen appears only
 * when they've finished everything. Vanilla JS, no build. Reads candidates from Supabase,
 * clips from R2 public URLs, writes gold_scores / gold_rankings.
 */

const CFG = {};
const S = {
  annotator: null, mode: "A",
  total: 0, done: 0, allCands: null, videoOrder: null,
  videos: [], vIdx: 0, cIdx: 0,
  cue: null, stopAt: null, scored: new Set(),
  bVideos: [], bIdx: 0, ranking: [], noStrongest: false, rankedVideos: new Set(),
};

const $ = (id) => document.getElementById(id);
const enc = encodeURIComponent;
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/* ------------------------------------------------------------------ Supabase REST */
function sbHeaders(extra) {
  return Object.assign({ apikey: CFG.supabaseAnonKey, Authorization: `Bearer ${CFG.supabaseAnonKey}`,
    "Content-Type": "application/json" }, extra || {});
}
async function sbGet(path) {
  const r = await fetch(`${CFG.supabaseUrl}/rest/v1/${path}`, { headers: sbHeaders() });
  if (!r.ok) throw new Error(`GET ${path} -> ${r.status}`);
  return r.json();
}
// Page through PostgREST's 1000-row cap.
async function sbAll(path) {
  const out = []; const step = 1000;
  for (let from = 0; ; from += step) {
    const r = await fetch(`${CFG.supabaseUrl}/rest/v1/${path}`, {
      headers: sbHeaders({ Range: `${from}-${from + step - 1}` }) });
    if (!r.ok) throw new Error(`GET ${path} -> ${r.status}`);
    const batch = await r.json(); out.push(...batch);
    if (batch.length < step) return out;
  }
}
async function sbUpsert(table, onConflict, rows) {
  const r = await fetch(`${CFG.supabaseUrl}/rest/v1/${table}?on_conflict=${onConflict}`, {
    method: "POST", headers: sbHeaders({ Prefer: "resolution=merge-duplicates,return=minimal" }),
    body: JSON.stringify(rows) });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
}
const faceUrl = (vid) => CFG.r2PublicBaseUrl + CFG.faceKeyTemplate.replace("{video_id}", vid);

/* ------------------------------------------------------------------ boot + gate */
async function boot() {
  wireKeys();
  $("gateForm").onsubmit = (e) => { e.preventDefault(); const v = $("workerId").value.trim(); if (v) startSession(v); };
  try {
    const r = await fetch("./manifest.json");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    Object.assign(CFG, await r.json());
  } catch (err) {
    console.error("manifest.json failed — serve over HTTP, not file://", err);
    $("subtitle").textContent = "Config didn’t load — open this page via a web server (http://…), not by double-clicking the file.";
    return;
  }
  S.mode = CFG.mode === "B" ? "B" : "A";
  $("paneA").classList.toggle("hidden", S.mode !== "A");
  $("paneB").classList.toggle("hidden", S.mode !== "B");
  const url = new URL(location.href);
  const pid = url.searchParams.get("PROLIFIC_PID") || url.searchParams.get("pid") || url.searchParams.get("worker");
  if (pid) startSession(pid);
}
async function startSession(id) {
  if (!CFG.supabaseUrl) { $("subtitle").textContent = "Config not loaded yet — serve over HTTP and reload."; return; }
  S.annotator = id;
  $("gate").classList.add("hidden");
  $("main").classList.remove("hidden");
  $("subtitle").textContent = `worker ${id} · ${S.mode === "A" ? "scoring" : "ranking"}`;
  if (S.mode === "A") loadStageA(); else loadStageB();
}

function updateProgress() {
  const bar = $(S.mode === "A" ? "barA" : "barB");
  const prog = $(S.mode === "A" ? "progA" : "progB");
  const pctv = S.total ? Math.round((100 * S.done) / S.total) : 0;
  bar.style.width = `${pctv}%`;
  const noun = S.mode === "A" ? "moments scored" : "clips ranked";
  prog.dataset.overall = `You’ve done ${S.done} / ${S.total} ${noun} (${pctv}%).`;
  prog.textContent = prog.dataset.overall;
}

/* ------------------------------------------------------------------ playback */
function play(video, from, until) {
  if (S.stopAt) video.removeEventListener("timeupdate", S.stopAt);
  const guard = () => { if (until != null && video.currentTime >= until) {
    video.pause(); video.removeEventListener("timeupdate", guard); S.stopAt = null; } };
  S.stopAt = until == null ? null : guard;
  const begin = () => { video.currentTime = Math.max(0, from);
    if (until != null) video.addEventListener("timeupdate", guard); video.play().catch(() => {}); };
  if (video.readyState >= 1) begin();
  else video.addEventListener("loadedmetadata", begin, { once: true });
}

/* ------------------------------------------------------------------ Stage A */
async function loadStageA() {
  // Fetch the whole candidate set and this worker's prior scores ONCE, then work from the
  // in-memory copies for every later batch.
  if (!S.allCands) {
    $("progA").textContent = "loading your progress…";
    const [allCands, mine] = await Promise.all([
      sbAll("gold_candidates?select=candidate_id,video_id&order=video_id.asc,prescreen_rank.asc"),
      sbAll(`gold_scores?annotator_id=eq.${enc(S.annotator)}&select=candidate_id`),
    ]);
    S.allCands = allCands;
    const valid = new Set(allCands.map((a) => a.candidate_id));
    S.scored = new Set(mine.map((r) => r.candidate_id).filter((c) => valid.has(c)));
    S.total = allCands.length;
    S.done = S.scored.size;
    // group into videos once, preserving order
    S.videoOrder = [];
    const idx = new Map();
    for (const c of allCands) {
      if (!idx.has(c.video_id)) { idx.set(c.video_id, S.videoOrder.length); S.videoOrder.push({ video_id: c.video_id, cids: [] }); }
      S.videoOrder[idx.get(c.video_id)].cids.push(c.candidate_id);
    }
  }
  updateProgress();
  if (S.done >= S.total) return doneScreen("A");

  const batch = S.videoOrder.filter((v) => v.cids.some((cid) => !S.scored.has(cid))).slice(0, CFG.batchVideos || 12);
  if (!batch.length) return doneScreen("A");

  const vids = batch.map((v) => v.video_id);
  const inList = `(${vids.map(enc).join(",")})`;
  const [cands, meta] = await Promise.all([
    sbGet(`gold_candidates?video_id=in.${inList}&select=candidate_id,video_id,start_s,end_s&order=video_id,prescreen_rank`),
    sbGet(`gold_videos?video_id=in.${inList}&select=video_id,duration_s`),
  ]);
  const dur = new Map(meta.map((v) => [v.video_id, v.duration_s]));
  const grouped = new Map();
  for (const c of cands) { if (!grouped.has(c.video_id)) grouped.set(c.video_id, []); grouped.get(c.video_id).push(c); }
  S.videos = vids.filter((v) => grouped.has(v)).map((vid) => ({
    video_id: vid, duration_s: dur.get(vid) || 60, faceUrl: faceUrl(vid),
    candidates: grouped.get(vid).sort((a, b) => a.start_s - b.start_s) }));
  S.vIdx = 0; S.cIdx = 0;
  loadClipA(true);
}

const curVideo = () => S.videos[S.vIdx];
const curCand = () => curVideo() && curVideo().candidates[S.cIdx];

function doneScreen(stage) {
  showCompletion(true);
}

function renderTimeline() {
  const v = curVideo(); const strip = $("timeline"); strip.innerHTML = "";
  v.candidates.forEach((c, i) => {
    const el = document.createElement("div");
    el.className = "seg" + (i === S.cIdx ? " current" : "") + (S.scored.has(c.candidate_id) ? " scored" : "");
    el.style.left = `${(100 * c.start_s) / v.duration_s}%`;
    el.style.width = `${Math.max(1.2, (100 * (c.end_s - c.start_s)) / v.duration_s)}%`;
    el.title = `${fmt(c.start_s)}–${fmt(c.end_s)}`;
    el.onclick = () => selectCand(i, "exact");
    strip.appendChild(el);
  });
  const head = document.createElement("div"); head.className = "playhead"; head.id = "playhead"; strip.appendChild(head);
}

function loadClipA(fromStart) {
  const v = curVideo();
  if (!v) return doneScreen("A");
  const video = $("videoA");
  if (video.src !== v.faceUrl) video.src = v.faceUrl;
  const firstUnscored = v.candidates.findIndex((c) => !S.scored.has(c.candidate_id));
  S.cIdx = firstUnscored >= 0 ? firstUnscored : 0;
  renderTimeline();
  selectCand(S.cIdx, fromStart ? "fromStart" : "exact");
}

function selectCand(i, mode) {
  const v = curVideo(); S.cIdx = i; const c = v.candidates[i];
  updateProgress();
  $("progA").textContent = `${$("progA").dataset.overall || ""}  ·  clip ${S.vIdx + 1}/${S.videos.length} in this batch · moment ${i + 1}/${v.candidates.length}`;
  $("metaA").textContent = `moment ${fmt(c.start_s)}–${fmt(c.end_s)} of a ${v.duration_s.toFixed(0)}s clip` +
    (S.scored.has(c.candidate_id) ? " · already scored" : "");
  S.cue = null;
  document.querySelectorAll(".score").forEach((b) => b.classList.remove("chosen"));
  document.querySelectorAll(".cue").forEach((b) => b.classList.remove("chosen"));
  $("savedA").textContent = "";
  renderTimeline();
  const video = $("videoA");
  if (mode === "fromStart") play(video, 0, c.end_s);
  else if (mode === "context") play(video, Math.max(0, c.start_s - (CFG.minWindowContextS || 4)), c.end_s);
  else play(video, c.start_s, c.end_s);
}

async function submitScore(score) {
  const c = curCand(); if (!c) return;
  const isNew = !S.scored.has(c.candidate_id);
  try {
    await sbUpsert("gold_scores", "candidate_id,annotator_id",
      [{ candidate_id: c.candidate_id, annotator_id: S.annotator, score, cue_type: S.cue }]);
  } catch (e) { $("savedA").textContent = `NOT SAVED: ${e.message}`; return; }
  if (isNew) { S.scored.add(c.candidate_id); S.done += 1; }
  updateProgress();
  document.querySelectorAll(".score").forEach((b) => b.classList.toggle("chosen", Number(b.dataset.score) === score));
  const v = curVideo();
  const next = v.candidates.findIndex((x, i) => i > S.cIdx && !S.scored.has(x.candidate_id));
  if (next >= 0) selectCand(next, "context");
  else nextClip();
}

function nextClip() {
  if (S.vIdx < S.videos.length - 1) { S.vIdx += 1; loadClipA(true); }
  else loadStageA();
}
function prevClip() { if (S.vIdx > 0) { S.vIdx -= 1; loadClipA(true); } }

/* ------------------------------------------------------------------ completion */
function showCompletion(complete) {
  const ov = $("doneOverlay"); if (!ov) return;
  if (!complete) { ov.classList.add("hidden"); return; }
  ov.classList.remove("hidden");
  $("doneTitle").textContent = "All done — thank you!";
  $("doneMsg").textContent = `You have finished all ${S.total} ${S.mode === "A" ? "moments" : "clips"}. Your work is saved.`;
  const link = $("doneLink");
  if (CFG.completionUrl) { link.href = CFG.completionUrl; link.classList.remove("hidden"); }
  else link.classList.add("hidden");
  const code = $("doneCode");
  if (CFG.completionCode) { code.textContent = `Completion code: ${CFG.completionCode}`; code.classList.remove("hidden"); }
  else code.classList.add("hidden");
  $("doneMore").classList.add("hidden");   // there's nothing left to keep going on
}

/* ------------------------------------------------------------------ Stage B (optional ranking stage) */
async function loadStageB() {
  $("progB").textContent = "loading your progress…";
  const [ready, mine] = await Promise.all([
    sbAll("gold_video_status?passes_stage_a=eq.true&n_candidates=gte.2&select=video_id"),
    sbAll(`gold_rankings?annotator_id=eq.${enc(S.annotator)}&select=video_id`),
  ]);
  S.rankedVideos = new Set(mine.map((r) => r.video_id));
  S.total = ready.length;
  S.done = ready.filter((r) => S.rankedVideos.has(r.video_id)).length;
  updateProgress();
  const todo = ready.filter((r) => !S.rankedVideos.has(r.video_id)).map((r) => r.video_id);
  if (!todo.length) return doneScreen("B");
  const vids = todo.slice(0, CFG.batchVideos || 12);
  const inList = `(${vids.map(enc).join(",")})`;
  const [cands, meta] = await Promise.all([
    sbGet(`gold_candidates?video_id=in.${inList}&select=candidate_id,video_id,start_s,end_s&order=video_id,start_s`),
    sbGet(`gold_videos?video_id=in.${inList}&select=video_id,duration_s`),
  ]);
  const dur = new Map(meta.map((v) => [v.video_id, v.duration_s]));
  const grouped = new Map();
  for (const c of cands) { if (!grouped.has(c.video_id)) grouped.set(c.video_id, []); grouped.get(c.video_id).push(c); }
  S.bVideos = vids.filter((v) => grouped.has(v)).map((vid) => ({ video_id: vid, duration_s: dur.get(vid) || 60, faceUrl: faceUrl(vid), candidates: grouped.get(vid) }));
  S.bIdx = 0; renderB();
}
function renderB() {
  const v = S.bVideos[S.bIdx]; if (!v) return doneScreen("B");
  updateProgress();
  $("progB").textContent = `${$("progB").dataset.overall || ""}  ·  clip ${S.bIdx + 1}/${S.bVideos.length} in this batch`;
  $("metaB").textContent = `${v.candidates.length} moments to rank`;
  $("savedB").textContent = "";
  const video = $("videoB"); if (video.src !== v.faceUrl) video.src = v.faceUrl; video.currentTime = 0;
  S.ranking = []; S.noStrongest = false; renderRankList();
}
function renderRankList() {
  const v = S.bVideos[S.bIdx]; const ol = $("rankList"); ol.innerHTML = "";
  v.candidates.forEach((c) => {
    const li = document.createElement("li");
    const rank = S.ranking.indexOf(c.candidate_id);
    li.className = "rankitem" + (rank >= 0 ? " picked" : "");
    li.innerHTML = `<span class="rk">${rank >= 0 ? rank + 1 : "·"}</span> <span>${fmt(c.start_s)}–${fmt(c.end_s)}</span>`;
    li.onclick = () => { const i = S.ranking.indexOf(c.candidate_id);
      if (i >= 0) S.ranking.splice(i, 1); else S.ranking.push(c.candidate_id);
      S.noStrongest = false; renderRankList(); };
    const peek = document.createElement("button");
    peek.type = "button"; peek.textContent = "▶"; peek.className = "peek";
    peek.onclick = (e) => { e.stopPropagation(); play($("videoB"), c.start_s, c.end_s); };
    li.appendChild(peek); ol.appendChild(li);
  });
}
async function saveRanking() {
  const v = S.bVideos[S.bIdx];
  if (!S.noStrongest && S.ranking.length < v.candidates.length) {
    $("savedB").textContent = "rank every moment, or choose “no clear strongest”."; return; }
  try {
    await sbUpsert("gold_rankings", "video_id,annotator_id", [{ video_id: v.video_id, annotator_id: S.annotator,
      ranking: S.noStrongest ? [] : S.ranking, no_clear_strongest: S.noStrongest }]);
  } catch (e) { $("savedB").textContent = `NOT SAVED: ${e.message}`; return; }
  if (!S.rankedVideos.has(v.video_id)) { S.rankedVideos.add(v.video_id); S.done += 1; }
  updateProgress();
  if (S.bIdx < S.bVideos.length - 1) { S.bIdx += 1; renderB(); } else loadStageB();
}

/* ------------------------------------------------------------------ keys */
function wireKeys() {
  $("btnReplay") && ($("btnReplay").onclick = () => { const c = curCand(); if (c) play($("videoA"), c.start_s, c.end_s); });
  $("btnFromStart") && ($("btnFromStart").onclick = () => selectCand(S.cIdx, "fromStart"));
  $("btnFree") && ($("btnFree").onclick = () => { const v = $("videoA"); if (S.stopAt) { v.removeEventListener("timeupdate", S.stopAt); S.stopAt = null; } v.play().catch(() => {}); });
  $("btnNextClip") && ($("btnNextClip").onclick = nextClip);
  document.querySelectorAll(".score").forEach((b) => (b.onclick = () => submitScore(Number(b.dataset.score))));
  document.querySelectorAll(".cue").forEach((b) => (b.onclick = () => {
    S.cue = S.cue === b.dataset.cue ? null : b.dataset.cue;
    document.querySelectorAll(".cue").forEach((x) => x.classList.toggle("chosen", x.dataset.cue === S.cue)); }));
  const va = $("videoA");
  va && va.addEventListener("timeupdate", () => { const c = curCand(); const h = $("playhead");
    if (c && h) h.style.left = `${(100 * va.currentTime) / curVideo().duration_s}%`; });
  $("btnNoStrongest") && ($("btnNoStrongest").onclick = () => { S.noStrongest = true; S.ranking = []; renderRankList(); $("savedB").textContent = "marked: no clear strongest"; });
  $("btnRankSave") && ($("btnRankSave").onclick = saveRanking);
  $("doneMore") && ($("doneMore").onclick = () => $("doneOverlay").classList.add("hidden"));

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    const video = $(S.mode === "A" ? "videoA" : "videoB");
    if (e.key === " ") { e.preventDefault(); video.paused ? video.play().catch(() => {}) : video.pause(); return; }
    if (S.mode === "A") {
      if (["0", "1", "2", "3"].includes(e.key)) { e.preventDefault(); submitScore(Number(e.key)); }
      else if (e.key.toLowerCase() === "r") { const c = curCand(); if (c) play($("videoA"), c.start_s, c.end_s); }
      else if (e.key.toLowerCase() === "f") selectCand(S.cIdx, "fromStart");
      else if (e.key.toLowerCase() === "g") $("btnFree").click();
      else if (e.key.toLowerCase() === "v") document.querySelector('.cue[data-cue="visual"]').click();
      else if (e.key.toLowerCase() === "a") document.querySelector('.cue[data-cue="audio"]').click();
      else if (e.key.toLowerCase() === "b") document.querySelector('.cue[data-cue="both"]').click();
      else if (e.key === "ArrowRight" || e.key.toLowerCase() === "n") { e.preventDefault(); nextClip(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); prevClip(); }
    } else if (e.key === "Enter") { e.preventDefault(); saveRanking(); }
  });
}

boot();
