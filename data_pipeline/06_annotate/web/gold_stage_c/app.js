/* ReactionBench gold annotation — Stage C (rationale correction).
 *
 * One annotator per id corrects the whole set of rationale tasks (candidates that passed
 * Stage A). Resumes progress on id entry, shows overall progress, completion only at 100%.
 * Shows the reaction clip AND the stimulus clip; a machine draft is pre-filled for editing.
 * Writes gold_rationales. Vanilla JS, no build.
 */

const CFG = {};
const S = { annotator: null, total: 0, done: 0, tasks: null, doneIds: new Set(), batch: [], idx: 0, stopAt: null, rate: 1 };
const $ = (id) => document.getElementById(id);
const enc = encodeURIComponent;
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function sbHeaders(extra) {
  return Object.assign({ apikey: CFG.supabaseAnonKey, Authorization: `Bearer ${CFG.supabaseAnonKey}`,
    "Content-Type": "application/json" }, extra || {});
}
async function sbAll(path) {
  const out = []; const step = 1000;
  for (let from = 0; ; from += step) {
    const r = await fetch(`${CFG.supabaseUrl}/rest/v1/${path}`, { headers: sbHeaders({ Range: `${from}-${from + step - 1}` }) });
    if (!r.ok) throw new Error(`${path} -> ${r.status}`);
    const b = await r.json(); out.push(...b);
    if (b.length < step) return out;
  }
}
async function sbUpsert(table, onConflict, rows) {
  const r = await fetch(`${CFG.supabaseUrl}/rest/v1/${table}?on_conflict=${onConflict}`, {
    method: "POST", headers: sbHeaders({ Prefer: "resolution=merge-duplicates,return=minimal" }), body: JSON.stringify(rows) });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
}
const mediaUrl = (key) => CFG.r2PublicBaseUrl + key;

async function boot() {
  wireKeys();
  $("gateForm").onsubmit = (e) => { e.preventDefault(); const v = $("workerId").value.trim(); if (v) startSession(v); };
  try {
    const r = await fetch("./manifest.json"); if (!r.ok) throw new Error(r.status);
    Object.assign(CFG, await r.json());
  } catch (err) {
    $("subtitle").textContent = "Config didn’t load — open via a web server (http://…), not file://";
    console.error(err); return;
  }
  const url = new URL(location.href);
  const pid = url.searchParams.get("PROLIFIC_PID") || url.searchParams.get("pid") || url.searchParams.get("worker");
  if (pid) startSession(pid);
}
async function startSession(id) {
  if (!CFG.supabaseUrl) return;
  S.annotator = id;
  $("gate").classList.add("hidden"); $("main").classList.remove("hidden");
  $("subtitle").textContent = `worker ${id} · rationale`;
  load();
}

function updateProgress() {
  const pctv = S.total ? Math.round((100 * S.done) / S.total) : 0;
  $("bar").style.width = `${pctv}%`;
  $("prog").dataset.overall = `You’ve done ${S.done} / ${S.total} rationales (${pctv}%).`;
  $("prog").textContent = $("prog").dataset.overall;
}

async function load() {
  if (!S.tasks) {
    $("prog").textContent = "loading your progress…";
    const [tasks, mine] = await Promise.all([
      sbAll("gold_rationale_tasks?select=candidate_id,video_id,start_s,end_s,face_r2_key,stimuli_r2_key,llm_draft&order=candidate_id.asc"),
      sbAll(`gold_rationales?annotator_id=eq.${enc(S.annotator)}&select=candidate_id`),
    ]);
    S.tasks = tasks;
    const valid = new Set(tasks.map((t) => t.candidate_id));
    S.doneIds = new Set(mine.map((r) => r.candidate_id).filter((c) => valid.has(c)));
    S.total = tasks.length; S.done = S.doneIds.size;
  }
  updateProgress();
  if (!S.total) { $("prog").textContent = "No rationale tasks yet — they appear after Stage A scoring is in."; return; }
  if (S.done >= S.total) return showCompletion();
  S.batch = S.tasks.filter((t) => !S.doneIds.has(t.candidate_id)).slice(0, CFG.batchTasks || 20);
  S.idx = 0;
  render();
}

function render() {
  const t = S.batch[S.idx];
  if (!t) return load();
  updateProgress();
  $("prog").textContent = `${$("prog").dataset.overall}  ·  task ${S.idx + 1}/${S.batch.length} in this batch`;
  $("meta").textContent = `reaction at ${fmt(t.start_s)}–${fmt(t.end_s)} · clip ${t.video_id}`;
  // Both clips are the SAME aligned window, so play them in sync. Stimulus leads (with audio —
  // the trigger the annotator must understand); the face follows, muted.
  const stim = $("stimVideo"), face = $("faceVideo");
  if (!stim.src.endsWith(t.stimuli_r2_key)) stim.src = mediaUrl(t.stimuli_r2_key);
  if (!face.src.endsWith(t.face_r2_key)) face.src = mediaUrl(t.face_r2_key);
  stim.controls = true; stim.muted = false;
  face.controls = false; face.muted = true;
  stim.playbackRate = face.playbackRate = S.rate;
  if (S._syncFor !== t.candidate_id) { syncFollow(stim, face); S._syncFor = t.candidate_id; }
  $("rationale").value = "";                       // blind write — no draft priming (human-only gold)
  $("saved").textContent = "";
  playMoment();
}

function syncFollow(leader, follower) {
  if (leader._sync) return;                       // wire the pair once
  leader._sync = true; follower.muted = true;
  const align = () => { if (Math.abs(leader.currentTime - follower.currentTime) > 0.15) follower.currentTime = leader.currentTime; };
  leader.addEventListener("play", () => { follower.currentTime = leader.currentTime; follower.play().catch(() => {}); });
  leader.addEventListener("pause", () => follower.pause());
  leader.addEventListener("seeking", align);
  leader.addEventListener("seeked", align);
  leader.addEventListener("timeupdate", () => { if (!leader.paused) align(); });
  leader.addEventListener("ratechange", () => { follower.playbackRate = leader.playbackRate; });
  leader.addEventListener("ended", () => follower.pause());
}

function playMoment() {
  const stim = $("stimVideo");
  const go = () => { stim.currentTime = 0; stim.play().catch(() => {}); };
  if (stim.readyState >= 1) go(); else stim.addEventListener("loadedmetadata", go, { once: true });
}

// status: "ok" (wrote a rationale) | "unfamiliar" (can't explain — outside annotator's culture,
// NOT a defect) | "unclear" (the stimulus itself is unclear or mis-timed). The two non-ok cases
// are kept distinct so cultural unfamiliarity is its own signal, not a quality judgement.
const MARK = { unfamiliar: "[cannot determine: unfamiliar with this content]",
               unclear: "[stimulus unclear or mis-timed]" };
async function save(status) {
  const t = S.batch[S.idx];
  const text = status === "ok" ? $("rationale").value.trim() : "";
  if (status === "ok" && text.length < 3) { $("saved").textContent = "write a sentence, or use a button below."; return; }
  try {
    await sbUpsert("gold_rationales", "candidate_id,annotator_id", [{
      candidate_id: t.candidate_id, annotator_id: S.annotator,
      rationale: status === "ok" ? text : MARK[status],
      status,
      edited_from_draft: status === "ok" ? text !== (t.llm_draft || "") : false,
    }]);
  } catch (e) { $("saved").textContent = `NOT SAVED: ${e.message}`; return; }
  if (!S.doneIds.has(t.candidate_id)) { S.doneIds.add(t.candidate_id); S.done += 1; }
  updateProgress();
  if (S.idx < S.batch.length - 1) { S.idx += 1; render(); } else load();
}

function showCompletion() {
  const ov = $("doneOverlay"); ov.classList.remove("hidden");
  $("doneMsg").textContent = `You have finished all ${S.total} rationales. Your work is saved.`;
  const link = $("doneLink");
  if (CFG.completionUrl) { link.href = CFG.completionUrl; link.classList.remove("hidden"); }
  const code = $("doneCode");
  if (CFG.completionCode) { code.textContent = `Completion code: ${CFG.completionCode}`; code.classList.remove("hidden"); }
}

function wireKeys() {
  $("btnSave") && ($("btnSave").onclick = () => save("ok"));
  $("btnUnfamiliar") && ($("btnUnfamiliar").onclick = () => save("unfamiliar"));
  $("btnSkip") && ($("btnSkip").onclick = () => save("unclear"));
  $("btnMoment") && ($("btnMoment").onclick = playMoment);
  document.querySelectorAll(".spd").forEach((b) => (b.onclick = () => {
    S.rate = parseFloat(b.dataset.rate);
    $("stimVideo").playbackRate = $("faceVideo").playbackRate = S.rate;
    document.querySelectorAll(".spd").forEach((x) => x.classList.toggle("on", x === b));
  }));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); save("ok"); }
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
    if (e.key.toLowerCase() === "r") playMoment();
  });
}
boot();
