/* Rationale set video-quality review (internal curation tool). Goes through all 200 gold_rationale_tasks,
 * sync-plays stimulus+reaction, and records a verdict (keep/delete) + optional trim window +
 * note to gold_rationale_review. Not Prolific — this is a cleanup pass before the real annotation.
 * Vanilla JS. */

const CFG = {};
const S = { reviewer: null, tasks: [], verdicts: {}, idx: 0, choice: null, target: 200 };
const keptCount = () => Object.values(S.verdicts).filter((v) => v.verdict === "keep").length;
const $ = (id) => document.getElementById(id);
const enc = encodeURIComponent;

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
  wire();
  try { Object.assign(CFG, await (await fetch("./manifest.json")).json()); }
  catch (e) { $("subtitle").textContent = "Open via http:// (not file://)"; return; }
  const pid = new URL(location.href).searchParams.get("id");
  if (pid) startSession(pid);
}
function startSession(id) {
  S.reviewer = id;
  $("gate").classList.add("hidden"); $("main").classList.remove("hidden");
  $("subhdr").textContent = `reviewer ${id}`;
  load();
}

async function load() {
  $("prog").textContent = "loading…";
  const [tasks, rev] = await Promise.all([
    sbAll("gold_rationale_tasks?select=candidate_id,video_id,face_r2_key,stimuli_r2_key,llm_draft,scene_cuts,cultural_specificity,consensus_score&order=candidate_id.asc"),
    sbAll("gold_rationale_review?select=candidate_id,verdict,trim_start,trim_end,note"),
  ]);
  S.tasks = tasks;
  S.target = CFG.target || 200;
  S.verdicts = Object.fromEntries(rev.map((r) => [r.candidate_id, r]));
  if (keptCount() >= S.target) return showDone();
  const first = tasks.findIndex((t) => !(S.verdicts[t.candidate_id] && S.verdicts[t.candidate_id].verdict));
  if (first < 0) return showNeedMore();          // reviewed everything but still short of target
  S.idx = first;
  render();
}

function updateProgress() {
  const kept = keptCount();
  const pct = Math.round((100 * kept) / S.target);
  $("bar").style.width = `${Math.min(100, pct)}%`;
  const need = Math.max(0, S.target - kept);
  $("prog").textContent = `kept ${kept} / ${S.target}` + (need ? ` · need ${need} more` : " · target reached") + `  ·  #${S.idx + 1}`;
}

function showNeedMore() {
  $("doneOverlay").classList.remove("hidden");
  $("doneMsg").textContent = `Reviewed everything, kept ${keptCount()} / ${S.target}. Ask for ${S.target - keptCount()} more backfill clips, then come back.`;
}

function syncFollow(leader, follower) {
  if (leader._sync) return; leader._sync = true; follower.muted = true;
  const align = () => { if (Math.abs(leader.currentTime - follower.currentTime) > 0.15) follower.currentTime = leader.currentTime; };
  leader.addEventListener("play", () => { follower.currentTime = leader.currentTime; follower.play().catch(() => {}); });
  leader.addEventListener("pause", () => follower.pause());
  leader.addEventListener("seeked", align);
  leader.addEventListener("timeupdate", () => { if (!leader.paused) align(); });
  leader.addEventListener("ended", () => follower.pause());
}

function render() {
  const t = S.tasks[S.idx];
  if (!t) return showDone();
  updateProgress();
  const v = S.verdicts[t.candidate_id] || {};
  const cut = t.scene_cuts > 0 ? ` · ⚠ ${t.scene_cuts} scene cut(s)` : "";
  $("meta").innerHTML = `<b>${t.candidate_id}</b> · clip ${t.video_id} · cultural ${t.cultural_specificity ?? "?"} · consensus ${t.consensus_score}${cut}`;
  const stim = $("stimVideo"), face = $("faceVideo");
  stim.src = mediaUrl(t.stimuli_r2_key); face.src = mediaUrl(t.face_r2_key);
  stim.muted = false; face.muted = true;
  syncFollow(stim, face);
  $("draft").textContent = `Qwen draft (reference only): ${t.llm_draft || "—"}`;
  $("trimA").value = v.trim_start ?? ""; $("trimB").value = v.trim_end ?? "";
  $("note").value = v.note ?? "";
  S.choice = v.verdict ?? null;
  paintVerdict();
  const go = () => { stim.currentTime = 0; stim.play().catch(() => {}); };
  if (stim.readyState >= 1) go(); else stim.addEventListener("loadedmetadata", go, { once: true });
}

function paintVerdict() {
  $("btnKeep").classList.toggle("on", S.choice === "keep");
  $("btnDel").classList.toggle("on", S.choice === "delete");
  $("btnSave").disabled = !S.choice;
}
function setChoice(v) { S.choice = v; paintVerdict(); }

async function save(dir) {
  const t = S.tasks[S.idx];
  if (!S.choice) { $("saved").textContent = "pick Keep or Delete first."; return; }
  const row = {
    candidate_id: t.candidate_id, verdict: S.choice, reviewer: S.reviewer,
    trim_start: $("trimA").value === "" ? null : +$("trimA").value,
    trim_end: $("trimB").value === "" ? null : +$("trimB").value,
    note: $("note").value.trim() || null,
  };
  try { await sbUpsert("gold_rationale_review", "candidate_id", [row]); }
  catch (e) { $("saved").textContent = `NOT SAVED: ${e.message}`; return; }
  S.verdicts[t.candidate_id] = row;
  if (keptCount() >= S.target) return showDone();
  // jump to the next unreviewed clip (kept clips are already done and skipped)
  let next = S.tasks.findIndex((x, i) => i > S.idx && !(S.verdicts[x.candidate_id] && S.verdicts[x.candidate_id].verdict));
  if (next < 0) next = S.tasks.findIndex((x) => !(S.verdicts[x.candidate_id] && S.verdicts[x.candidate_id].verdict));
  if (next < 0) return showNeedMore();
  S.idx = next;
  render();
}

function showDone() {
  $("doneOverlay").classList.remove("hidden");
  $("doneMsg").textContent = `Target reached — ${keptCount()} kept. You can stop; ping to apply deletes/trims and finalize the 200.`;
}

function wire() {
  $("gateForm").onsubmit = (e) => { e.preventDefault(); const v = $("reviewerId").value.trim(); if (v) startSession(v); };
  $("btnKeep").onclick = () => setChoice("keep");
  $("btnDel").onclick = () => setChoice("delete");
  $("btnSave").onclick = () => save(1);
  $("btnPrev").onclick = () => { S.idx = Math.max(0, S.idx - 1); render(); };
  $("btnReplay").onclick = () => { const s = $("stimVideo"); s.currentTime = 0; s.play().catch(() => {}); };
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") { if (e.key === "Enter" && e.target.id === "note" && S.choice) save(1); return; }
    if (e.key.toLowerCase() === "k") setChoice("keep");
    if (e.key.toLowerCase() === "d") setChoice("delete");
    if (e.key === "Enter" && S.choice) save(1);
  });
}
boot();
