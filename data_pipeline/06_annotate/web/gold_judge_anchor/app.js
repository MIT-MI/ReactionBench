/* ReactionBench judge-anchor rubric. One annotator grades every anchor task on the
 * judge's exact rubric (cue 0-2, link 0-2, faithful 0-1). Model identity is never shown.
 * Resumes on id entry; writes gold_anchor_scores. Vanilla JS, no build. */

const CFG = {};
const S = { annotator: null, total: 0, done: 0, tasks: null, doneIds: new Set(),
            batch: [], idx: 0, rate: 1, pick: { cue: null, link: null, faithful: null } };
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
  try {
    const r = await fetch("./manifest.json"); if (!r.ok) throw new Error(r.status);
    Object.assign(CFG, await r.json());
  } catch (err) {
    $("subtitle").textContent = "Config didn’t load — open via a web server (http://…), not file://";
    console.error(err); return;
  }
  $("subtitle").textContent = "grade model explanations on the judge's rubric";
  const url = new URL(location.href);
  const pid = url.searchParams.get("pid") || url.searchParams.get("PROLIFIC_PID");
  if (pid) startSession(pid);
}
function startSession(id) {
  if (!CFG.supabaseUrl) return;
  S.annotator = id;
  $("gate").classList.add("hidden"); $("main").classList.remove("hidden");
  $("subtitle").textContent = `annotator ${id} · anchor rubric`;
  load();
}

function updateProgress() {
  const pct = S.total ? Math.round((100 * S.done) / S.total) : 0;
  $("bar").style.width = `${pct}%`;
  $("prog").textContent = `You’ve graded ${S.done} / ${S.total} (${pct}%).`;
}

async function load() {
  if (!S.tasks) {
    $("prog").textContent = "loading…";
    let tasks, mine;
    try {
      [tasks, mine] = await Promise.all([
        sbAll("gold_anchor_tasks?select=anchor_id,candidate_id,rationale,refs,face_r2_key,stimuli_r2_key&order=anchor_id.asc"),
        sbAll(`gold_anchor_scores?annotator_id=eq.${enc(S.annotator)}&select=anchor_id`),
      ]);
    } catch (e) {
      $("prog").textContent = "Tables not found — deploy schema_anchor.sql in Supabase, then run build_anchor_tasks.";
      console.error(e); return;
    }
    S.tasks = tasks;
    const valid = new Set(tasks.map((t) => t.anchor_id));
    S.doneIds = new Set(mine.map((r) => r.anchor_id).filter((a) => valid.has(a)));
    S.total = tasks.length; S.done = S.doneIds.size;
  }
  updateProgress();
  if (!S.total) { $("prog").textContent = "No anchor tasks loaded yet (run build_anchor_tasks)."; return; }
  if (S.done >= S.total) return showCompletion();
  S.batch = S.tasks.filter((t) => !S.doneIds.has(t.anchor_id)).slice(0, 25);
  S.idx = 0;
  render();
}

function setPick(dim, v) {
  S.pick[dim] = v;
  const seg = { cue: "segCue", link: "segLink", faithful: "segFaith" }[dim];
  document.querySelectorAll(`#${seg} button`).forEach((b) =>
    b.classList.toggle("on", Number(b.dataset.v) === v));
}

function render() {
  const t = S.batch[S.idx];
  if (!t) return load();
  updateProgress();
  $("prog").textContent += `  ·  ${S.idx + 1}/${S.batch.length} in this batch`;
  const stim = $("stimVideo"), face = $("faceVideo");
  if (!stim.src.endsWith(t.stimuli_r2_key)) stim.src = mediaUrl(t.stimuli_r2_key);
  if (!face.src.endsWith(t.face_r2_key)) face.src = mediaUrl(t.face_r2_key);
  stim.controls = true; stim.muted = false;
  face.controls = false; face.muted = true;
  stim.playbackRate = face.playbackRate = S.rate;
  if (S._syncFor !== t.anchor_id) { syncFollow(stim, face); S._syncFor = t.anchor_id; }
  $("refs").innerHTML = (t.refs || []).map((x) => `<li>“${esc(x)}”</li>`).join("");
  $("cand").textContent = `“${t.rationale}”`;
  ["cue", "link", "faithful"].forEach((d) => setPick(d, null));
  $("saved").textContent = "";
  playMoment();
}
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

function syncFollow(leader, follower) {
  if (leader._sync) return;
  leader._sync = true; follower.muted = true;
  const align = () => { if (Math.abs(leader.currentTime - follower.currentTime) > 0.15) follower.currentTime = leader.currentTime; };
  leader.addEventListener("play", () => { follower.currentTime = leader.currentTime; follower.play().catch(() => {}); });
  leader.addEventListener("pause", () => follower.pause());
  leader.addEventListener("seeking", align);
  leader.addEventListener("seeked", align);
  leader.addEventListener("timeupdate", () => { if (!leader.paused) align(); });
  leader.addEventListener("ended", () => follower.pause());
}
function playMoment() {
  const stim = $("stimVideo");
  const go = () => { stim.currentTime = 0; stim.play().catch(() => {}); };
  if (stim.readyState >= 1) go(); else stim.addEventListener("loadedmetadata", go, { once: true });
}

async function save() {
  const t = S.batch[S.idx];
  const { cue, link, faithful } = S.pick;
  if (cue === null || link === null || faithful === null) {
    $("saved").textContent = "pick all three dimensions first."; return;
  }
  try {
    await sbUpsert("gold_anchor_scores", "anchor_id,annotator_id",
      [{ anchor_id: t.anchor_id, annotator_id: S.annotator, cue, link, faithful }]);
  } catch (e) { $("saved").textContent = `NOT SAVED: ${e.message}`; return; }
  if (!S.doneIds.has(t.anchor_id)) { S.doneIds.add(t.anchor_id); S.done += 1; }
  if (S.idx < S.batch.length - 1) { S.idx += 1; render(); } else load();
}

function showCompletion() {
  $("doneOverlay").classList.remove("hidden");
  $("doneMsg").textContent = `All ${S.total} anchor items graded. Everything is saved.`;
  const link = $("doneLink");
  if (CFG.completionUrl) { link.href = CFG.completionUrl; link.classList.remove("hidden"); }
  const code = $("doneCode");
  if (CFG.completionCode) { code.textContent = `Completion code: ${CFG.completionCode}`; code.classList.remove("hidden"); }
}

function wire() {
  $("gateForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const id = $("workerId").value.trim();
    if (id) startSession(id);
  });
  $("btnSave").onclick = save;
  $("btnMoment").onclick = playMoment;
  document.querySelectorAll("#segCue button").forEach((b) => (b.onclick = () => setPick("cue", Number(b.dataset.v))));
  document.querySelectorAll("#segLink button").forEach((b) => (b.onclick = () => setPick("link", Number(b.dataset.v))));
  document.querySelectorAll("#segFaith button").forEach((b) => (b.onclick = () => setPick("faithful", Number(b.dataset.v))));
  document.querySelectorAll(".spd").forEach((b) => (b.onclick = () => {
    S.rate = parseFloat(b.dataset.rate);
    $("stimVideo").playbackRate = $("faceVideo").playbackRate = S.rate;
    document.querySelectorAll(".spd").forEach((x) => x.classList.toggle("on", x === b));
  }));
  const KEYS = { q: ["cue", 0], w: ["cue", 1], e: ["cue", 2],
                 a: ["link", 0], s: ["link", 1], d: ["link", 2],
                 z: ["faithful", 0], x: ["faithful", 1] };
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); save(); return; }
    const k = e.key.toLowerCase();
    if (k === "r") { playMoment(); return; }
    if (KEYS[k]) setPick(...KEYS[k]);
  });
}
boot();
