/* Stimulus<->reaction alignment review. Local-only QC: renders every item in data.js as a
 * card (stimulus event | face reaction | Gemini's event text), lets you mark good/bad, persists
 * verdicts to localStorage, and exports them for copy. No backend, no Prolific -- this is an
 * internal quality pass before the sets go out. Vanilla JS, opens from file://. */

const KEY = "stim_review_verdicts_v1";
const verdicts = JSON.parse(localStorage.getItem(KEY) || "{}");
const $ = (id) => document.getElementById(id);
const DOM = { Patreon: "Patreon", Comedy_internet_culture: "Comedy",
  "General_pop-culture_React": "PopCult", game: "Game" };

function save() { localStorage.setItem(KEY, JSON.stringify(verdicts)); }

function counts() {
  const v = Object.values(verdicts);
  const good = v.filter((x) => x === "good").length;
  const bad = v.filter((x) => x === "bad").length;
  $("counts").textContent = `${good} good · ${bad} bad · ${ITEMS.length - good - bad} left`;
}

function setVerdict(cid, val, card) {
  verdicts[cid] = verdicts[cid] === val ? undefined : val;
  if (verdicts[cid] === undefined) delete verdicts[cid];
  save();
  card.classList.toggle("good", verdicts[cid] === "good");
  card.classList.toggle("bad", verdicts[cid] === "bad");
  card.querySelector(".btn-good").classList.toggle("on", verdicts[cid] === "good");
  card.querySelector(".btn-bad").classList.toggle("on", verdicts[cid] === "bad");
  counts();
}

// Single-direction sync: FACE is the leader (controls + audio), STIMULUS follows it (muted).
// Bidirectional mirroring loops, because the follower's play/pause events fire asynchronously
// and re-trigger the leader -- that was the rapid play/pause / busy-tone bug. One direction only,
// and we never listen to the follower, so there is nothing to echo.
function syncFollow(leader, follower) {
  follower.muted = true;
  const align = () => { if (Math.abs(leader.currentTime - follower.currentTime) > 0.15) follower.currentTime = leader.currentTime; };
  leader.addEventListener("play", () => { follower.currentTime = leader.currentTime; follower.play().catch(() => {}); });
  leader.addEventListener("pause", () => follower.pause());
  leader.addEventListener("seeking", align);
  leader.addEventListener("seeked", align);
  leader.addEventListener("timeupdate", () => { if (!leader.paused) align(); });
  leader.addEventListener("ended", () => follower.pause());
}

function playBoth(leader) { leader.currentTime = 0; leader.play().catch(() => {}); }

function card(it, i) {
  const el = document.createElement("div");
  el.className = "card";
  if (verdicts[it.cid]) el.classList.add(verdicts[it.cid]);
  const mism = it.dur_diff && it.dur_diff > 0.5;
  el.innerHTML = `
    <div class="videos">
      <div class="pane"><h3>Stimulus (what they watched) · follows, muted</h3>
        <video class="v-stim" src="${it.stimUrl}" muted preload="metadata" playsinline></video></div>
      <div class="pane"><h3>Reaction (the reviewer) · press play here</h3>
        <video class="v-face" src="${it.faceUrl}" controls preload="metadata" playsinline></video></div>
    </div>
    <div class="meta">
      <div class="event">${it.event}</div>
      <div class="tags">
        <span class="tag">${DOM[it.domain] || it.domain}</span>
        <span class="tag">reaction ${it.reaction_mean.toFixed(2)}</span>
        <span class="tag">stim Q${it.stim_quality}</span>
        ${it.self_contained ? '<span class="tag">self-contained</span>' : ""}
        ${it.match_ready ? '<span class="tag">match-ready</span>' : ""}
        ${it.stim_source === "recut" ? '<span class="tag recut">recut ⟳</span>' : ""}
        ${mism ? `<span class="tag warn">Δdur ${it.dur_diff}s</span>` : ""}
        <span class="tag">${it.cid}</span>
      </div>
      <div class="actions">
        <button class="btn-play">▶ play both</button>
        <button class="btn-stimaudio">🔊 stim audio</button>
        <button class="btn-good ${verdicts[it.cid] === "good" ? "on" : ""}">✓ stimulus explains it</button>
        <button class="btn-bad ${verdicts[it.cid] === "bad" ? "on" : ""}">✗ wrong / unclear</button>
        <span class="idx">${i + 1} / ${ITEMS.length}</span>
      </div>
    </div>`;
  const stim = el.querySelector(".v-stim"), face = el.querySelector(".v-face");
  syncFollow(face, stim);
  el.querySelector(".btn-play").onclick = () => playBoth(face);
  const sa = el.querySelector(".btn-stimaudio");
  sa.onclick = () => { stim.muted = !stim.muted; sa.classList.toggle("on", !stim.muted);
    sa.textContent = stim.muted ? "🔊 stim audio" : "🔇 stim muted"; };
  el.querySelector(".btn-good").onclick = () => setVerdict(it.cid, "good", el);
  el.querySelector(".btn-bad").onclick = () => setVerdict(it.cid, "bad", el);
  return el;
}

function exportVerdicts() {
  const lines = ITEMS.map((it) => `${it.cid}\t${verdicts[it.cid] || "unrated"}\t${it.domain}\tQ${it.stim_quality}\t${it.event}`);
  const bad = ITEMS.filter((it) => verdicts[it.cid] === "bad").map((it) => it.cid);
  const dump = `# ${bad.length} marked bad: ${bad.join(", ")}\n\n` + lines.join("\n");
  const ta = $("dump"); ta.hidden = false; ta.value = dump; ta.select();
  navigator.clipboard?.writeText(dump).then(
    () => ($("export").textContent = "Copied ✓"),
    () => ($("export").textContent = "Select+copy ↓"));
  setTimeout(() => ($("export").textContent = "Copy verdicts"), 1500);
}

function boot() {
  if (typeof ITEMS === "undefined" || !ITEMS.length) {
    $("sub").textContent = "data.js not loaded — run build_review_page.py first."; return;
  }
  const list = $("list");
  ITEMS.forEach((it, i) => list.appendChild(card(it, i)));
  $("export").onclick = exportVerdicts;
  counts();
}
boot();
