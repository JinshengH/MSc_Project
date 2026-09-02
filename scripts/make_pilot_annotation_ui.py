"""Generate the two-stage annotation page for the NewsCLIPpings pilot.

Stage 1 (blind): relation_label for all samples — image + claim only.
Stage 2 (falsified only): evidence_channel via the addendum-2 decision tree,
comparing the mismatched claim against the image's original caption, which
the stage deliberately reveals (annotation = caption comparison, not intent
inference).  Pristine samples get evidence_channel "n/a" automatically at
export.

Progress auto-saves to browser localStorage; "Export CSV" matches the
pilot_annotations.csv schema; "Import CSV" loads a (partially) filled CSV
for review — the hand-off point when an agent pre-labels.

Run with: conda run -n hf_latest python scripts/make_pilot_annotation_ui.py
Rerunning is safe: annotations live in localStorage / exported CSVs, not here.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# packet dir name as optional CLI arg; defaults to the pilot packet
PACKET = sys.argv[1] if len(sys.argv) > 1 else "pilot_seed42"
PILOT_DIR = PROJECT_ROOT / "data" / "newsclippings" / PACKET
OUTPUT = PILOT_DIR / "annotate.html"

RELATIONS = ["entailment", "neutral", "contradiction"]
CHANNELS = ["visible_conflict", "temporal_provenance", "event_provenance",
            "external_knowledge", "other"]
CONFIDENCES = ["high", "medium", "low"]

CSV_FIELDS = ["display_order", "sample_id", "caption_id", "image_id",
              "official_split", "caption", "relation_label",
              "evidence_channel", "annotator_confidence", "notes"]


def build_payload() -> list[dict]:
    samples = json.loads((PILOT_DIR / "pilot_samples.json").read_text())
    # Copy full-size images into the packet: file:// pages cannot load images
    # outside the html's directory tree (Safari sandbox; thumbnails work only
    # because they are in a subfolder), so the packet must be self-contained.
    full_dir = PILOT_DIR / "full"
    full_dir.mkdir(exist_ok=True)
    payload = []
    for s in sorted(samples, key=lambda x: x["display_order"]):
        src = PROJECT_ROOT / s["image_path"]
        full_image = f"full/{s['sample_id']:03d}.jpg"
        dst = PILOT_DIR / full_image
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, dst)
        payload.append({
            # similarity_score stays out; original_label / original_caption /
            # official_split are embedded but only rendered in stage 2.
            "display_order": s["display_order"],
            "sample_id": s["sample_id"],
            "caption_id": s["caption_id"],
            "image_id": s["image_id"],
            "official_split": s["official_split"],
            "original_label": s["original_label"],
            "source": s.get("source", ""),
            "caption": s["caption"],
            "original_caption": s.get("original_caption", ""),
            "thumb": f"thumbnails/{s['sample_id']:03d}.jpg",
            "full": full_image,
        })
    return payload


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NewsCLIPpings pilot annotation</title>
<style>
  :root { --line:#d9dee7; --muted:#606a78; --accent:#2563eb; --bg:#f5f6f8; }
  * { box-sizing: border-box; }
  body { background:var(--bg); color:#1c2026; font:15px/1.45 system-ui,sans-serif; margin:0; }
  header { position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid var(--line); padding:10px 20px; }
  .bar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; max-width:1100px; margin:0 auto; }
  .bar h1 { font-size:1.05rem; margin:0 12px 0 0; }
  .bar .spacer { flex:1; }
  button, .filterbtn, .stagebtn { background:#fff; border:1px solid var(--line); border-radius:6px; cursor:pointer; font:inherit; padding:5px 12px; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  .filterbtn.active, .stagebtn.active { background:#e8eefc; border-color:var(--accent); color:var(--accent); }
  #progress { color:var(--muted); font-variant-numeric:tabular-nums; }
  details.guide { max-width:1100px; margin:14px auto 0; background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px 16px; }
  details.guide summary { cursor:pointer; font-weight:600; }
  details.guide table { border-collapse:collapse; margin:10px 0; width:100%; }
  details.guide td, details.guide th { border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top; }
  details.guide code { background:#eef1f5; border-radius:4px; padding:0 4px; }
  main { max-width:1100px; margin:14px auto 60px; padding:0 16px; display:flex; flex-direction:column; gap:14px; }
  .card { background:#fff; border:1px solid var(--line); border-left:4px solid var(--line); border-radius:8px; display:flex; gap:16px; padding:14px; }
  .card.done { border-left-color:#16a34a; }
  .card .imgcol { flex:0 0 340px; }
  .card img { background:#eceef2; border-radius:6px; display:block; max-height:280px; object-fit:contain; width:100%; cursor:zoom-in; }
  .imgmeta { color:var(--muted); font-size:.8rem; margin-top:6px; }
  .formcol { flex:1; min-width:0; }
  .claim { font-size:1.02rem; margin:2px 0 10px; }
  .claim b, .orig b { color:var(--muted); font-size:.8rem; font-weight:600; letter-spacing:.03em; text-transform:uppercase; display:block; }
  .orig { background:#f2f6ee; border:1px solid #d4e2c8; border-radius:6px; font-size:.98rem; margin:0 0 10px; padding:8px 10px; }
  .chip { background:#eef1f5; border-radius:12px; color:var(--muted); display:inline-block; font-size:.8rem; margin-right:6px; padding:1px 9px; }
  fieldset { border:0; margin:0 0 8px; padding:0; }
  fieldset legend { color:var(--muted); font-size:.8rem; font-weight:600; letter-spacing:.03em; text-transform:uppercase; margin-bottom:4px; }
  .opts { display:flex; flex-wrap:wrap; gap:6px; }
  .opts label { border:1px solid var(--line); border-radius:16px; cursor:pointer; padding:3px 11px; user-select:none; }
  .opts input { display:none; }
  .opts label:has(input:checked) { background:#e8eefc; border-color:var(--accent); color:var(--accent); font-weight:600; }
  textarea { border:1px solid var(--line); border-radius:6px; font:inherit; min-height:34px; padding:6px 9px; resize:vertical; width:100%; }
  .num { color:var(--muted); font-size:.85rem; font-variant-numeric:tabular-nums; }
  #lightbox { position:fixed; inset:0; background:rgba(10,12,16,.88); display:none; align-items:center; justify-content:center; z-index:50; cursor:zoom-out; }
  /* fill the viewport, upscaling small originals (most VisualNews images are
     narrower than the 480px thumbnails); contain keeps the aspect ratio */
  #lightbox img { width:96vw; height:96vh; object-fit:contain; }
  @media (max-width:760px){ .card { flex-direction:column; } .card .imgcol { flex:none; } }
</style>
</head>
<body>
<header>
  <div class="bar">
    <h1>Pilot annotation</h1>
    <span class="stagebtn active" data-stage="1">Stage 1 &middot; Relation (blind)</span>
    <span class="stagebtn" data-stage="2">Stage 2 &middot; Evidence channel</span>
    <span id="progress"></span>
    <span class="spacer"></span>
    <span class="filterbtn active" data-f="all">All</span>
    <span class="filterbtn" data-f="todo">Unlabelled</span>
    <span class="filterbtn" data-f="done">Labelled</span>
    <button id="importBtn">Import CSV</button>
    <input type="file" id="importFile" accept=".csv" hidden>
    <button class="primary" id="exportBtn">Export CSV</button>
  </div>
</header>

<details class="guide" id="guide1">
  <summary>Stage 1 rules — relation (blind; read before starting)</summary>
  <p>Judge only what the image itself shows about the claim — never use
  outside knowledge or beliefs about whether the pair is genuine. You will
  not see whether a sample is pristine or falsified — that is deliberate.
  Full calibrated rules: <code>ANNOTATION_GUIDELINES.md</code> in the repo.</p>
  <p><b>Decision order: conflict scan &rarr; coverage check &rarr; neutral default.</b></p>
  <table>
    <tr><th>1. contradiction</th><td>Any <b>visible</b> element conflicts with the claim — action, scene/venue,
      object identity, time-of-day, era, readable text — however "secondary".
      <b>Asymmetric rule:</b> absence of a detail never penalises; a visible conflict always counts.</td></tr>
    <tr><th>2. entailment</th><td>No conflict, and <b>two checks both pass</b>:
      <b>(1) coverage</b> — the claim's depictable content (who, doing what, what scene) is fully
      supported; people in frame default to the caption's identities; inherently undepictable
      metadata (photographer, award, kinship, quotes, statistics) never counts.
      <b>(2) exclusivity</b> — the pairing is specific: this caption could NOT equally sit on
      hundreds of similar photos (substitution test). A readable event-level cue (event
      branding/scoreboard, both team kits, a uniquely identifiable place) makes a pairing
      exclusive. Coverage without exclusivity &rarr; neutral.</td></tr>
    <tr><th>3. neutral</th><td>Everything else — deliberately rule-free residual: whatever fails either gate.
      Includes the big class "generic scene + named event/place/date/opponent" (substitution
      test passes) and the unidentifiable/unsure default.</td></tr>
  </table>
  <p><b>Entity identity default rule:</b> assume named people/places are who
  the caption says they are, unless the image visibly conflicts. Set
  confidence (high/medium/low) and note edge cases. Finish stage 1 for all
  samples before opening stage 2.</p>
</details>

<details class="guide" id="guide2" style="display:none">
  <summary>Stage 2 rules — evidence channel (falsified only; decision tree)</summary>
  <p>This stage shows <b>falsified samples only</b> and reveals each image's
  <b>original caption</b>. Label the evidence channel needed to detect the
  mismatch by comparing the claim against the original caption — no intent
  guessing.</p>
  <table>
    <tr><th>Q1</th><td>Does visible image content directly conflict with the claim? Yes &rarr; <code>visible_conflict</code>. No &rarr; Q2.</td></tr>
    <tr><th>Q2</th><td>Where is the mismatch between claim and original caption?<br>
      Time differs (date/year/"today" misplaced) &rarr; <code>temporal_provenance</code><br>
      Same kind of entity, different event/place &rarr; <code>event_provenance</code><br>
      Entity/fact differs, needs external knowledge &rarr; <code>external_knowledge</code><br>
      Nothing fits &rarr; <code>other</code></td></tr>
  </table>
  <p><b>Priority when channels co-occur:</b> label the first channel that
  exposes the mismatch — visible &gt; temporal &gt; event &gt; external.
  Pristine samples are excluded here; their channel is recorded as
  <code>n/a</code> automatically on export.</p>
</details>

<main id="cards"></main>
<div id="lightbox"><img alt=""></div>

<script>
const SAMPLES = __DATA__;
const RELATIONS = __RELATIONS__;
const CHANNELS = __CHANNELS__;
const CONFIDENCES = __CONFIDENCES__;
const CSV_FIELDS = __CSV_FIELDS__;
const STORE_KEY = "__STORE_KEY__";

let ann = {};
try { ann = JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch (e) {}
let stage = 1;
let stage2Warned = false;

function get(id) { return ann[id] || {}; }
function set(id, field, value) {
  ann[id] = ann[id] || {};
  ann[id][field] = value;
  localStorage.setItem(STORE_KEY, JSON.stringify(ann));
  refreshCard(id);
  refreshProgress();
}
function stageSamples() {
  return stage === 1 ? SAMPLES
       : SAMPLES.filter(s => s.original_label === "falsified");
}
function isDone(id) {
  const a = get(id);
  const s = SAMPLES.find(x => x.sample_id === id);
  if (stage === 1) return !!(a.relation_label && a.annotator_confidence);
  return s.original_label !== "falsified" || !!a.evidence_channel;
}

function optGroup(id, field, options) {
  const current = get(id)[field] || "";
  const buttons = options.map(o =>
    `<label><input type="radio" name="${field}_${id}" value="${o}"` +
    `${o === current ? " checked" : ""}> ${o}</label>`).join("");
  return `<div class="opts" data-id="${id}" data-field="${field}">${buttons}</div>`;
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                  .replace(/"/g, "&quot;");
}

function cardHTML(s) {
  const a = get(s.sample_id);
  const stage2 = stage === 2;
  const meta = stage2
    ? `<span class="chip">${esc(s.official_split)}</span>` +
      `<span class="chip">relation: ${esc(a.relation_label || "?")}</span>`
    : "";
  const orig = stage2
    ? `<p class="orig"><b>Original caption of this image</b>${esc(s.original_caption)}</p>`
    : "";
  const controls = stage2
    ? `<fieldset><legend>evidence_channel (Q1&rarr;Q2, priority visible&gt;temporal&gt;event&gt;external)</legend>
         ${optGroup(s.sample_id, "evidence_channel", CHANNELS)}</fieldset>`
    : `<fieldset><legend>relation_label</legend>${optGroup(s.sample_id, "relation_label", RELATIONS)}</fieldset>
       <fieldset><legend>annotator_confidence</legend>${optGroup(s.sample_id, "annotator_confidence", CONFIDENCES)}</fieldset>`;
  return `
  <div class="card${isDone(s.sample_id) ? " done" : ""}" id="card_${s.sample_id}">
    <div class="imgcol">
      <img src="${s.thumb}" data-full="${esc(s.full)}" alt="sample ${s.sample_id}" loading="lazy">
      <div class="imgmeta">source: ${esc(s.source)} &middot; caption_id ${s.caption_id} &middot; image_id ${s.image_id}
        &middot; <a href="${esc(s.full)}" target="_blank">full image</a></div>
    </div>
    <div class="formcol">
      <div class="num">#${String(s.display_order).padStart(3, "0")} (sample ${s.sample_id}) ${meta}</div>
      <p class="claim"><b>Claim (caption under test)</b>${esc(s.caption)}</p>
      ${orig}
      ${controls}
      <textarea placeholder="notes (edge cases, reasoning)" data-id="${s.sample_id}">${esc(a.notes || "")}</textarea>
    </div>
  </div>`;
}

function render() {
  document.getElementById("cards").innerHTML =
    stageSamples().map(cardHTML).join("");
  document.getElementById("guide1").style.display = stage === 1 ? "" : "none";
  document.getElementById("guide2").style.display = stage === 2 ? "" : "none";
  document.querySelectorAll(".stagebtn").forEach(b =>
    b.classList.toggle("active", Number(b.dataset.stage) === stage));
  applyFilter();
  refreshProgress();
}

function refreshCard(id) {
  const card = document.getElementById("card_" + id);
  if (card) card.classList.toggle("done", isDone(id));
}
function refreshProgress() {
  const s1 = SAMPLES.filter(s => get(s.sample_id).relation_label &&
                                 get(s.sample_id).annotator_confidence).length;
  const fal = SAMPLES.filter(s => s.original_label === "falsified");
  const s2 = fal.filter(s => get(s.sample_id).evidence_channel).length;
  document.getElementById("progress").textContent =
    `S1 ${s1}/${SAMPLES.length} · S2 ${s2}/${fal.length}`;
}
function applyFilter() {
  const f = document.querySelector(".filterbtn.active").dataset.f;
  stageSamples().forEach(s => {
    const show = f === "all" || (f === "done") === isDone(s.sample_id);
    const card = document.getElementById("card_" + s.sample_id);
    if (card) card.style.display = show ? "" : "none";
  });
}

document.addEventListener("change", e => {
  const grp = e.target.closest(".opts");
  if (grp) set(Number(grp.dataset.id), grp.dataset.field, e.target.value);
});
document.addEventListener("input", e => {
  if (e.target.tagName === "TEXTAREA")
    set(Number(e.target.dataset.id), "notes", e.target.value);
});
document.addEventListener("click", e => {
  if (e.target.matches(".card img")) {
    const lb = document.getElementById("lightbox");
    lb.querySelector("img").src = e.target.dataset.full;
    lb.style.display = "flex";
  } else if (e.target.closest("#lightbox")) {
    document.getElementById("lightbox").style.display = "none";
  }
  const st = e.target.closest(".stagebtn");
  if (st) {
    const target = Number(st.dataset.stage);
    if (target === 2 && !stage2Warned) {
      const s1done = SAMPLES.every(s => get(s.sample_id).relation_label);
      if (!confirm(
        (s1done ? "" : "Stage 1 is NOT complete. ") +
        "Stage 2 reveals which samples are falsified and shows original " +
        "captions. Relation labels made after this point are no longer " +
        "blind. Continue?")) return;
      stage2Warned = true;
    }
    stage = target;
    render();
  }
  const f = e.target.closest(".filterbtn");
  if (f) {
    document.querySelectorAll(".filterbtn").forEach(b => b.classList.remove("active"));
    f.classList.add("active");
    applyFilter();
  }
});

// --- CSV export ---
function csvCell(v) {
  v = v == null ? "" : String(v);
  return /[",\\n\\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}
document.getElementById("exportBtn").addEventListener("click", () => {
  const lines = [CSV_FIELDS.join(",")];
  for (const s of SAMPLES) {
    const a = get(s.sample_id);
    lines.push(CSV_FIELDS.map(f => {
      if (f === "evidence_channel")
        return csvCell(s.original_label === "pristine" ? "n/a"
                       : (a.evidence_channel || ""));
      return csvCell(f in s ? s[f] : (a[f] || ""));
    }).join(","));
  }
  const blob = new Blob([lines.join("\\r\\n") + "\\r\\n"], {type: "text/csv"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "pilot_annotations.csv";
  link.click();
});

// --- CSV import (accepts the exported format / pilot_annotations.csv) ---
function parseCSV(text) {
  const rows = []; let row = [], cell = "", inQ = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"') { if (text[i+1] === '"') { cell += '"'; i++; } else inQ = false; }
      else cell += c;
    } else if (c === '"') inQ = true;
    else if (c === ",") { row.push(cell); cell = ""; }
    else if (c === "\\n" || c === "\\r") {
      if (c === "\\r" && text[i+1] === "\\n") i++;
      row.push(cell); cell = "";
      if (row.some(x => x !== "")) rows.push(row);
      row = [];
    } else cell += c;
  }
  row.push(cell);
  if (row.some(x => x !== "")) rows.push(row);
  return rows;
}
document.getElementById("importBtn").addEventListener("click", () =>
  document.getElementById("importFile").click());
document.getElementById("importFile").addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;
  file.text().then(text => {
    const rows = parseCSV(text);
    const head = rows.shift();
    const idx = Object.fromEntries(head.map((h, i) => [h.trim(), i]));
    if (!("sample_id" in idx)) { alert("CSV missing column: sample_id"); return; }
    let n = 0;
    for (const r of rows) {
      const id = Number(r[idx.sample_id]);
      if (!SAMPLES.some(s => s.sample_id === id)) continue;
      for (const f of ["relation_label", "evidence_channel",
                       "annotator_confidence", "notes"]) {
        const v = f in idx ? r[idx[f]] : undefined;
        if (v !== undefined && v !== "" && v !== "n/a") {
          ann[id] = ann[id] || {};
          ann[id][f] = v;
        }
      }
      n++;
    }
    localStorage.setItem(STORE_KEY, JSON.stringify(ann));
    render();
    alert("Imported " + n + " rows (non-empty fields only).");
    e.target.value = "";
  });
});

render();
</script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    page = (HTML_TEMPLATE
            .replace("__STORE_KEY__", f"{PACKET}_annotations_v2")
            .replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))
            .replace("__RELATIONS__", json.dumps(RELATIONS))
            .replace("__CHANNELS__", json.dumps(CHANNELS))
            .replace("__CONFIDENCES__", json.dumps(CONFIDENCES))
            .replace("__CSV_FIELDS__", json.dumps(CSV_FIELDS)))
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(payload)} samples, two-stage)")


if __name__ == "__main__":
    main()
