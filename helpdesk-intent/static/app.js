/* Helpdesk Intent Assistant - front-end logic (no external libraries) */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const NS = "http://www.w3.org/2000/svg";

const state = { status: null, model: null, history: [], metrics: null };

/* ---------- tabs ---------- */
$$(".tab").forEach((b) => b.addEventListener("click", () => {
  $$(".tab").forEach((t) => t.classList.toggle("active", t === b));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "bench") renderBench();
}));

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

function showBanner(msg) {
  const b = $("#banner");
  b.textContent = msg;
  b.classList.toggle("hidden", !msg);
}

/* ---------- init ---------- */
async function init() {
  try {
    state.status = await api("/api/status");
  } catch (e) { showBanner("Cannot reach the server: " + e.message); return; }

  const avail = state.status.models.filter((m) => m.available);
  if (!avail.length) showBanner("No trained models found. Run  python prepare_data.py  then  python train.py  and restart the app.");
  state.model = (avail[0] || state.status.models[0]).name;

  $("#modelSeg").innerHTML = state.status.models.map((m) =>
    `<button class="seg" data-m="${m.name}" ${m.available ? "" : "disabled title='Not trained yet: python train.py --models " + m.name + "'"}>
       <b>${esc(m.name.toUpperCase())}</b><span>${m.available ? "tuned @ " + m.threshold.toFixed(2) : "not trained"}</span></button>`).join("");
  $$(".seg").forEach((b) => b.addEventListener("click", () => selectModel(b.dataset.m)));
  selectModel(state.model);

  $("#examples").innerHTML = state.status.examples.map((e) => `<button class="chip" type="button">${esc(e)}</button>`).join("");
  $$(".chip").forEach((c) => c.addEventListener("click", () => { $("#query").value = c.textContent; updateCounter(); classify(); }));

  try { state.history = JSON.parse(localStorage.getItem("hd_hist") || "[]"); } catch { state.history = []; }
  renderHistory();
}

function selectModel(name) {
  state.model = name;
  $$(".seg").forEach((b) => b.classList.toggle("active", b.dataset.m === name));
  resetThreshold();
}
function modelInfo(name) { return state.status.models.find((m) => m.name === name); }
function resetThreshold() {
  const t = modelInfo(state.model)?.threshold ?? 0.6;
  $("#thr").value = t;
  $("#thrVal").textContent = (+t).toFixed(2);
}
$("#thr").addEventListener("input", (e) => { $("#thrVal").textContent = (+e.target.value).toFixed(2); });
$("#thrReset").addEventListener("click", resetThreshold);
function updateCounter() { $("#counter").textContent = `${$("#query").value.length} / 500`; }
$("#query").addEventListener("input", updateCounter);
$("#query").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) classify(); });
$("#go").addEventListener("click", classify);

/* ---------- classify ---------- */
async function classify() {
  const text = $("#query").value.trim();
  if (!text) { $("#query").focus(); return; }
  const btn = $("#go");
  btn.disabled = true; btn.textContent = "Classifying...";
  try {
    const r = await api("/api/predict", { text, model: state.model, threshold: +$("#thr").value });
    showBanner("");
    renderResult(r);
    state.history.unshift({ text, label: r.label, conf: r.confidence, fb: r.fallback, model: r.model });
    state.history = state.history.slice(0, 15);
    try { localStorage.setItem("hd_hist", JSON.stringify(state.history)); } catch {}
    renderHistory();
  } catch (e) { showBanner(e.message); }
  btn.disabled = false; btn.textContent = "Classify query";
}

function ringSVG(conf) {
  const r = 44, c = 2 * Math.PI * r;
  return `<svg width="104" height="104" viewBox="0 0 104 104"><circle class="track" cx="52" cy="52" r="${r}" fill="none" stroke-width="9"/>
    <circle class="val" cx="52" cy="52" r="${r}" fill="none" stroke-width="9" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - conf)}"/></svg>`;
}

function renderResult(r) {
  $("#resultEmpty").classList.add("hidden");
  const body = $("#resultBody");
  body.classList.remove("hidden");
  const dept = r.department;
  body.innerHTML = `
    <div class="res-head">
      <div class="ring ${r.fallback ? "low" : ""}">${ringSVG(r.confidence)}<div class="num"><div>${pct(r.confidence, 0)}<small>confidence</small></div></div></div>
      <div>
        <span class="tag ${r.fallback ? "warn" : ""}">${r.fallback ? "Fallback" : "Predicted: " + esc(r.label)}</span>
        <div class="res-dept">${esc(dept.title)}</div>
        <div class="muted small">${esc(dept.desc)}</div>
      </div>
    </div>
    ${r.fallback ? `<div class="fallback-box"><b>Low confidence.</b> Best guess was <b>${esc(r.label)}</b> at ${pct(r.confidence)}, below the ${(+r.threshold).toFixed(2)} threshold, so this query is handed to a human agent instead of risking a wrong desk.</div>` : ""}
    <div class="bars">${r.probabilities.map((p, i) => `
      <div class="bar-row"><span class="name">${esc(p.label)}</span>
        <div class="bar ${i === 0 ? "top" : ""}"><i style="width:${(p.p * 100).toFixed(1)}%"></i><span class="tick" style="left:${r.threshold * 100}%" title="threshold"></span></div>
        <span class="pct">${pct(p.p)}</span></div>`).join("")}</div>
    <div class="meta"><span>Model: ${esc(r.model_title)}</span><span>${r.latency_ms.toFixed(1)} ms</span><span>White tick = fallback threshold</span></div>`;
}

function renderHistory() {
  const h = $("#history");
  if (!state.history.length) { h.innerHTML = `<p class="muted small">Nothing yet.</p>`; return; }
  h.innerHTML = state.history.map((x, i) => `
    <div class="h-item" data-i="${i}"><span class="txt">${esc(x.text)}</span>
      <span class="tag ${x.fb ? "warn" : ""}">${x.fb ? "fallback" : esc(x.label)}</span>
      <span class="muted small">${pct(x.conf, 0)} · ${esc(x.model)}</span></div>`).join("");
  $$(".h-item").forEach((el) => el.addEventListener("click", () => { $("#query").value = state.history[el.dataset.i].text; updateCounter(); window.scrollTo({ top: 0, behavior: "smooth" }); }));
}
$("#clearHist").addEventListener("click", () => { state.history = []; try { localStorage.removeItem("hd_hist"); } catch {} renderHistory(); });

/* ---------- compare ---------- */
$("#cmpGo").addEventListener("click", compare);
$("#cmpQuery").addEventListener("keydown", (e) => { if (e.key === "Enter") compare(); });
async function compare() {
  const text = $("#cmpQuery").value.trim();
  if (!text) return;
  const out = $("#cmpOut");
  out.innerHTML = `<p class="muted">Running models...</p>`;
  try {
    const { results } = await api("/api/compare", { text });
    if (!results.length) { out.innerHTML = `<p class="muted">No trained models yet.</p>`; return; }
    const labels = new Set(results.filter((r) => !r.fallback).map((r) => r.label));
    const disagree = labels.size > 1 || (labels.size === 1 && results.some((r) => r.fallback));
    out.innerHTML = results.map((r) => `
      <div class="cmp ${disagree ? "diff" : ""}">
        <h3>${esc(r.model_title)}</h3>
        <span class="tag ${r.fallback ? "warn" : ""}">${r.fallback ? "Fallback to human" : "Routed"}</span>
        <div class="big">${r.fallback ? "Human agent" : esc(r.label)}</div>
        <div class="muted small">best guess: ${esc(r.label)} · ${pct(r.confidence)} (threshold ${(+r.threshold).toFixed(2)})</div>
        <div class="bars">${r.probabilities.slice(0, 3).map((p, i) => `<div class="bar-row"><span class="name">${esc(p.label)}</span><div class="bar ${i === 0 ? "top" : ""}"><i style="width:${p.p * 100}%"></i></div><span class="pct">${pct(p.p, 0)}</span></div>`).join("")}</div>
        <div class="meta"><span>${r.latency_ms.toFixed(1)} ms</span></div>
      </div>`).join("");
  } catch (e) { out.innerHTML = ""; showBanner(e.message); }
}

/* ---------- benchmarks ---------- */
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
const COLORS = { blue: "#7fb2ff", cherry: "#e0456f", pale: "#c9d3f0" };

async function renderBench() {
  if (!state.metrics) { try { state.metrics = await api("/api/metrics"); } catch { state.metrics = { models: {} }; } }
  const m = state.metrics, ORDER = ["tfidf", "gru", "lstm", "bert"];
  const names = Object.keys(m.models || {}).sort((a, b) => ORDER.indexOf(a) - ORDER.indexOf(b));
  $("#benchEmpty").classList.toggle("hidden", names.length > 0);
  $("#benchBody").classList.toggle("hidden", names.length === 0);
  if (!names.length) return;

  const d = m.dataset || {};
  $("#dsInfo").textContent = `train ${d.train} · val ${d.val} · test ${d.test} · out-of-scope test ${d.oos_test}`;
  const bestAcc = Math.max(...names.map((n) => m.models[n].accuracy));
  $("#leader").innerHTML = `<thead><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>Threshold</th><th>Auto-handled</th><th>Accepted acc.</th><th>OOS to human</th><th>Train (s)</th><th>ms/query</th></tr></thead><tbody>` +
    names.map((n) => { const r = m.models[n], a = r.at_threshold;
      return `<tr class="${r.accuracy === bestAcc ? "best" : ""}"><td><b>${esc(n.toUpperCase())}</b><br><span class="muted small">${esc(r.title || "")}</span></td><td>${pct(r.accuracy)}</td><td>${pct(r.macro_f1)}</td><td>${r.threshold.toFixed(2)}</td><td>${pct(a.coverage)}</td><td>${pct(a.accepted_accuracy)}</td><td>${pct(a.oos_rejection)}</td><td>${r.train_seconds.toFixed(1)}</td><td>${r.latency_ms.toFixed(1)}</td></tr>`; }).join("") + "</tbody>";

  drawBars(names, m);
  const sel = $("#benchModel");
  sel.innerHTML = names.map((n) => `<option value="${n}">${n.toUpperCase()}</option>`).join("");
  sel.onchange = () => { drawSweep(m.models[sel.value]); drawCM(m.models[sel.value], m.labels); };
  drawSweep(m.models[names[0]]); drawCM(m.models[names[0]], m.labels);
}

function drawBars(names, m) {
  const W = 480, H = 260, pad = { l: 40, r: 10, t: 14, b: 34 };
  const host = $("#barChart"); host.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, host);
  const lo = Math.max(0, Math.floor(Math.min(...names.map((n) => Math.min(m.models[n].accuracy, m.models[n].macro_f1))) * 20) / 20 - 0.05);
  const y = (v) => pad.t + (H - pad.t - pad.b) * (1 - (v - lo) / (1 - lo));
  for (let v = Math.ceil(lo * 10) / 10; v <= 1.0001; v += 0.1) {
    el("line", { x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v), stroke: "rgba(140,160,225,.15)" }, svg);
    el("text", { x: pad.l - 6, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = Math.round(v * 100) + "%";
  }
  const gw = (W - pad.l - pad.r) / names.length;
  names.forEach((n, i) => {
    [["accuracy", COLORS.cherry], ["macro_f1", COLORS.blue]].forEach(([k, c], j) => {
      const v = m.models[n][k], bw = gw * 0.28, x = pad.l + gw * i + gw * 0.5 - bw + j * bw;
      el("rect", { x, y: y(v), width: bw - 3, height: H - pad.b - y(v), fill: c, rx: 4 }, svg);
      el("text", { x: x + (bw - 3) / 2, y: y(v) - 4, "text-anchor": "middle" }, svg).textContent = (v * 100).toFixed(1);
    });
    el("text", { x: pad.l + gw * i + gw / 2, y: H - 12, "text-anchor": "middle" }, svg).textContent = n.toUpperCase();
  });
  host.insertAdjacentHTML("beforeend", `<div class="legend"><span><i style="background:${COLORS.cherry}"></i>Accuracy</span><span><i style="background:${COLORS.blue}"></i>Macro-F1</span><span>Axis starts at ${Math.round(lo * 100)}% to make small gaps visible</span></div>`);
}

function drawSweep(r) {
  const W = 480, H = 260, pad = { l: 40, r: 10, t: 14, b: 34 };
  const host = $("#sweepChart"); host.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, host);
  const x = (t) => pad.l + (W - pad.l - pad.r) * t, y = (v) => pad.t + (H - pad.t - pad.b) * (1 - v);
  for (let v = 0; v <= 1.0001; v += 0.25) {
    el("line", { x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v), stroke: "rgba(140,160,225,.15)" }, svg);
    el("text", { x: pad.l - 6, y: y(v) + 4, "text-anchor": "end" }, svg).textContent = Math.round(v * 100) + "%";
    el("text", { x: x(v), y: H - 14, "text-anchor": "middle" }, svg).textContent = v.toFixed(2);
  }
  [["coverage", COLORS.pale, "In-scope auto-handled"], ["accepted_accuracy", COLORS.blue, "Accuracy of accepted"], ["oos_rejection", COLORS.cherry, "Out-of-scope sent to human"]].forEach(([k, c]) => {
    const pts = r.sweep.map((s) => `${x(s.threshold)},${y(s[k])}`).join(" ");
    el("polyline", { points: pts, fill: "none", stroke: c, "stroke-width": 2.4, "stroke-linejoin": "round" }, svg);
  });
  el("line", { x1: x(r.threshold), x2: x(r.threshold), y1: pad.t, y2: H - pad.b, stroke: "#fff", "stroke-dasharray": "4 4", opacity: .7 }, svg);
  host.insertAdjacentHTML("beforeend", `<div class="legend"><span><i style="background:${COLORS.pale}"></i>In-scope auto-handled</span><span><i style="background:${COLORS.blue}"></i>Accuracy of accepted</span><span><i style="background:${COLORS.cherry}"></i>Out-of-scope to human</span></div>`);
}

function drawCM(r, labels) {
  const cm = r.confusion_matrix, host = $("#cm"), n = labels.length;
  host.innerHTML = `<div class="cm" style="grid-template-columns: 90px repeat(${n}, 1fr)"><div></div>${labels.map((l) => `<div class="lab">pred ${esc(l)}</div>`).join("")}` +
    cm.map((row, i) => {
      const tot = row.reduce((a, b) => a + b, 0) || 1;
      return `<div class="lab">true ${esc(labels[i])}</div>` + row.map((v, j) => {
        const a = v / tot, bg = i === j ? `rgba(127,178,255,${0.12 + a * 0.7})` : `rgba(215,71,111,${v ? 0.2 + a * 1.4 : 0.04})`;
        return `<div class="cell" style="background:${bg}">${v}</div>`; }).join("");
    }).join("") + "</div>";
}

init();
