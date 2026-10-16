"use strict";

// IBIS 0.3.2 interface revision 4. The data table is editable without the
// Python backend; only import and Bayesian calculations require the server.
if (typeof window !== "undefined") {
  window.addEventListener("error", event => {
    const status = document.getElementById("uiHealth");
    if (status) {
      status.textContent = `Interface error: ${event.message}. Check that index.html and script.js come from the same download.`;
      status.classList.remove("ready");
    }
  });
}

const fields = ["Depth", "U234_r", "U234_r_err", "Th230_r", "Th230_r_err", "Th232_r", "Th232_r_err", "Depth_err"];
const tableBody = document.querySelector("#inputTable tbody");
const $ = id => document.getElementById(id);
let priorJobId = null;
let priorSnapshot = null;
let resultsRows = null;
let farajiReferences = null;
let busy = false;

function addRow(values = []) {
  // insertRow() attaches the row immediately; detached rows were the original
  // cause of the empty table and non-functional Add row/Faraji buttons.
  const tr = tableBody.insertRow();
  const number = tableBody.rows.length;
  const index = tr.insertCell();
  index.className = "index";
  index.textContent = number;
  fields.forEach((_, i) => {
    const td = tr.insertCell();
    td.className = "editable";
    td.contentEditable = "true";
    td.spellcheck = false;
    td.textContent = values[i] == null ? "" : String(values[i]);
    td.setAttribute("aria-label", `${fields[i]} at row ${number}`);
  });
  return tr;
}

function clearTable() {
  tableBody.replaceChildren();
  for (let i = 0; i < 8; i++) addRow();
  changed();
}

function countRows() {
  let n = 0;
  for (const row of tableBody.rows) {
    if (Array.from(row.cells).slice(1, 8).some(cell => cell.textContent.trim())) n++;
  }
  $("dataCount").textContent = n;
}

function changed({preserveReference = false} = {}) {
  if (!preserveReference) farajiReferences = null;
  countRows();
  if (!busy) {
    priorJobId = null;
    priorSnapshot = null;
    resultsRows = null;
    $("runButton").disabled = true;
    $("priorResult").classList.add("hidden");
    $("ibisResult").classList.add("hidden");
    $("farajiResult").classList.add("hidden");
    $("jobText").textContent = "Changes detected. Generate the Thoth prior to continue.";
  }
}

function readData() {
  const records = [];
  for (let i = 0; i < tableBody.rows.length; i++) {
    const cells = Array.from(tableBody.rows[i].cells).slice(1);
    const vals = cells.map(cell => cell.textContent.trim());
    if (vals.every(v => !v)) continue;
    if (vals.slice(0, 7).some(v => !v)) throw new Error(`Row ${i + 1}: all seven required columns need numbers.`);
    const record = {};
    fields.forEach((field, k) => {
      if (!vals[k] && field === "Depth_err") record[field] = null;
      else {
        const number = Number(vals[k]);
        if (!Number.isFinite(number)) throw new Error(`Row ${i + 1}: invalid ${field}.`);
        if (k > 0 && (k <= 6) && number <= 0) throw new Error(`Row ${i + 1}: ratios and 1σ errors must be positive.`);
        if (field === "Depth_err" && number < 0) throw new Error(`Row ${i + 1}: depth uncertainty cannot be negative.`);
        record[field] = number;
      }
    });
    records.push(record);
  }
  if (records.length < 3) throw new Error("Enter at least three complete measurements.");
  if (new Set(records.map(r => r.Depth)).size < 2) throw new Error("At least two distinct depths are required.");
  return records;
}

function numeric(id) {
  const number = Number($(id).value);
  if (!Number.isFinite(number)) throw new Error(`Enter a valid number for ${id}.`);
  return number;
}

function readSettings() {
  return {
    fraction_det: numeric("fractionDet"), global_mass: numeric("globalMass"),
    use_r02_in_sigma: $("useR02Sigma").checked,
    n_samples_global: numeric("nGlobal"), n_samples_layer: numeric("nLayer"),
    neighbors: numeric("neighbors"), n_chains: numeric("chains"),
    iterations: numeric("iterations"), burn_in: numeric("burnIn")
  };
}

function snapshot() {
  return JSON.stringify({rows: readData(), settings: readSettings(), reference: farajiReferences});
}

function loading(on, message = "") {
  busy = on;
  $("spinner").classList.toggle("hidden", !on);
  $("priorButton").disabled = on;
  $("runButton").disabled = on || !priorJobId;
  if (message) $("jobText").textContent = message;
}

function parseError(data) {
  if (typeof data?.detail === "string") return data.detail;
  if (Array.isArray(data?.detail)) return data.detail.map(e => e.msg || JSON.stringify(e)).join("; ");
  return "The request failed.";
}

async function post(path, body) {
  const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(parseError(data));
  return data;
}

async function waitForJob(jobId) {
  while (true) {
    await new Promise(resolve => setTimeout(resolve, 1400));
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {cache: "no-store"});
    const data = await response.json();
    if (!response.ok) throw new Error(parseError(data));
    $("jobText").textContent = data.stage || "Processing";
    if (data.status === "error") throw new Error(data.error || "The model did not complete.");
    if (data.status === "complete") return data.result;
  }
}

function showPrior(data) {
  const s = data.prior_summary;
  $("priorMedian").textContent = s.median.toFixed(4);
  $("priorLo").textContent = s.lo95.toFixed(4);
  $("priorHi").textContent = s.hi95.toFixed(4);
  $("priorPlot").src = `data:image/png;base64,${data.prior_plot}`;
  $("priorResult").classList.remove("hidden");
}

const formatted = number => Number(number).toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1});
const four = number => Number(number).toFixed(4);

function showIBIS(data) {
  resultsRows = data.rows;
  $("agePlot").src = `data:image/png;base64,${data.age_plot}`;
  $("thoriumPlot").src = `data:image/png;base64,${data.thorium_plot}`;
  const comparison = $("farajiResult");
  if (data.faraji_plot && data.faraji_summary) {
    $("farajiPlot").src = `data:image/png;base64,${data.faraji_plot}`;
    const stats = data.faraji_summary;
    $("farajiSummary").textContent = `${stats.n} matched samples · Mean IBIS − Faraji: ${stats.mean_difference.toFixed(3)} · RMSE: ${stats.rmse.toFixed(3)} (activity ratio units)`;
    comparison.classList.remove("hidden");
  } else {
    comparison.classList.add("hidden");
  }
  const body = document.querySelector("#resultTable tbody");
  body.replaceChildren();
  data.rows.forEach(r => {
    const values = [formatted(r.Depth),
      `${formatted(r.uncorrected_age)} ± ${formatted(r.uncorrected_sigma)}`,
      `${formatted(r.bulk_age)} ± ${formatted(r.bulk_sigma)}`,
      `${formatted(r.ibis_age)} ± ${formatted(r.ibis_sigma)}`,
      `${formatted(r.ibis_lo95)} – ${formatted(r.ibis_hi95)}`,
      `${four(r.th0)} ± ${four(r.th0_sigma)}`];
    const tr = body.insertRow();
    values.forEach(value => {const td = tr.insertCell(); td.textContent = value;});
  });
  $("ibisResult").classList.remove("hidden");
}

$("priorButton").addEventListener("click", async () => {
  try {
    const body = {rows: readData(), settings: readSettings(), reference: farajiReferences};
    const expected = JSON.stringify(body);
    loading(true, "Preparing Thoth prior...");
    $("priorResult").classList.add("hidden");
    $("ibisResult").classList.add("hidden");
    priorJobId = null;
    const job = await post("/api/prior", body);
    const result = await waitForJob(job.job_id);
    if (snapshot() !== expected) throw new Error("Inputs changed during the run. Generate the prior again.");
    priorJobId = job.job_id;
    priorSnapshot = expected;
    showPrior(result);
    loading(false, "Thoth prior complete. Ready to run IBIS Part 1.");
    $("priorResult").scrollIntoView({behavior: "smooth", block: "start"});
  } catch (error) {
    priorJobId = null;
    loading(false, `Error: ${error.message}`);
  }
});

$("runButton").addEventListener("click", async () => {
  try {
    if (!priorJobId || snapshot() !== priorSnapshot) throw new Error("Generate the Thoth prior using the current inputs first.");
    loading(true, "Starting Bayesian age estimation...");
    const job = await post("/api/part1", {prior_job_id: priorJobId});
    const result = await waitForJob(job.job_id);
    if (snapshot() !== priorSnapshot) throw new Error("Inputs changed during the run. Results have not been displayed.");
    showIBIS(result);
    loading(false, "IBIS Part 1 complete. Review the age comparisons and download the CSV.");
    $("ibisResult").scrollIntoView({behavior: "smooth", block: "start"});
  } catch (error) {loading(false, `Error: ${error.message}`);}
});

function escapeCsv(value) {
  const string = String(value ?? "");
  return /[\n\r,\"]/.test(string) ? `"${string.replaceAll('"', '""')}"` : string;
}

$("downloadButton").addEventListener("click", () => {
  if (!resultsRows?.length) return;
  const keys = Object.keys(resultsRows[0]);
  const csv = [keys.join(","), ...resultsRows.map(r => keys.map(k => escapeCsv(r[k])).join(","))].join("\r\n");
  const link = document.createElement("a");
  const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
  link.href = url;
  link.download = "IBIS_Part1_Age_Results.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
});

function activity(age, u48, r28, initialTh) {
  const l230 = 9.17055e-6;
  const l234 = 2.82203e-6;
  return 1 - Math.exp(-l230 * age) * (1 - initialTh*r28)
    + (u48 - 1) * l230/(l230-l234) * (1 - Math.exp(-(l230-l234)*age));
}

function exampleRows() {
  const ages = [1600, 2500, 3900, 5200, 7000, 8500, 10200, 12000];
  return ages.map((age, i) => {
    const depth = 5 + 8*i;
    const r48 = 1.085 + 0.004*Math.sin(i * 0.7);
    const r28 = [0.0030,0.0050,0.0025,0.0040,0.0020,0.0055,0.0035,0.0045][i];
    const r08 = activity(age, r48, r28, 0.7 + 0.12*Math.cos(i));
    return [depth, r48.toFixed(6), "0.0008", r08.toFixed(6), "0.00045", r28.toFixed(6), "0.00010", "0.1"];
  });
}

$("exampleButton").addEventListener("click", () => {
  tableBody.replaceChildren();
  exampleRows().forEach(row => addRow(row));
  changed();
  $("dataStatus").textContent = "Synthetic example loaded";
});

// Bundled directly with the page so the test button never depends on a separate
// CSV request. Order follows increasing sampling depth, not sample name.
const farajiPreset = [
  ["*Pu17-1",  6.0, 1.0263, 0.0020, 0.01740, 0.00035, 0.002117, 0.000042, 8.05, 0.24],
  ["Pu17-2",  11.2, 1.0223, 0.0019, 0.00617, 0.00019, 0.000912, 0.000018, 6.00, 0.25],
  ["Pu17-3",  15.2, 1.0221, 0.0018, 0.00598, 0.00016, 0.000837, 0.000017, 5.98, 0.27],
  ["Pu17-4",  19.7, 1.0334, 0.0020, 0.00758, 0.00024, 0.000954, 0.000019, 6.63, 0.25],
  ["*Pu17-5", 29.4, 1.0166, 0.0021, 0.06950, 0.00080, 0.008750, 0.000170, 7.74, 0.18],
  ["Pu17-6",  34.5, 1.0212, 0.0021, 0.01310, 0.00027, 0.001665, 0.000033, 6.55, 0.21],
  ["PU17-10", 38.4, 1.0224, 0.0019, 0.00477, 0.00021, 0.000387, 0.000008, 6.01, 0.50],
  ["Pu17-7",  39.3, 1.0280, 0.0018, 0.00528, 0.00018, 0.000444, 0.000009, 6.21, 0.46],
  ["Pu17-8",  45.0, 1.0329, 0.0019, 0.00859, 0.00024, 0.000828, 0.000017, 6.87, 0.27],
  ["PU17-9",  48.2, 1.0283, 0.0018, 0.00409, 0.00019, 0.000204, 0.000004, 4.88, 0.97]
];

$("farajiButton").addEventListener("click", () => {
  tableBody.replaceChildren();
  farajiPreset.forEach(([, ...values]) => addRow([...values.slice(0, 7), ""]));
  farajiReferences = farajiPreset.map(r =>
    ({sample: r[0], depth: r[1], initial_th: r[8], sigma: r[9]}));
  changed({preserveReference: true});
  $("dataStatus").textContent = "Faraji et al. (2021) loaded · 10 analyses with reference values · Generate the Thoth prior next";
});

$("addRowButton").addEventListener("click", () => {addRow(); changed();});
$("clearButton").addEventListener("click", () => {clearTable(); $("dataStatus").textContent = "Table cleared";});
tableBody.addEventListener("input", changed);

tableBody.addEventListener("paste", event => {
  const cell = event.target.closest("td.editable");
  if (!cell) return;
  const text = event.clipboardData.getData("text/plain").trimEnd();
  if (!text.includes("\t") && !text.includes("\n")) return;
  event.preventDefault();
  const matrix = text.replaceAll("\r", "").split("\n").map(row => row.split("\t"));
  const col = cell.cellIndex - 1;
  if (matrix.some(row => row.length + col > fields.length)) {
    $("dataStatus").textContent = "Pasted data extend past the last column.";
    return;
  }
  const start = cell.parentElement.sectionRowIndex;
  while (tableBody.rows.length < start + matrix.length) addRow();
  matrix.forEach((row, i) => row.forEach((value, j) => {
    tableBody.rows[start+i].cells[col+j+1].textContent = value.trim();
  }));
  changed();
  $("dataStatus").textContent = `${matrix.length} rows pasted`;
});

// Handles ordinary CSV quoting, including commas inside quoted cells.
function parseCsv(source) {
  let rows = [], row = [], value = "", quoted = false;
  for (let i = 0; i < source.length; i++) {
    const c = source[i];
    if (c === '"') {
      if (quoted && source[i+1] === '"') {value += '"'; i++;}
      else quoted = !quoted;
    } else if (c === "," && !quoted) {row.push(value); value = "";}
    else if ((c === "\n" || c === "\r") && !quoted) {
      if (c === "\r" && source[i+1] === "\n") i++;
      row.push(value); value = "";
      if (row.some(v => v.trim())) rows.push(row);
      row = [];
    } else value += c;
  }
  row.push(value);
  if (row.some(v => v.trim())) rows.push(row);
  return rows;
}

$("csvFile").addEventListener("change", async event => {
  try {
    const file = event.target.files?.[0];
    if (!file) return;
    const response = await fetch(`/api/import?filename=${encodeURIComponent(file.name)}`, {
      method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: file
    });
    const data = await response.json();
    if (!response.ok) throw new Error(parseError(data));
    tableBody.replaceChildren();
    data.rows.forEach(row => addRow(fields.map(field => row[field] ?? "")));
    changed();
    $("dataStatus").textContent = `${file.name} loaded · ${data.rows.length} analyses`;
  } catch (error) {$("dataStatus").textContent = `Error: ${error.message}`;}
  finally {event.target.value = "";}
});

function refreshLabels() {
  $("fractionDetValue").textContent = `${Math.round(numeric("fractionDet") * 100)}%`;
  $("globalMassValue").textContent = `${Math.round(numeric("globalMass") * 100)}%`;
}

document.querySelectorAll("#settings input").forEach(el => el.addEventListener("input", () => {refreshLabels(); changed({preserveReference: true});}));
clearTable();
refreshLabels();
const uiHealth = $("uiHealth");
if (uiHealth) {
  uiHealth.textContent = "✓ Table controls ready · Interface revision 4";
  uiHealth.classList.add("ready");
}
if (typeof location !== "undefined" && location.protocol === "file:") {
  $("jobText").textContent = "The editable table works here, but to calculate Thoth or IBIS ages, start the Python server and open http://127.0.0.1:8000 instead of this file.";
}
