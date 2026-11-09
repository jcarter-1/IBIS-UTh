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
// Build verification: this status only appears when the v2 JavaScript actually runs.
const ASCII_CSV_EXPORT_VERSION = "ASCII-v2-20261102";
const csvExporterStatus = $("csvExporterStatus");
if (csvExporterStatus) csvExporterStatus.textContent =
  "ASCII CSV exporter v2 loaded - headings use 1sigma (plain text).";

let priorJobId = null;
let priorSnapshot = null;
let resultsRows = null;
let conventionalRows = null;
let conventionalTimer = null;
let conventionalRequestId = 0;
let farajiReferences = null;
let busy = false;

// ========
// ========
// Data Exploration
// ========
// ========
// ========================================================
// STANDARDIZED ACTIVITY RATIOS VS DEPTH
// ========================================================

// ========================================================
// CALCULATE UNCORRECTED AND BULK EARTH AGES
// ========================================================

async function updateConventionalAges() {

    // Each request gets a unique number so older results
    // cannot overwrite newer measurements.
    const requestId = ++conventionalRequestId;

    let rows;

    try {
        rows = readData();
    } catch (error) {
        conventionalRows = null;
        plotAgeComparison();
        return;
    }

    try {
        const data = await post(
            "/api/conventional-ages",
            {rows: rows}
        );

        if (requestId !== conventionalRequestId) return;

        conventionalRows = data.rows;

        // Draw the plot!
        plotAgeComparison();

    } catch (error) {

        if (requestId !== conventionalRequestId) return;

        conventionalRows = null;
        plotAgeComparison();

        console.error(
            "Conventional age calculation failed:",
            error
        );
    }
}

function plotStandardizedData() {

    const plotDiv = document.getElementById("plotStandardized");

    if (!plotDiv || typeof Plotly === "undefined") return;

    const rows = document.querySelectorAll("#inputTable tbody tr");

    const depth = [];
    const depthErr = [];

    const u234 = [];
    const u234Err = [];

    const th230 = [];
    const th230Err = [];

    const th232 = [];
    const th232Err = [];

    // ----------------------------------------
    // Read measurements
    // ----------------------------------------

    rows.forEach(row => {

        const cells = row.querySelectorAll("td");

        const values = [
            1, 2, 3, 4, 5, 6, 7, 8
        ].map(i => {
            const text = cells[i]?.textContent.trim();
            return text === "" || text == null ? NaN : Number(text);
        });

        const [d, u48, su48, r08, sr08, r28, sr28, sd] = values;

        if (
            [d, u48, su48, r08, sr08, r28, sr28]
                .every(Number.isFinite)
        ) {
            depth.push(d);
            depthErr.push(
                Number.isFinite(sd) && sd >= 0 ? sd : 0
            );

            u234.push(u48);
            u234Err.push(su48);

            th230.push(r08);
            th230Err.push(sr08);

            th232.push(r28);
            th232Err.push(sr28);
        }
    });

    if (depth.length < 2) {
        Plotly.purge(plotDiv);
        return;
    }

    // ----------------------------------------
    // Standardization function
    // ----------------------------------------

    function standardize(values, errors) {

        const n = values.length;

        const mean = values.reduce(
            (a, b) => a + b, 0
        ) / n;

        const variance = values.reduce(
            (a, b) => a + (b - mean)**2, 0
        ) / (n - 1);

        const sd = Math.sqrt(variance);

        if (!(sd > 0)) {
            return {
                z: values.map(() => 0),
                zErr: errors.map(() => 0)
            };
        }

        return {
            z: values.map(v => (v - mean) / sd),
            zErr: errors.map(e => e / sd)
        };
    }

    const U234 = standardize(u234, u234Err);
    const Th230 = standardize(th230, th230Err);
    const Th232 = standardize(th232, th232Err);

    // ----------------------------------------
    // Create traces
    // ----------------------------------------

    function makeTrace(name, values, errors, color) {

        return {
            x: values,
            y: depth,

            type: "scatter",
            mode: "markers+lines",
            name: name,

            marker: {
                size: 7,
                color: color
            },

            line: {
                color: color,
                width: 1.5
            },

            error_x: {
                type: "data",
                array: errors,
                visible: true,
                color: color,
                thickness: 1,
                width: 3
            },

            error_y: {
                type: "data",
                array: depthErr,
                visible: true,
                color: color,
                thickness: 1,
                width: 3
            },

            hovertemplate:
                name +
                "<br>Standardized ratio: %{x:.3f}" +
                "<br>Depth: %{y:.2f}" +
                "<extra></extra>"
        };
    }

    const traces = [
        makeTrace("²³⁴U/²³⁸U", U234.z, U234.zErr, "#213555"),
        makeTrace("²³⁰Th/²³⁸U", Th230.z, Th230.zErr, "#C76D5E"),
        makeTrace("²³²Th/²³⁸U", Th232.z, Th232.zErr, "#729A91")
    ];

    // ----------------------------------------
    // Depth limits
    // ----------------------------------------

    const minDepth = Math.min(
        ...depth.map((d, i) => d - depthErr[i])
    );

    const maxDepth = Math.max(
        ...depth.map((d, i) => d + depthErr[i])
    );

    const padding = Math.max(
        (maxDepth - minDepth) * 0.05,
        0.1
    );

    // ----------------------------------------
    // Layout
    // ----------------------------------------

    const layout = {

        paper_bgcolor: "white",
        plot_bgcolor: "white",

        height: 550,

        margin: {
            l: 75,
            r: 30,
            t: 35,
            b: 70
        },

        xaxis: {
            title: "Standardized activity ratio (Z-score)",
            zeroline: true,
            zerolinecolor: "#999",
            zerolinewidth: 1,
            showline: true,
            linecolor: "#333",
            ticks: "outside"
        },

        yaxis: {
            title: "Depth",
            range: [
                maxDepth + padding,
                minDepth - padding
            ],
            showline: true,
            linecolor: "#333",
            ticks: "outside"
        },

        legend: {
            orientation: "h",
            x: 0.5,
            xanchor: "center",
            y: 1.08,
            yanchor: "bottom"
        },

        font: {
            family: "Arial, sans-serif",
            size: 12,
            color: "#333"
        }
    };

    const config = {
        responsive: true,
        displaylogo: false,
        scrollZoom: true
    };

    Plotly.react(plotDiv, traces, layout, config);
}



// ========================================================
// DATA EXPLORATION — CONVENTIONAL U–Th AGE COMPARISON
// ========================================================

function plotAgeComparison() {

    const plotDiv = document.getElementById("plotAgeComparison");

    if (!plotDiv || typeof Plotly === "undefined") return;

    // ----------------------------------------------------
    // 1. Check whether conventional ages are available
    // ----------------------------------------------------

    if (!conventionalRows || conventionalRows.length === 0) {

        Plotly.purge(plotDiv);

        plotDiv.textContent =
            "Enter at least three complete measurements to calculate ages.";

        return;
    }

    // ----------------------------------------------------
    // 2. Extract conventional ages and uncertainties
    // ----------------------------------------------------

    const rows = [...conventionalRows].sort(
        (a, b) => Number(a.Depth) - Number(b.Depth)
    );

    const depth = rows.map(r => Number(r.Depth));

    const uncorrected = rows.map(
        r => Number(r.uncorrected_age)
    );

    const uncorrectedErr = rows.map(
        r => Number(r.uncorrected_sigma)
    );

    const bulk = rows.map(
        r => Number(r.bulk_age)
    );

    const bulkErr = rows.map(
        r => Number(r.bulk_sigma)
    );

    // ----------------------------------------------------
    // 3. Create Plotly traces
    // ----------------------------------------------------

    function makeTrace(name, ages, errors, color, symbol) {

        return {

            x: ages,
            y: depth,

            type: "scatter",
            mode: "markers",
            name: name,

            marker: {
                size: 8,
                color: color,
                symbol: symbol,
                line: {
                    color: "white",
                    width: 1
                }
            },

            line: {
                color: color,
                width: 1.5
            },

            error_x: {
                type: "data",
                array: errors,
                visible: true,
                color: color,
                thickness: 1.3,
                width: 3
            },

            hovertemplate:
                "<b>" + name + "</b>" +
                "<br>Depth: %{y:.2f}" +
                "<br>Age: %{x:,.1f} yr" +
                "<extra></extra>"
        };
    }

    const traces = [

        makeTrace(
            "Uncorrected",
            uncorrected,
            uncorrectedErr,
            "#213555",
            "circle"
        ),

        makeTrace(
            "Bulk Earth",
            bulk,
            bulkErr,
            "#C76D5E",
            "square"
        )
    ];

    // ----------------------------------------------------
    // 4. Calculate depth limits with 5% padding
    // ----------------------------------------------------

    const minDepth = Math.min(...depth);
    const maxDepth = Math.max(...depth);

    const depthPadding = Math.max(
        (maxDepth - minDepth) * 0.05,
        0.1
    );

    // ----------------------------------------------------
    // 5. Calculate age limits including 1sigma errors
    // ----------------------------------------------------

    const ageLower = [
        ...uncorrected.map((age, i) => age - uncorrectedErr[i]),
        ...bulk.map((age, i) => age - bulkErr[i])
    ];

    const ageUpper = [
        ...uncorrected.map((age, i) => age + uncorrectedErr[i]),
        ...bulk.map((age, i) => age + bulkErr[i])
    ];

    const minAge = Math.min(...ageLower);
    const maxAge = Math.max(...ageUpper);

    const agePadding = Math.max(
        (maxAge - minAge) * 0.05,
        1
    );

    // ----------------------------------------------------
    // 6. Plot layout
    // ----------------------------------------------------

    const layout = {

        paper_bgcolor: "white",
        plot_bgcolor: "white",

        height: 520,

        margin: {
            l: 75,
            r: 30,
            t: 55,
            b: 70
        },

        xaxis: {
            title: "U–Th age (years)",

            range: [
                minAge - agePadding,
                maxAge + agePadding
            ],

            zeroline: false,
            showline: true,
            linecolor: "#333",
            ticks: "outside",
            tickformat: ",.0f"
        },

        yaxis: {
            title: "Depth",

            // Reverse stratigraphic depth
            range: [
                maxDepth + depthPadding,
                minDepth - depthPadding
            ],

            zeroline: false,
            showline: true,
            linecolor: "#333",
            ticks: "outside"
        },

        legend: {
            orientation: "h",
            x: 0.5,
            xanchor: "center",
            y: 1.10,
            yanchor: "bottom"
        },

        font: {
            family: "Arial, sans-serif",
            size: 12,
            color: "#333"
        },

        hovermode: "closest"
    };

    // ----------------------------------------------------
    // 7. Interactive plot controls
    // ----------------------------------------------------

    const config = {
        responsive: true,
        displaylogo: false,
        scrollZoom: true,
        displayModeBar: true,
        modeBarButtonsToRemove: [
            "lasso2d",
            "select2d"
        ],
        toImageButtonOptions: {
            format: "svg",
            filename: "IBIS_Conventional_Age_Comparison",
            height: 600,
            width: 800,
            scale: 1
        }
    };

    // ----------------------------------------------------
    // 8. Draw or update the plot
    // ----------------------------------------------------

    Plotly.react(plotDiv, traces, layout, config);
}



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
    $("downloadPriorButton").disabled = true;
    priorSnapshot = null;
    resultsRows = null;
    $("runButton").disabled = true;
    $("priorResult").classList.add("hidden");
    $("ibisResult").classList.add("hidden");
    $("farajiResult").classList.add("hidden");
    $("jobText").textContent = "Changes detected. Generate the Thoth prior to continue.";
  }
  plotStandardizedData();

// Request conventional ages after the user stops typing
	clearTimeout(conventionalTimer);
	conventionalRequestId++;

	conventionalTimer = setTimeout(
	    updateConventionalAges,
    500
);
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
    neighbors: numeric("neighbors")
  };
}

function readMCMCSettings() {
  return {n_chains: numeric("chains"), iterations: numeric("iterations"), burn_in: numeric("burnIn")};
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


// ========================================================
// COMPLETE IBIS RESULTS TABLE — ONE SHARED COLUMN SCHEMA
// ========================================================
// Keep the display columns and exported CSV in precisely the same order.
// The full output retains numeric precision in CSV (table formatting is display-only).
const RESULT_COLUMNS = [
  {key: "Depth",              csv: "depth",                                 digits: 2},
  {key: "Depth_err",          csv: "depth_1sigma",                          digits: 2},
  {key: "U234_r",             csv: "u234_u238_activity_ratio",              digits: 6},
  {key: "U234_r_err",         csv: "u234_u238_activity_ratio_1sigma",       digits: 6},
  {key: "Th230_r",            csv: "th230_u238_activity_ratio",             digits: 6},
  {key: "Th230_r_err",        csv: "th230_u238_activity_ratio_1sigma",      digits: 6},
  {key: "Th232_r",            csv: "th232_u238_activity_ratio",             digits: 6},
  {key: "Th232_r_err",        csv: "th232_u238_activity_ratio_1sigma",      digits: 6},
  {key: "uncorrected_age",    csv: "uncorrected_age_yr",                    digits: 1},
  {key: "uncorrected_sigma",  csv: "uncorrected_age_1sigma_yr",             digits: 1},
  {key: "bulk_age",           csv: "bulk_earth_age_yr",                    digits: 1},
  {key: "bulk_sigma",         csv: "bulk_earth_age_1sigma_yr",             digits: 1},
  {key: "ibis_age",           csv: "ibis_age_median_yr",                   digits: 1},
  {key: "ibis_sigma",         csv: "ibis_age_symmetric_1sigma_yr",         digits: 1},
  {key: "ibis_lo68",          csv: "ibis_age_68pct_lower_yr",              digits: 1},
  {key: "ibis_hi68",          csv: "ibis_age_68pct_upper_yr",              digits: 1},
  {key: "ibis_lo95",          csv: "ibis_age_95pct_lower_yr",              digits: 1},
  {key: "ibis_hi95",          csv: "ibis_age_95pct_upper_yr",              digits: 1},
  {key: "ibis_err95_minus",   csv: "ibis_age_95pct_minus_error_yr",        digits: 1},
  {key: "ibis_err95_plus",    csv: "ibis_age_95pct_plus_error_yr",         digits: 1},
  {key: "th0",                csv: "initial_th",                           digits: 4},
  {key: "th0_sigma",          csv: "initial_th_1sigma",                    digits: 4}
];

// The 95% +/- errors are the distances from the posterior median to the
// interval limits, not the lower/upper limits themselves.
function withIbisErrors(row) {
  const age = Number(row.ibis_age);
  const low = Number(row.ibis_lo95);
  const high = Number(row.ibis_hi95);
  const validAge = row.ibis_age != null && Number.isFinite(age);
  return {
    ...row,
    ibis_err95_minus: validAge && row.ibis_lo95 != null && Number.isFinite(low) ? age - low : null,
    ibis_err95_plus: validAge && row.ibis_hi95 != null && Number.isFinite(high) ? high - age : null
  };
}

function formatResultValue(value, digits = 4) {
  if (value == null || value === "") return "N/A";
  const n = Number(value);
  if (!Number.isFinite(n)) return "N/A";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
}

function populateResultsTable(rows) {
  const tbody = document.querySelector("#resultTable tbody");
  tbody.replaceChildren();
  const sortedRows = [...rows].sort((a, b) => Number(a.Depth) - Number(b.Depth));
  for (const rawRow of sortedRows) {
    const row = withIbisErrors(rawRow);
    const tr = tbody.insertRow();
    for (const column of RESULT_COLUMNS) {
      const td = tr.insertCell();
      td.textContent = formatResultValue(row[column.key], column.digits);
    }
  }
}


// =========================================================
// INTERACTIVE POST-IBIS FIGURES — figures export as SVG/PNG
// =========================================================
const ibisColors = {navy:'#213555', red:'#C76D5E', sage:'#729A91', gold:'#D8B26E', grey:'#8997A8'};

function ibisPlotConfig(filename) {
    return {
        responsive: true, scrollZoom: true, displaylogo: false,
        toImageButtonOptions: {format:'svg', filename, width:1050, height:650, scale:1}
    };
}

function ibisPlotLayout(xlabel, depth, height=420) {
    const low = Math.min(...depth), high = Math.max(...depth);
    const pad = Math.max((high-low)*0.055, 0.1);
    return {
        autosize:true, height,
        margin:{l:76,r:30,t:52,b:70}, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
        font:{family:'Arial, sans-serif',color:'#213555',size:12},
        xaxis:{title:xlabel,showline:true,linecolor:'#BCC8D5',zeroline:false},
        yaxis:{title:'Depth (input units)',range:[high+pad,low-pad],showline:true,linecolor:'#BCC8D5'},
        legend:{orientation:'h',y:1.15,x:0,xanchor:'left'},
        hovermode:'closest', uirevision:'ibis-posterior'
    };
}

function ibisTrace(name, x, depth, color, symbol, plus, minus=plus) {
    return {
        type:'scatter',mode:'markers', name, x, y:depth,
        marker:{size:8,color,symbol,line:{color:'#fff',width:0.8}},
        error_x:{type:'data',symmetric:false,array:plus,arrayminus:minus,visible:true,
                 color,thickness:1.3,width:3},
        customdata:x.map((v,i)=>[minus[i],plus[i]]),
        hovertemplate:`<b>${name}</b><br>Depth: %{y:g}<br>Age/value: %{x:.5g}`+
            '<br>−error: %{customdata[0]:.4g}<br>+error: %{customdata[1]:.4g}<extra></extra>'
    };
}

function drawIBISPlots(sourceRows) {
    if (typeof Plotly === 'undefined') return;
    const rows=[...sourceRows].sort((a,b)=>a.Depth-b.Depth);
    const depth=rows.map(r=>Number(r.Depth));
    const take=k=>rows.map(r=>Number(r[k]));
    const uAge=take('uncorrected_age'), bAge=take('bulk_age'), iAge=take('ibis_age');
    const ageTraces=[
        ibisTrace('Uncorrected (±2σ)',uAge,depth,ibisColors.grey,'circle',take('uncorrected_sigma').map(v=>2*v)),
        ibisTrace('Bulk Earth (±2σ)',bAge,depth,ibisColors.gold,'square',take('bulk_sigma').map(v=>2*v)),
        ibisTrace('IBIS (95% CrI)',iAge,depth,ibisColors.red,'diamond',rows.map(r=>r.ibis_hi95-r.ibis_age),rows.map(r=>r.ibis_age-r.ibis_lo95))
    ];
    Plotly.react('agePlot',ageTraces,ibisPlotLayout('U–Th age (years)',depth),ibisPlotConfig('IBIS_Age_Comparison'));
    const th0=take('th0');
    const thoriumLayout=ibisPlotLayout('Initial (²³⁰Th/²³²Th) activity ratio',depth);
    thoriumLayout.shapes=[{type:'line',x0:0.83,x1:0.83,y0:0,y1:1,yref:'paper',line:{color:ibisColors.gold,dash:'dash',width:1.5}}];
    Plotly.react('thoriumPlot',[
        ibisTrace('IBIS initial Th (95% CrI)',th0,depth,ibisColors.navy,'circle',rows.map(r=>r.th0_hi95-r.th0),rows.map(r=>r.th0-r.th0_lo95))
    ],thoriumLayout,ibisPlotConfig('IBIS_Initial_Thorium'));

    // Standardize across samples, not within posterior samples. All error
    // bars are divided by the same cross-sample standard deviation.
    function standardized(values, lower, upper) {
        const n=values.length;
        const mean=values.reduce((a,b)=>a+b,0)/n;
        const sd=Math.sqrt(values.reduce((s,v)=>s+(v-mean)**2,0)/(n-1));
        if (!(sd>0)) return {z:values.map(()=>0),minus:values.map(()=>0),plus:values.map(()=>0)};
        return {z:values.map(v=>(v-mean)/sd),minus:lower.map(v=>v/sd),plus:upper.map(v=>v/sd)};
    }
    const series=[
        ['²³⁴U/²³⁸U', 'U234_r','U234_r_err',ibisColors.navy],
        ['²³⁰Th/²³⁸U','Th230_r','Th230_r_err',ibisColors.red],
        ['²³²Th/²³⁸U','Th232_r','Th232_r_err',ibisColors.sage]
    ];
    const zTraces=series.map(([name,key,err,color])=>{
        const vals=take(key), errors=take(err), z=standardized(vals,errors,errors);
        return ibisTrace(name,z.z,depth,color,'circle',z.plus,z.minus);
    });
    const zTh=standardized(th0,rows.map(r=>r.th0-r.th0_lo95),rows.map(r=>r.th0_hi95-r.th0));
    zTraces.push(ibisTrace('Inferred initial Th (95% CrI)',zTh.z,depth,ibisColors.gold,'diamond',zTh.plus,zTh.minus));
    const zLayout=ibisPlotLayout('Standardized value (Z-score)',depth);
    zLayout.shapes=[{type:'line',x0:0,x1:0,y0:0,y1:1,yref:'paper',line:{color:'#ADB8C6',dash:'dot'}}];
    Plotly.react('posteriorStandardizedPlot',zTraces,zLayout,ibisPlotConfig('IBIS_Standardized_Results'));
}

function showIBIS(data) {

    resultsRows = data.rows;

    const comparison = $("farajiResult");

    if (data.faraji_plot && data.faraji_summary) {

        $("farajiPlot").src =
            `data:image/png;base64,${data.faraji_plot}`;

        const stats = data.faraji_summary;

        $("farajiSummary").textContent =
            `Matched samples: ${stats.n} · ` +
            `RMSE: ${stats.rmse.toFixed(3)}`;

        comparison.classList.remove("hidden");

    } else {
        comparison.classList.add("hidden");
    }

    populateResultsTable(data.rows);

    $("ibisResult").classList.remove("hidden");
    drawIBISPlots(data.rows);
}

// Save the exact workspace built by the backend.
$("downloadPriorButton").addEventListener("click", () => {
  if (!priorJobId) return;
  window.location.href = `/api/prior/${encodeURIComponent(priorJobId)}/download`;
});

// Import a saved prior. We compare depth/ratios and prior settings first:
// the file must correspond to the measurements currently shown.
$("uploadPriorFile").addEventListener("change", async event => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const current = readData().slice().sort((a,b) => a.Depth-b.Depth);
    const settings = readSettings();
    loading(true, "Validating saved Thoth prior...");
    const response = await fetch("/api/prior/upload", {
      method: "POST", headers: {"Content-Type":"application/octet-stream"}, body:file
    });
    const data = await response.json();
    if (!response.ok) throw new Error(parseError(data));
    const saved = data.saved_rows;
    const numericKeys = ["Depth","U234_r","U234_r_err","Th230_r","Th230_r_err","Th232_r","Th232_r_err","Depth_err"];
    const equal = current.length === saved.length && current.every((row,i) =>
      numericKeys.every(key => {
        const a = row[key] ?? 0;
        const b = saved[i][key] ?? 0;
        return Math.abs(Number(a)-Number(b)) <= 1e-10 * Math.max(1,Math.abs(Number(a)),Math.abs(Number(b)));
      })
    );
    if (!equal) throw new Error("Saved prior measurements do not match the current table. Load the original measurements first.");
    const priorKeys = ["fraction_det","global_mass","use_r02_in_sigma","n_samples_global","n_samples_layer","neighbors"];
    if (!priorKeys.every(key => settings[key] === data.result.settings[key])) {
      throw new Error("Prior-generation settings differ. Set the Thoth settings to the saved values first.");
    }
    // MCMC iteration and chain counts may still be changed independently.
    priorJobId = data.job_id;
    priorSnapshot = snapshot();
    $("ibisResult").classList.add("hidden");
    showPrior(data.result);
    $("downloadPriorButton").disabled = false;
    loading(false, "Saved Thoth prior loaded. Ready for IBIS Part 1.");
    $("priorResult").scrollIntoView({behavior:"smooth"});
  } catch(error) {
    priorJobId = null;
    $("downloadPriorButton").disabled = true;
    loading(false, `Could not load prior: ${error.message}`);
  } finally {
    event.target.value = "";
  }
});

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
    $("downloadPriorButton").disabled = false;
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
    const mcmc = readMCMCSettings();
    const job = await post("/api/part1", {prior_job_id: priorJobId, mcmc});
    const result = await waitForJob(job.job_id);
    if (snapshot() !== priorSnapshot) throw new Error("Inputs changed during the run. Results have not been displayed.");
    showIBIS(result);
    drawMCMCDiagnostics(result, job.job_id);
    loading(false, "IBIS Part 1 complete. Review the age comparisons and download the CSV.");
    $("ibisResult").scrollIntoView({behavior: "smooth", block: "start"});
  } catch (error) {loading(false, `Error: ${error.message}`);}
});

let currentDiagnosticsJobId = null;
let diagnosticTraceRequest = 0;
let currentDiagnosticStats = [];
const diagnosticColors = ["#213555", "#C76D5E", "#729A91", "#D8B26E", "#7C9AB7", "#8C6BB1"];

async function updateDiagnosticTrace() {
  if (!currentDiagnosticsJobId || !$('mcmcDiagnostics').open) return;
  const requestId = ++diagnosticTraceRequest;
  const parameter = $("traceParameter").value;
  const depthIndex = Number($("traceDepth").value || 0);
  const query = new URLSearchParams({parameter, depth_index: String(depthIndex)});
  try {
    const response = await fetch(`/api/diagnostics/${encodeURIComponent(currentDiagnosticsJobId)}/trace?${query}`);
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    // Ignore obsolete responses after a parameter change or new MCMC run.
    if (requestId !== diagnosticTraceRequest || !$('mcmcDiagnostics').open) return;
    const traces = data.chains.map((values, i) => ({
      type: "scattergl", mode: "lines", name: `Chain ${i + 1}`,
      x: data.draw_index, y: values,
      line: {width: 1, color: diagnosticColors[i % diagnosticColors.length]}
    }));
    Plotly.react("logPosteriorPlot", traces, {
      height: 370, autosize: true, margin: {l: 80, r: 20, t: 30, b: 65},
      xaxis: {title: "Retained draw index (after burn-in)"},
      yaxis: {title: parameter === "Log posterior" ? "Log posterior" : parameter},
      hovermode: "x", legend: {orientation: "h"}, uirevision: `${parameter}:${depthIndex}`
    }, ibisPlotConfig(`IBIS_MCMC_${parameter.replaceAll('/', '_')}`));
  } catch (error) {
    if (requestId === diagnosticTraceRequest && $('mcmcDiagnostics').open) {
      $("diagnosticStatus").textContent = `Trace unavailable: ${error.message}`;
    }
  }
}

function drawMCMCDiagnostics(result, jobId) {
  // Start every new run with the diagnostics hidden, and load its plot on demand.
  $('mcmcDiagnostics').open = false;
  diagnosticTraceRequest++;
  currentDiagnosticsJobId = jobId;
  const settings = result.mcmc_settings_used || {};
  $("mcmcSettingsUsed").textContent =
    `MCMC configuration used: ${settings.n_chains ?? "?"} chains · ${settings.iterations ?? "?"} iterations/chain · ${settings.burn_in ?? "?"} burn-in/chain.`;
  const d = result.diagnostics || {};
  $("diagnosticStatus").textContent = d.message || "No sampler diagnostics returned.";
  const depthSelect = $("traceDepth");
  depthSelect.replaceChildren();
  result.rows.forEach((row, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `Depth ${Number(row.Depth).toPrecision(6)} (row ${i+1})`;
    depthSelect.appendChild(opt);
  });
  const summary = $("diagnosticSummary");
  summary.replaceChildren();
  const body = document.querySelector("#rhatDiagnostics tbody");
  body.replaceChildren();
  if (d.status === "available") {
    const format = (x, digits) => x == null ? "N/A" : Number(x).toFixed(digits);
    summary.innerHTML = `<div class="stat"><small>Chains / retained draws</small><strong>${d.n_chains} / ${d.n_draws}</strong></div>` +
      `<div class="stat"><small>Maximum R̂</small><strong>${format(d.max_rhat, 3)}</strong></div>` +
      `<div class="stat"><small>Minimum approx. ESS</small><strong>${format(d.min_ess_bulk, 0)}</strong></div>`;
    const sorted = [...d.stats].sort((a,b) => (b.rhat ?? -1) - (a.rhat ?? -1));
    sorted.forEach(r => {
      const tr = body.insertRow();
      [result.rows[r.depth_index]?.Depth ?? r.depth_index, r.parameter,
       format(r.rhat,3), format(r.ess_bulk,0)].forEach(value => {
        const td = tr.insertCell(); td.textContent = String(value);
      });
    });
  }
  // Plotly must only render after the native <details> panel is expanded.
}
$('mcmcDiagnostics').addEventListener('toggle', () => {
  if ($('mcmcDiagnostics').open) updateDiagnosticTrace();
  else diagnosticTraceRequest++;
});
$("traceParameter").addEventListener("change", updateDiagnosticTrace);
$("traceDepth").addEventListener("change", updateDiagnosticTrace);

function escapeCsv(value) {
  // Preserve numbers and string values; quote commas, double quotes or line breaks.
  const text = value == null ? "" : typeof value === "object" ? JSON.stringify(value) : String(value);
  return /[\n\r,"]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

// Clean names of any optional backend fields. Only [a-z0-9_] is exported.
function asciiCsvHeader(value) {
  return String(value)
    .replace(/σ/g, "sigma").replace(/Σ/g, "sigma")
    .replace(/ρ/g, "rho").replace(/Ρ/g, "rho")
    .replace(/λ/g, "lambda").replace(/Λ/g, "lambda")
    .replace(/μ/g, "mu").replace(/Μ/g, "mu")
    .replace(/±/g, "plus_minus")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase() || "unnamed_column";
}

$("downloadButton").addEventListener("click", () => {
  if (!resultsRows || resultsRows.length === 0) {
    alert("Run IBIS Part 1 before downloading results.");
    return;
  }

  // Core columns exactly mirror the on-screen table. Append optional output
  // columns from the backend so that no posterior information is lost.
  const rows = resultsRows.map(withIbisErrors);
  const standardKeys = RESULT_COLUMNS.map(column => column.key);
  const knownKeys = new Set(standardKeys);
  const extraKeys = [];
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!knownKeys.has(key)) {
        knownKeys.add(key);
        extraKeys.push(key);
      }
    }
  }
  const keys = [...standardKeys, ...extraKeys];
  const coreLabels = Object.fromEntries(RESULT_COLUMNS.map(column => [column.key, column.csv]));
  const additionalLabels = {
    th0_lo68: "initial_th_68pct_lower",
    th0_hi68: "initial_th_68pct_upper",
    th0_lo95: "initial_th_95pct_lower",
    th0_hi95: "initial_th_95pct_upper",
    u0: "initial_u234_u238_ratio",
    u0_sigma: "initial_u234_u238_ratio_1sigma"
  };
  // Ensure unique, ASCII-only headers, including optional extra fields.
  const usedHeaders = new Set();
  const headers = keys.map(key => {
    const base = asciiCsvHeader(coreLabels[key] || additionalLabels[key] || key);
    let label = base;
    let suffix = 2;
    while (usedHeaders.has(label)) label = `${base}_${suffix++}`;
    usedHeaders.add(label);
    return label;
  });

  // Prevent any future regression to superscripts, +/- signs or Greek letters.
  if (headers.some(name => !/^[a-z0-9_]+$/.test(name))) {
    alert("CSV header validation failed: only ASCII names are allowed.");
    return;
  }
  const csv = [
    headers.map(escapeCsv).join(","),
    ...rows.map(row => keys.map(key => escapeCsv(row[key])).join(","))
  ].join("\r\n");

  // Excel on both Mac/Windows recognizes the UTF-8 BOM. Without it, Excel can
  // misinterpret otherwise valid Unicode as garbled or 'hieroglyphic' text.
  const blob = new Blob(["\uFEFF", csv], {type: "text/csv;charset=utf-8;"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "IBIS_Part1_Results_ASCII_v2.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
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
// Deliberately do not call changed(): changing MCMC controls does not invalidate a prior.
document.querySelectorAll("#mcmcStage input").forEach(el => el.addEventListener("input", () => {
  if (!busy) $("jobText").textContent = "MCMC controls updated. Thoth prior remains ready.";
}));
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
