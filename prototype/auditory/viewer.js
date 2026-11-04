"use strict";

const STAGES = [
  ["0", "Difficulty Stage 0 (D0)", "Stage 0 (D0)"],
  ["2", "Difficulty Stage 1 (D2)", "Stage 1 (D2)"],
  ["6", "Difficulty Stage 2 (D6)", "Stage 2 (D6)"],
  ["10", "Difficulty Stage 3 (D10)", "Stage 3 (D10)"]
];
const EXPECTED_FEATURES = new Set([
  "Mental demand", "Time between qualifying grabs",
  "Time to locate target", "Head turning"
]);
const NS = "http://www.w3.org/2000/svg";

function finite(value, optional = false) {
  if (value === null || value === undefined || String(value).trim() === "" ||
      (optional && /^(nan|none|null)$/i.test(String(value).trim()))) {
    if (optional) return null;
    throw new Error("Missing required numerical value");
  }
  if (typeof value === "number") {
    if (Number.isFinite(value)) return value;
    if (optional) return null;
    throw new Error("Nonfinite required number");
  }
  const str = String(value).trim();
  // One decimal separator only. Never interpret commas as thousands separators.
  if (!/^[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:e[+-]?\d+)?$/i.test(str)) {
    throw new Error("Invalid numeric representation: " + str.slice(0, 25));
  }
  const result = Number(str.replace(",", "."));
  if (!Number.isFinite(result)) throw new Error("Nonfinite number");
  return result;
}
function stageCode(label) {
  const found = STAGES.find(s => s[1] === label);
  if (!found) throw new Error("Unknown difficulty stage");
  return found[0];
}
function displayStage(code) {
  const found = STAGES.find(s => s[0] === String(code));
  return found ? found[2] : "Unavailable";
}
function validateDataset(data) {
  if (!Array.isArray(data) || data.length !== 8) {
    throw new Error("Expected exactly eight Auditory observations");
  }
  const seen = new Set(), participantNames = new Map();
  const cases = new Set();
  for (const record of data) {
    if (!record || !["A", "B"].includes(record.Case)) {
      throw new Error("Expected original A/B case keys");
    }
    if (typeof record.ParticipantId !== "string" ||
        !/^[A-Za-z0-9_-]{2,25}$/.test(record.ParticipantId)) {
      throw new Error("ParticipantId missing; prepare local viewer JSON using README command");
    }
    const code = stageCode(record.DifficultyStage);
    const key = record.Case + "|" + code;
    if (seen.has(key)) throw new Error("Repeated participant-stage observation");
    seen.add(key);
    cases.add(record.Case);
    if (participantNames.has(record.Case) &&
        participantNames.get(record.Case) !== record.ParticipantId) {
      throw new Error("Inconsistent participant ID for one case");
    }
    participantNames.set(record.Case, record.ParticipantId);

    const reg = record.Regression;
    if (!reg || !Array.isArray(reg.Measurements) ||
        reg.Measurements.length !== 4) {
      throw new Error("Each regression record needs four measurements");
    }
    const reference = finite(reg.ReferencePrediction);
    const prediction = finite(reg.Prediction);
    const names = new Set();
    let total = reference;
    for (const feature of reg.Measurements) {
      if (!feature || !EXPECTED_FEATURES.has(feature.Name) ||
          names.has(feature.Name) || typeof feature.Unit !== "string") {
        throw new Error("Unexpected or duplicate Auditory feature");
      }
      names.add(feature.Name);
      finite(feature.Observed, true);
      total += finite(feature.ShapContribution);
      const m = feature.Membership || {};
      ["Low", "Medium", "High"].forEach(k => {
        const v = finite(m[k], true);
        if (v !== null && (v < -1e-12 || v > 1 + 1e-12)) {
          throw new Error("Membership outside [0, 1]");
        }
      });
    }
    if (Math.abs(total - prediction) > 1e-8) {
      throw new Error("SHAP reconstruction failed for " + key);
    }

    const fuzzy = record.Fuzzy;
    if (!fuzzy || !Array.isArray(fuzzy.Recommendations) ||
        fuzzy.Recommendations.length !== 2 ||
        !Array.isArray(fuzzy.Rules) || fuzzy.Rules.length !== 10) {
      throw new Error("Expected two recommendations and ten rules");
    }
    const thresholds = fuzzy.Recommendations.map(r => finite(r.Threshold))
      .sort((a, b) => a - b);
    if (Math.abs(thresholds[0] - 0.1) > 1e-12 ||
        Math.abs(thresholds[1] - 0.2) > 1e-12) {
      throw new Error("Unexpected fuzzy thresholds");
    }
    for (const rec of fuzzy.Recommendations) {
      if (!["Increase", "Decrease", "Maintain", "Review", "Unavailable"]
        .includes(rec.FinalAction)) {
        throw new Error("Unknown fuzzy recommendation action");
      }
      if (!["issued", "boundary_review", "data_unavailable"]
        .includes(rec.Status)) {
        throw new Error("Unknown recommendation status");
      }
      if (rec.FinalAction === "Review" && rec.Status !== "boundary_review" ||
          rec.FinalAction === "Unavailable" && rec.Status !== "data_unavailable" ||
          !["Review", "Unavailable"].includes(rec.FinalAction) && rec.Status !== "issued") {
        throw new Error("Inconsistent fuzzy action and status");
      }
      if (rec.ProposedLevel !== null && rec.ProposedLevel !== undefined &&
          String(rec.ProposedLevel).trim() !== "" &&
          !STAGES.some(s => s[0] === String(rec.ProposedLevel))) {
        throw new Error("Unsupported proposed stage code");
      }
      finite(rec.IncreaseSupport, true);
      finite(rec.DecreaseSupport, true);
    }
    const rules = new Set();
    for (const rule of fuzzy.Rules) {
      if (!/^[DI][1-5]$/.test(rule.Rule) || rules.has(rule.Rule)) {
        throw new Error("Duplicate or unknown fuzzy rule");
      }
      rules.add(rule.Rule);
      finite(rule.Activation, true);
    }
  }
  if (cases.size !== 2 ||
      participantNames.get("A") === participantNames.get("B")) {
    throw new Error("Expected two distinct participants");
  }
  return data;
}
function svgNode(name, attributes, label) {
  const node = document.createElementNS(NS, name);
  Object.entries(attributes || {}).forEach(([key, val]) =>
    node.setAttribute(key, String(val)));
  if (label !== undefined) node.textContent = String(label);
  return node;
}
function format(value, digits = 3) {
  if (value === null || value === undefined) return "Unavailable";
  const n = finite(value, true);
  if (n === null) return "Unavailable";
  return n.toLocaleString("en-GB", {maximumFractionDigits: digits});
}
function sign(value) {
  return (value >= 0 ? "+" : "−") + format(Math.abs(value));
}
function makeWaterfall(regression) {
  const baseline = finite(regression.ReferencePrediction);
  const prediction = finite(regression.Prediction);
  const values = [baseline], features = regression.Measurements;
  features.forEach(f => values.push(values[values.length - 1] + finite(f.ShapContribution)));
  const min = Math.min(...values), max = Math.max(...values);
  const pad = Math.max((max - min) * .16, .15);
  const lo = min - pad, hi = max + pad;
  const x = v => 252 + (v - lo) / (hi - lo) * 455;
  const svg = svgNode("svg", {
    viewBox: "0 0 860 392", role: "img",
    "aria-label": "SHAP waterfall. Baseline " + format(baseline) +
      ", final estimate " + format(prediction) + ". " +
      features.map(f => f.Name + " " + sign(finite(f.ShapContribution))).join("; ")
  });
  const title = svgNode("title", {}, "Auditory stage-specific SHAP waterfall");
  svg.append(title);
  const top = 68, step = 56, axisY = 342;
  for (let tick = 0; tick <= 4; tick++) {
    const val = lo + (hi - lo) * tick / 4, xpos = x(val);
    svg.append(svgNode("line", {x1:xpos,y1:38,x2:xpos,y2:axisY,stroke:"currentColor","stroke-opacity":".09"}));
    svg.append(svgNode("text",{x:xpos,y:axisY+19,"text-anchor":"middle",class:"axis-label"},format(val,2)));
  }
  svg.append(svgNode("text",{x:14,y:top+5,class:"row-name"},"Stage-specific baseline"));
  svg.append(svgNode("circle",{cx:x(baseline),cy:top,r:6,fill:"#64748b"}));
  svg.append(svgNode("text",{x:790,y:top+5,"text-anchor":"end",class:"row-value"},format(baseline)));
  features.forEach((f,i) => {
    const previous = values[i], next = values[i+1], contribution = finite(f.ShapContribution);
    const y = top + (i+1)*step;
    const prevX=x(previous), nextX=x(next);
    const priorY = top + i*step;
    svg.append(svgNode("line",{x1:prevX,y1:priorY+8,x2:prevX,y2:y-12,stroke:"#94a3b8","stroke-dasharray":"4 4"}));
    svg.append(svgNode("text",{x:14,y:y+5,class:"row-name"},f.Name));
    const bar = svgNode("rect",{
      x:Math.min(prevX,nextX), y:y-12,
      width:Math.max(Math.abs(nextX-prevX),2), height:24, rx:3,
      fill: contribution >= 0 ? "#2a719f" : "#b85e46"
    });
    bar.append(svgNode("title",{},f.Name + ": " + sign(contribution) +
      "; observed " + format(finite(f.Observed,true)) + " " + f.Unit));
    svg.append(bar);
    svg.append(svgNode("text",{x:790,y:y+5,"text-anchor":"end",class:"row-value"},sign(contribution)));
  });
  const finalY = top + 5*step;
  svg.append(svgNode("line",{x1:x(prediction),y1:top+4*step+12,x2:x(prediction),y2:finalY-8,stroke:"#94a3b8","stroke-dasharray":"4 4"}));
  svg.append(svgNode("text",{x:14,y:finalY+5,class:"row-name final-name"},"Estimated difficulty"));
  svg.append(svgNode("circle",{cx:x(prediction),cy:finalY,r:6,fill:"#334155"}));
  svg.append(svgNode("text",{x:790,y:finalY+5,"text-anchor":"end",class:"row-value final-name"},format(prediction)));
  svg.append(svgNode("text",{x:490,y:axisY+45,"text-anchor":"middle",class:"axis-label"},"Difficulty prediction scale"));
  return svg;
}
function cell(value) {
  const td = document.createElement("td");
  td.textContent = String(value);
  return td;
}
function row(cells) {
  const tr = document.createElement("tr");
  cells.forEach(v => tr.append(cell(v)));
  return tr;
}
function renderRegression(record) {
  const reg = record.Regression;
  document.getElementById("prediction").textContent = format(reg.Prediction);
  const target = document.getElementById("waterfall");
  target.replaceChildren(makeWaterfall(reg));
  const tbody = document.getElementById("measurement-rows");
  tbody.replaceChildren(...reg.Measurements.map(m => row([
    m.Name, format(finite(m.Observed,true)) + " " + m.Unit,
    sign(finite(m.ShapContribution))
  ])));
}
function renderFuzzy(record) {
  const recs = [...record.Fuzzy.Recommendations].sort((a,b) =>
    finite(a.Threshold) - finite(b.Threshold));
  const agree = recs[0].FinalAction === recs[1].FinalAction;
  document.getElementById("threshold-message").textContent = agree ?
    "Both candidate thresholds produce " + recs[0].FinalAction +
    ". Agreement is not confidence in clinical correctness." :
    "The two candidate thresholds produce different outcomes. No preferred threshold has been selected.";
  const holder = document.getElementById("threshold-cards");
  holder.replaceChildren(...recs.map(rec => {
    const card = document.createElement("article");
    card.className = "threshold-card";
    const heading = document.createElement("h3");
    heading.textContent = "Rule threshold " + format(finite(rec.Threshold),2);
    const action = document.createElement("p");
    action.className = "action " + (rec.Status === "issued" ? "" : "caution");
    action.textContent = rec.FinalAction;
    const detail = document.createElement("p");
    detail.textContent = "Next stage: " + displayStage(rec.ProposedLevel) +
      " · Status: " + rec.Status.replaceAll("_"," ");
    const support = document.createElement("p");
    support.textContent = "Increase support: " + format(finite(rec.IncreaseSupport,true)) +
      " · Decrease support: " + format(finite(rec.DecreaseSupport,true));
    const strongest = document.createElement("p");
    strongest.textContent = "Strongest rule(s): " + (rec.StrongestRules || "None");
    card.append(heading,action,detail,support,strongest);
    return card;
  }));
  const notes = [];
  for (const rec of recs) {
    if (rec.ReviewFlags) notes.push("Threshold " + rec.Threshold + ": " + rec.ReviewFlags);
    if (rec.DiagnosticFlags) notes.push("Threshold " + rec.Threshold + ": " + rec.DiagnosticFlags);
    if (rec.Status !== "issued") notes.push("Threshold " + rec.Threshold + ": " + rec.Status);
  }
  for (const m of record.Regression.Measurements) {
    if (String(m.OutOfCalibration).toLowerCase() === "true") {
      notes.push(m.Name + ": outside development calibration range");
    }
  }
  const notices = document.getElementById("warnings");
  notices.replaceChildren();
  if (notes.length) {
    const h = document.createElement("h3"); h.textContent = "Warnings and review notes";
    const list = document.createElement("ul");
    for(const note of notes){const li = document.createElement("li");li.textContent=note;list.append(li);}
    notices.append(h,list);
    notices.className = "warning-list";
  } else {
    notices.textContent = "No additional warnings are recorded for this observation.";
    notices.className = "hint";
  }
  document.getElementById("rule-rows").replaceChildren(
    ...[...record.Fuzzy.Rules].sort((a,b)=>a.Rule.localeCompare(b.Rule)).map(r =>
      row([r.Rule + " — " + r.RuleName,r.Action,format(finite(r.Activation,true)),r.Status]))
  );
  document.getElementById("membership-rows").replaceChildren(
    ...record.Regression.Measurements.map(m => row([
      m.Name, format(finite(m.Membership.Low,true)),format(finite(m.Membership.Medium,true)),
      format(finite(m.Membership.High,true)),
      String(m.OutOfCalibration).toLowerCase()==="true" ? "Outside range" : "Within range"
    ]))
  );
}
function installApp() {
  let records = [], activeTab = "regression";
  const file = document.getElementById("data-file");
  const participant = document.getElementById("participant");
  const stage = document.getElementById("stage");
  const status = document.getElementById("load-status");
  const workspace = document.getElementById("workspace");
  const clearDecision = () => {
    document.getElementById("next-stage").value = "";
    document.getElementById("rationale").value = "";
  };
  function render() {
    const selected = records.find(r => r.ParticipantId === participant.value &&
      stageCode(r.DifficultyStage) === stage.value);
    if (!selected) throw new Error("No matching participant-stage observation");
    document.getElementById("selection-context").textContent =
      "Participant " + selected.ParticipantId + " · Auditory · " +
      displayStage(stage.value) + " completed";
    renderRegression(selected);
    renderFuzzy(selected);
  }
  function setTab(tab) {
    activeTab = tab;
    for (const name of ["regression","fuzzy"]) {
      const selected = activeTab === name;
      document.getElementById(name+"-view").hidden = !selected;
      document.getElementById("tab-"+name).setAttribute("aria-selected",String(selected));
    }
  }
  file.addEventListener("change", async () => {
    workspace.hidden = true;
    records = [];
    try {
      if (!file.files || file.files.length !== 1) throw new Error("Select one JSON file");
      if (file.files[0].size > 3_000_000) throw new Error("File exceeds the 3 MB local limit");
      const parsed = JSON.parse(await file.files[0].text());
      records = validateDataset(parsed);
      const labels = [...new Set(records.map(r => r.ParticipantId))].sort();
      participant.replaceChildren(...labels.map(label=>{
        const option=document.createElement("option");option.value=label;option.textContent=label;
        return option;
      }));
      stage.value = "0";
      setTab("regression");
      clearDecision();
      render();
      workspace.hidden = false;
      status.textContent = "Data validated: eight observations; two participants; four stages each.";
    } catch (error) {
      records = [];status.textContent = "Cannot display file: " + error.message;
    }
  });
  participant.addEventListener("change",()=>{clearDecision();render();});
  stage.addEventListener("change",()=>{clearDecision();render();});
  document.getElementById("tab-regression").addEventListener("click",()=>setTab("regression"));
  document.getElementById("tab-fuzzy").addEventListener("click",()=>setTab("fuzzy"));
}
if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", installApp);
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = {finite,stageCode,displayStage,validateDataset};
}
