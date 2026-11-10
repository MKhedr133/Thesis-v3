"use strict";

const STAGES = [
  ["0", "Difficulty Stage 0 (D0)", "Level 0"],
  ["2", "Difficulty Stage 1 (D2)", "Level 1"],
  ["6", "Difficulty Stage 2 (D6)", "Level 2"],
  ["10", "Difficulty Stage 3 (D10)", "Level 3"]
];
const EXPECTED_FEATURES = new Set([
  "Mental demand", "Time between qualifying grabs",
  "Time to locate target", "Head turning"
]);
const FEATURE_LABELS = Object.freeze({
  "Mental demand": "Subjective mental demand rating",
  "Time between qualifying grabs": "Time interval between grabs",
  "Time to locate target": "Duration to locate a target product",
  "Head turning": "Head rotation while locating a product"
});
// Short display labels apply to SHAP only; A/B source identities and C/D labels stay frozen.
const SHAP_LABELS = Object.freeze({
  "Mental demand": "Mental demand rating",
  "Time between qualifying grabs": "Time between grabs",
  "Time to locate target": "Time to locate target",
  "Head turning": "Head rotation while searching"
});
// Exact Auditory antecedents from MOD-18's frozen RULES registry.
// Low/Medium/High match the memberships already stored in the local JSON.
const AUDITORY_RULES = Object.freeze({
  D1: [["Mental demand","High"],["Time between qualifying grabs","High"]],
  D2: [["Mental demand","High"],["Time to locate target","High"]],
  D3: [["Time between qualifying grabs","High"],["Time to locate target","High"]],
  D4: [["Mental demand","High"],["Head turning","High"]],
  D5: [["Time between qualifying grabs","High"],["Head turning","High"]],
  I1: [["Mental demand","Low"],["Time between qualifying grabs","Low"],["Time to locate target","Low"]],
  I2: [["Mental demand","Low"],["Time between qualifying grabs","Low"],["Head turning","Low"]],
  I3: [["Mental demand","Low"],["Time to locate target","Low"],["Head turning","Low"]],
  I4: [["Mental demand","Low"],["Time to locate target","Low"],["Time between qualifying grabs","Medium"]],
  I5: [["Mental demand","Low"],["Time between qualifying grabs","Low"],["Head turning","Medium"]]
});
const RULE_TIE_TOLERANCE = 1e-12;
function featureLabel(name) {
  if (!Object.hasOwn(FEATURE_LABELS,name)) throw new Error("Unknown Auditory feature label");
  return FEATURE_LABELS[name];
}
function ruleSentence(id) {
  const antecedents=AUDITORY_RULES[id];
  if (!antecedents) throw new Error("Unknown Auditory rule");
  const text=antecedents.map(([name,category]) =>
    featureLabel(name).toLowerCase()+" is "+(category==="Medium"?"intermediate":category.toLowerCase())).join(" AND ");
  return "If "+text+", then "+(id.startsWith("I")?"increase":"decrease")+" difficulty.";
}
function ruleSummary(record) {
  const recs=record.Fuzzy.Recommendations;
  if (recs.some(rec=>rec.Status==="data_unavailable")) {
    return {suggestion:"Unavailable",main:[],other:[],reason:"The required rule inputs are unavailable."};
  }
  const active=record.Fuzzy.Rules.map(rule=>({
    ...rule, score:finite(rule.Activation,true)
  })).filter(rule=>rule.score!==null && rule.score>0)
    .sort((a,b)=>b.score-a.score || a.Rule.localeCompare(b.Rule));
  if (!active.length) {
    return {suggestion:"Maintain",main:[],other:[],reason:"No Increase or Decrease rule(s) were active."};
  }
  const strongest=active[0].score;
  const main=active.filter(rule=>Math.abs(rule.score-strongest)<=RULE_TIE_TOLERANCE);
  const other=active.filter(rule=>Math.abs(rule.score-strongest)>RULE_TIE_TOLERANCE);
  const actions=new Set(main.map(rule=>rule.Action));
  let suggestion;
  let reason="";
  if (actions.size>1) {
    suggestion="Mixed indications";
    reason="The strongest Increase and Decrease rules have equal support. No single direction is selected.";
  } else {
    const action=main[0].Action;
    const code=stageCode(record.DifficultyStage);
    if ((action==="Increase"&&code==="10")||(action==="Decrease"&&code==="0")) {
      suggestion="Review";
      reason=action==="Increase"?
        "The suggested action was Increase, which exceeds the available difficulty levels (above Level 3).":
        "The suggested action was Decrease, which exceeds the available difficulty levels (below Level 0).";
    } else suggestion=action;
  }
  return {suggestion,main,other,reason};
}
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
function proposedStageCode(value) {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  let numeric;
  try {
    numeric = finite(value);
  } catch {
    throw new Error("Unsupported proposed stage code");
  }
  // MOD-18 exports integer stage codes as decimal strings (e.g. "6.0").
  // Accept equivalent numeric representations, never round an invalid code.
  const found = STAGES.find(s => Number(s[0]) === numeric);
  if (!Number.isInteger(numeric) || !found) {
    throw new Error("Unsupported proposed stage code");
  }
  return found[0];
}
function displayStage(code) {
  const canonical = proposedStageCode(code);
  if (canonical === null) return "Not proposed";
  return STAGES.find(s => s[0] === canonical)[2];
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
      proposedStageCode(rec.ProposedLevel);
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
function shapFeatureLabel(name) {
  if (!Object.hasOwn(SHAP_LABELS,name)) throw new Error("Unknown Auditory SHAP feature");
  return SHAP_LABELS[name];
}
function shapMeasuredValue(feature) {
  const unit = feature.Name === "Mental demand" ? "/10" :
    feature.Name === "Head turning" ? "°" : "s";
  return format(finite(feature.Observed,true)) + " " + unit;
}
function makeWaterfall(regression, onFeatureFocus) {
  const baseline = finite(regression.ReferencePrediction);
  const prediction = finite(regression.Prediction);
  const values = [baseline], features = regression.Measurements;
  features.forEach(f => values.push(values[values.length - 1] + finite(f.ShapContribution)));
  const min = Math.min(...values), max = Math.max(...values);
  const pad = Math.max((max - min) * .16, .15);
  const lo = min - pad, hi = max + pad;
  const x = v => 234 + (v - lo) / (hi - lo) * 470;
  const svg = svgNode("svg", {
    viewBox: "0 0 860 392", role: "group",
    "aria-label": "SHAP waterfall. Baseline " + format(baseline) +
      ", final prediction " + format(prediction) + ". " +
      features.map(f => shapFeatureLabel(f.Name) + " " + sign(finite(f.ShapContribution))).join("; ")
  });
  const title = svgNode("title", {}, "SHAP feature contributions");
  svg.append(title);
  const top = 68, step = 56, axisY = 342;
  for (let tick = 0; tick <= 4; tick++) {
    const val = lo + (hi - lo) * tick / 4, xpos = x(val);
    svg.append(svgNode("line", {x1:xpos,y1:38,x2:xpos,y2:axisY,stroke:"currentColor","stroke-opacity":".09"}));
    svg.append(svgNode("text",{x:xpos,y:axisY+19,"text-anchor":"middle",class:"axis-label"},format(val,2)));
  }
  svg.append(svgNode("text",{x:14,y:top-5,class:"row-name","font-weight":"700"},"Population difficulty level"));
  svg.append(svgNode("text",{x:14,y:top+11,class:"row-name","font-weight":"700"},"prediction reference"));
  svg.append(svgNode("circle",{cx:x(baseline),cy:top,r:6,fill:"#64748b"}));
  svg.append(svgNode("text",{x:790,y:top+5,"text-anchor":"end",class:"row-value"},format(baseline)));
  features.forEach((f,i) => {
    const previous = values[i], next = values[i+1], contribution = finite(f.ShapContribution);
    const y = top + (i+1)*step;
    const prevX=x(previous), nextX=x(next);
    const priorY = top + i*step;
    svg.append(svgNode("line",{x1:prevX,y1:priorY+8,x2:prevX,y2:y-12,stroke:"#94a3b8","stroke-dasharray":"4 4"}));
    svg.append(svgNode("text",{x:14,y:y+5,class:"row-name"},shapFeatureLabel(f.Name)));
    const bar = svgNode("rect",{
      x:Math.min(prevX,nextX), y:y-12,
      width:Math.max(Math.abs(nextX-prevX),2), height:24, rx:3,
      fill: contribution >= 0 ? "#25835c" : "#c04c46",
      "data-shap-feature": f.Name
    });
    bar.append(svgNode("title",{},shapFeatureLabel(f.Name) + ": " + sign(contribution) +
      "; observed " + shapMeasuredValue(f)));
    if (typeof onFeatureFocus === "function") {
      bar.setAttribute("tabindex", "0");
      bar.setAttribute("role", "button");
      bar.setAttribute("aria-label", "Compare " + shapFeatureLabel(f.Name) + " with other observations");
      bar.addEventListener("click", () => onFeatureFocus(f.Name));
      bar.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onFeatureFocus(f.Name);
        }
      });
    }
    svg.append(bar);
    svg.append(svgNode("text",{x:790,y:y+5,"text-anchor":"end",class:"row-value"},sign(contribution)));
  });
  const finalY = top + 5*step;
  svg.append(svgNode("line",{x1:x(prediction),y1:top+4*step+12,x2:x(prediction),y2:finalY-8,stroke:"#94a3b8","stroke-dasharray":"4 4"}));
  svg.append(svgNode("text",{x:14,y:finalY+5,class:"row-name final-name"},"Predicted difficulty level"));
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
function validateShapComparison(data, records) {
  if (data === undefined || data === null) return null; // Backward-compatible old local JSON.
  const stages = STAGES.map(item => item[0]);
  const expected = [...EXPECTED_FEATURES];
  if (data.schema !== "auditory-shap-comparison-v1" ||
      data.population !== "development" || data.model !== "auditory" ||
      data.developmentParticipants !== 26 || data.developmentObservations !== 104 ||
      !Array.isArray(data.features) || data.features.join("|") !== expected.join("|") ||
      !Array.isArray(data.coefficients) || data.coefficients.length !== expected.length ||
      !data.byReferenceStage ||
      Object.keys(data.byReferenceStage).sort().join("|") !== [...stages].sort().join("|")) {
    throw new Error("SHAP comparison provenance or feature identities are invalid");
  }
  const coefficients = data.coefficients.map(n => finite(n));
  const intercept = finite(data.intercept);
  let sharedObservations = null;
  for (const stage of stages) {
    const group = data.byReferenceStage[stage];
    if (!group || !Array.isArray(group.meanFeatures) ||
        group.meanFeatures.length !== expected.length ||
        !Array.isArray(group.observations) || group.observations.length !== 104) {
      throw new Error("Missing development SHAP comparison observations");
    }
    const baseline = finite(group.referencePrediction);
    const means = group.meanFeatures.map(n => finite(n));
    const expectedBaseline = intercept + means.reduce((a, v, j) =>
      a + v * coefficients[j], 0);
    if (Math.abs(baseline - expectedBaseline) > 1e-8) {
      throw new Error("SHAP comparison reference disagrees with frozen coefficients");
    }
    const counts = new Map(stages.map(s => [s, 0]));
    for (let i = 0; i < group.observations.length; i++) {
      const point = group.observations[i];
      if (!point || !counts.has(point.stage) ||
          !Array.isArray(point.values) || point.values.length !== expected.length ||
          !Array.isArray(point.contributions) ||
          point.contributions.length !== expected.length ||
          Object.keys(point).some(k => !["stage", "values", "contributions"].includes(k))) {
        throw new Error("Invalid or identifying SHAP development observation");
      }
      counts.set(point.stage, counts.get(point.stage) + 1);
      const values = point.values.map(n => finite(n));
      const contribution = point.contributions.map(n => finite(n));
      for (let j = 0; j < expected.length; j++) {
        if (Math.abs(contribution[j] - coefficients[j] * (values[j] - means[j])) > 1e-8) {
          throw new Error("SHAP comparison contributions use the wrong reference");
        }
      }
      const predicted = intercept + values.reduce((a, v, j) =>
        a + coefficients[j] * v, 0);
      if (Math.abs(baseline + contribution.reduce((a, v) => a + v, 0) - predicted) > 1e-8) {
        throw new Error("SHAP comparison does not reconstruct the frozen prediction");
      }
      if (sharedObservations && (point.stage !== sharedObservations[i].stage ||
          values.some((v, j) => Math.abs(v - sharedObservations[i].values[j]) > 1e-10))) {
        throw new Error("Comparison observations changed across reference selections");
      }
    }
    if ([...counts.values()].some(n => n !== 26)) {
      throw new Error("Expected 26 development trials per assigned level");
    }
    for (let j = 0; j < means.length; j++) {
      const actual = group.observations.filter(p => p.stage === stage)
        .reduce((a, p) => a + finite(p.values[j]), 0) / 26;
      if (Math.abs(means[j] - actual) > 1e-9) {
        throw new Error("SHAP reference feature mean differs from its stage population");
      }
    }
    records.filter(record => stageCode(record.DifficultyStage) === stage)
      .forEach(record => {
        if (Math.abs(finite(record.Regression.ReferencePrediction) - baseline) > 1e-7) {
          throw new Error("SHAP comparison does not match the original waterfall reference");
        }
        expected.forEach((name, j) => {
          const measure = record.Regression.Measurements.find(m => m.Name === name);
          if (!measure ||
              Math.abs(finite(measure.ShapContribution) -
                coefficients[j] * (finite(measure.Observed) - means[j])) > 1e-7) {
            throw new Error("Selected trial SHAP contribution mismatches comparison");
          }
        });
      });
    if (!sharedObservations) sharedObservations = group.observations;
  }
  return data;
}
function shapComparisonPoints(data, referenceStage, filter) {
  if (!data) return [];
  const points = data.byReferenceStage[referenceStage].observations;
  if (filter === "current") return points.filter(item => item.stage === referenceStage);
  if (filter === "all") return points;
  if (!STAGES.some(item => item[0] === filter)) {
    throw new Error("Unknown SHAP comparison stage filter");
  }
  return points.filter(item => item.stage === filter);
}
function swarmOffsets(horizontalPositions) {
  // Stable, deterministic 1D collision-avoiding stack; vertical offset has no data meaning.
  const offsets = Array(horizontalPositions.length).fill(0);
  const placed = [];
  const candidates = [0,8,-8,16,-16,24,-24];
  horizontalPositions.map((x,index)=>({x,index}))
    .sort((a,b)=>a.x-b.x||a.index-b.index)
    .forEach(item=>{
      const offset = candidates.find(dy => placed.every(previous =>
        Math.abs(previous.x-item.x) >= 9 || Math.abs(previous.dy-dy) >= 8));
      offsets[item.index] = offset === undefined ? 0 : offset;
      placed.push({x:item.x,dy:offsets[item.index]});
    });
  return offsets;
}
function makeShapBeeswarm(regression, data, referenceStage, filter, focused) {
  const group = data.byReferenceStage[referenceStage];
  const all = group.observations;
  const points = shapComparisonPoints(data, referenceStage, filter);
  const features = regression.Measurements;
  const left = 242, right = 865, top = 55, spacing = 55, axisY = 273;
  const extent = [
    0,
    ...all.flatMap(p => p.contributions),
    ...features.map(m => finite(m.ShapContribution))
  ];
  const min = Math.min(...extent), max = Math.max(...extent);
  const pad = Math.max((max - min) * .09, .025);
  const lo = min - pad, hi = max + pad;
  const x = v => left + (v - lo) / (hi - lo) * (right - left);
  const plot = svgNode("svg", {
    viewBox: "0 0 900 332", role: "img",
    "aria-label": "SHAP contributions of " + points.length +
      " other development observations using the " + displayStage(referenceStage) +
      " reference. The selected observation is outlined."
  });
  plot.append(svgNode("title", {}, "Compare feature contributions with other observations"));
  // Round-number ticks always include the meaningful zero-contribution reference.
  const rawStep = (hi - lo) / 5;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const step = [1,2,2.5,5,10].map(n => n * magnitude)
    .reduce((best,n) => Math.abs(n-rawStep)<Math.abs(best-rawStep)?n:best);
  const ticks = [];
  const firstTick = Math.ceil(lo/step - 1e-10);
  const lastTick = Math.floor(hi/step + 1e-10);
  for (let i = firstTick; i <= lastTick; i++) ticks.push(i * step);
  if (!ticks.some(value => Math.abs(value) < step * 1e-8)) ticks.push(0);
  ticks.sort((a,b) => a-b);
  for (const value of ticks) {
    const px = x(value);
    plot.append(svgNode("line", {
      x1:px,y1:38,x2:px,y2:axisY,stroke:"#94a3b8",
      "stroke-opacity":.18,"stroke-dasharray":"3 5"
    }));
    plot.append(svgNode("text",{x:px,y:axisY+20,"text-anchor":"middle",
      class:"axis-label"},Math.abs(value)<step*1e-8?"0":format(value,2)));
  }
  plot.append(svgNode("line",{x1:x(0),y1:38,x2:x(0),y2:axisY,
    stroke:"#5f7689","stroke-width":1.8,"stroke-opacity":.7}));
  features.forEach((feature,j) => {
    const featureIndex = data.features.indexOf(feature.Name);
    const y = top + j * spacing;
    if (focused === feature.Name) {
      plot.append(svgNode("rect",{x:2,y:y-23,width:896,height:47,rx:7,
        class:"shap-row-selected"}));
    }
    plot.append(svgNode("text",{x:14,y:y+4,class:"row-name"},shapFeatureLabel(feature.Name)));
    const observedValues = all.map(p => finite(p.values[featureIndex]));
    const low = Math.min(...observedValues), high = Math.max(...observedValues);
    const offsets = swarmOffsets(points.map(p => x(finite(p.contributions[featureIndex]))));
    points.forEach((point,i) => {
      const value = finite(point.values[featureIndex]);
      const t = high > low ? (value - low) / (high - low) : .5;
      const hue = 145 * (1-t);
      const dot = svgNode("circle", {
        cx:x(finite(point.contributions[featureIndex])),
        cy:y+offsets[i],
        r:3.7,fill:"hsl("+hue.toFixed(1)+",57%,49%)",
        opacity:.67
      });
      dot.append(svgNode("title",{},displayStage(point.stage) + "; " +
        shapFeatureLabel(feature.Name) + ": observed " + format(value,3) +
        " " + feature.Unit + "; SHAP " + sign(finite(point.contributions[featureIndex]))));
      plot.append(dot);
    });
    const selectedContribution = finite(feature.ShapContribution);
    const selected = svgNode("circle", {
      cx:x(selectedContribution), cy:y,r:7.5,
      fill:"#fff",stroke:"#7052b4","stroke-width":3
    });
    selected.append(svgNode("title",{}, "Selected observation; " +
      shapFeatureLabel(feature.Name) + ": observed " +
      format(finite(feature.Observed),3) + " " + feature.Unit +
      "; SHAP " + sign(selectedContribution)));
    plot.append(selected);
    plot.append(svgNode("circle",{cx:x(selectedContribution),cy:y,r:2.6,
      fill:"#7052b4","pointer-events":"none"}));
  });
  plot.append(svgNode("text",{x:554,y:axisY+42,"text-anchor":"middle",
    class:"axis-label"},"Feature contribution"));
  return plot;
}
function renderRegression(record, comparison, selectedStage, focused, onFeatureFocus) {
  const reg = record.Regression;
  document.getElementById("prediction").textContent = format(reg.Prediction);
  document.getElementById("waterfall")
    .replaceChildren(makeWaterfall(reg, onFeatureFocus));
  const tbody = document.getElementById("measurement-rows");
  tbody.replaceChildren(...reg.Measurements.map(m => row([
    shapFeatureLabel(m.Name),shapMeasuredValue(m)
  ])));
  const svgTarget = document.getElementById("shap-beeswarm");
  const warning = document.getElementById("shap-comparison-unavailable");
  const filter = document.getElementById("shap-stage-filter").value;
  if (!comparison) {
    svgTarget.replaceChildren();
    warning.hidden = false;
    warning.textContent = "Development comparison is unavailable in this file. Regenerate the local unified A–D JSON with the updated exporter.";
    return;
  }
  const points = shapComparisonPoints(comparison, selectedStage, filter);
  warning.hidden = true;
  svgTarget.replaceChildren(makeShapBeeswarm(reg,comparison,selectedStage,filter,focused));
}
function ruleCard(rule) {
  const node=document.createElement("div");
  node.className="rule-item";
  const identifier=document.createElement("span");
  identifier.className="rule-id";
  identifier.textContent=rule.Rule;
  const sentence=document.createElement("p");
  sentence.textContent=ruleSentence(rule.Rule);
  node.append(identifier,sentence);
  return node;
}
function renderFuzzy(record) {
  const summary=ruleSummary(record);
  document.getElementById("rule-indication").textContent=summary.suggestion;
  const context=document.getElementById("rule-context");
  context.textContent=summary.reason;
  context.hidden=!summary.reason;

  const main=document.getElementById("main-rules");
  main.replaceChildren(...summary.main.map(ruleCard));
  document.getElementById("active-rule-section").hidden=!summary.main.length;

  // Keep all rule activations and the original policy outcomes in the input
  // JSON unchanged. This tab displays a threshold-free rule-level indication,
  // not a replacement for MOD-18's two-threshold candidate decision policy.
  const rules=[...record.Fuzzy.Rules].sort((a,b)=>a.Rule.localeCompare(b.Rule));
  document.getElementById("rule-rows").replaceChildren(...rules.map(rule=>
    row([
      rule.Rule+": "+ruleSentence(rule.Rule),
      rule.Action,
      format(finite(rule.Activation,true))
    ])
  ));
  document.getElementById("membership-rows").replaceChildren(
    ...record.Regression.Measurements.map(m=>row([
      featureLabel(m.Name),
      format(finite(m.Membership.Low,true)),
      format(finite(m.Membership.Medium,true)),
      format(finite(m.Membership.High,true))
    ]))
  );
}
function validateUnifiedDataset(bundle, validateCD) {
  if (!bundle || bundle.schema!=="auditory-abcd-unified-exploratory-v1" ||
      bundle.scientificStatus!=="exploratory_previously_inspected_reserved_test_cases" ||
      !Array.isArray(bundle.participants) ||
      bundle.participants.slice().sort().join(",")!=="P06,P07") {
    throw new Error("Unified file must be the labelled P06/P07 exploratory bundle");
  }
  if (typeof validateCD!=="function") throw new Error("C/D model validator unavailable");
  const records=validateDataset(bundle.records);
  const cd=validateCD(bundle.cd);
  if (cd.population!=="reserved_test_exploratory" ||
      cd.testParticipantsDisplayed!==2 ||
      cd.testParticipantsUsedForFitting!==0) {
    throw new Error("C/D must use frozen development-fitted models and explicit exploratory status");
  }
  const featureNames=[
    "Mental demand","Time between qualifying grabs","Time to locate target","Head turning"
  ];
  const seen=new Set();
  for (const legacy of records) {
    const who=legacy.ParticipantId, stage=stageCode(legacy.DifficultyStage);
    if (!["P06","P07"].includes(who)) throw new Error("Unexpected unified participant ID");
    const selected=cd.cases.find(item=>item.example===who&&item.stage===stage);
    if (!selected) throw new Error("Missing C/D result for "+who+" stage "+stage);
    const key=who+"|"+stage;
    if (seen.has(key)) throw new Error("Duplicate unified participant-stage case");
    seen.add(key);
    if (Math.abs(finite(legacy.Regression.Prediction)-selected.originalPrediction)>1e-7) {
      throw new Error("A/B and C/D original regression predictions differ");
    }
    const measurements=new Map(legacy.Regression.Measurements.map(
      item=>[item.Name,finite(item.Observed)]
    ));
    featureNames.forEach((name,index)=>{
      if (!measurements.has(name) ||
          Math.abs(measurements.get(name)-selected.measurements[index])>1e-7) {
        throw new Error("A/B and C/D measured inputs differ: "+name);
      }
    });
  }
  if (seen.size!==8) throw new Error("Incomplete P06/P07 unified case coverage");
  return {records,cd};
}
function installApp() {
  let records=[], cd=null, comparison=null, focused=null, activeTab="regression";
  const file=document.getElementById("data-file");
  const participant=document.getElementById("participant");
  const stage=document.getElementById("stage");
  const status=document.getElementById("load-status");
  const workspace=document.getElementById("workspace");
  const CD=window.AuditoryCD;
  if(!CD)throw new Error("C/D numerical explanation module missing");
  const isCD=()=>activeTab==="mamdani"||activeTab==="sugeno";
  const clearDecision=()=>{
    document.getElementById("next-stage").value="";
    document.getElementById("rationale").value="";
  };
  function render(){
    if(!records.length)return;
    const chosen=records.find(r=>r.ParticipantId===participant.value &&
      stageCode(r.DifficultyStage)===stage.value);
    if(!chosen)throw new Error("Selected participant-stage record missing");
    if(isCD()){
      if(!cd)throw new Error("Unified P06/P07 file required for C/D");
      CD.show(cd,participant.value,stage.value,activeTab);
    }else{
      renderRegression(chosen,comparison,stage.value,focused,name=>{
        focused=name;
        document.getElementById("shap-compare-details").open=true;
        render();
      });
      renderFuzzy(chosen);
    }
  }
  function setTab(name){
    if(!["regression","fuzzy","mamdani","sugeno"].includes(name))
      throw new Error("Unknown explanation method");
    if((name==="mamdani"||name==="sugeno")&&!cd){
      status.hidden=false;
      status.textContent="This file has A/B only. Load the unified A–D JSON for C/D.";
      return;
    }
    if(!records.length){
      status.hidden=false;
      status.textContent="Load a local explanation file first.";
      return;
    }
    activeTab=name;
    status.hidden=true;
    for(const method of ["regression","fuzzy","mamdani","sugeno"]){
      const selected=method===name;
      document.getElementById(method+"-view").hidden=!selected;
      document.getElementById("tab-"+method).setAttribute("aria-selected",String(selected));
    }
    clearDecision();render();
  }
  file.addEventListener("change",async()=>{
    workspace.hidden=true;records=[];cd=null;comparison=null;focused=null;status.hidden=false;
    try{
      if(!file.files||file.files.length!==1)throw new Error("Select one local JSON file");
      if(file.files[0].size>12_000_000)throw new Error("File exceeds local 12 MB limit");
      const parsed=JSON.parse(await file.files[0].text());
      if(Array.isArray(parsed)){
        records=validateDataset(parsed);
      }else{
        const unified=validateUnifiedDataset(parsed,CD.validate);
        records=unified.records;cd=unified.cd;
        comparison=validateShapComparison(parsed.shapComparison,records);
      }
      const labels=[...new Set(records.map(r=>r.ParticipantId))].sort();
      participant.replaceChildren(...labels.map(name=>{
        const option=document.createElement("option");
        option.value=name;option.textContent=name;return option;
      }));
      stage.value="0";
      document.getElementById("shap-stage-filter").value=stage.value;
      document.getElementById("shap-compare-details").open=false;
      workspace.hidden=false;
      status.textContent="";
      status.hidden=true;
      setTab("regression");
    }catch(error){
      records=[];cd=null;comparison=null;workspace.hidden=true;status.hidden=false;
      status.textContent="Cannot display file: "+error.message;
    }
  });
  participant.addEventListener("change",()=>{
    focused=null;
    document.getElementById("shap-stage-filter").value=stage.value;
    clearDecision();render();
  });
  stage.addEventListener("change",()=>{
    focused=null;
    document.getElementById("shap-stage-filter").value=stage.value;
    clearDecision();render();
  });
  document.getElementById("shap-stage-filter").addEventListener("change",()=>{
    if(activeTab==="regression")render();
  });
  for(const name of ["regression","fuzzy","mamdani","sugeno"])
    document.getElementById("tab-"+name).addEventListener("click",()=>setTab(name));
  for(const [suffix,action] of [["in","in"],["out","out"],["reset","reset"]])
    document.getElementById("cd-tree-zoom-"+suffix)
      .addEventListener("click",()=>CD.zoomTree(action));
}
if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", installApp);
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = {finite,stageCode,displayStage,validateDataset,validateUnifiedDataset,
    validateShapComparison,shapComparisonPoints,swarmOffsets,
    makeWaterfall,makeShapBeeswarm,shapFeatureLabel,shapMeasuredValue,
    featureLabel,ruleSentence,ruleSummary};
}
