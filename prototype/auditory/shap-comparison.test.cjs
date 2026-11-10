"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const {validateShapComparison,shapComparisonPoints,
  makeWaterfall,makeShapBeeswarm,swarmOffsets,
  shapFeatureLabel,shapMeasuredValue,signTick,shapContributionColor} = require("./viewer.js");

const stages=["0","2","6","10"];
const names=["Mental demand","Time between qualifying grabs","Time to locate target","Head turning"];
const observed=[5,4.5,3,16], means=[4.6,4.6,2.8,15.9];
const intercept=1.2-means.reduce((a,b)=>a+b,0);
function fixture(){
  const rows=stages.flatMap(stage=>Array.from({length:26},()=>({
    stage,values:means.slice(),contributions:[0,0,0,0]
  })));
  const byReferenceStage=Object.fromEntries(stages.map(stage=>[stage,{
    referencePrediction:1.2,meanFeatures:means.slice(),
    observations:structuredClone(rows)
  }]));
  const comparison={
    schema:"auditory-shap-comparison-v1",population:"development",
    model:"auditory",developmentParticipants:26,developmentObservations:104,
    features:names.slice(),coefficients:[1,1,1,1],intercept,byReferenceStage
  };
  const labels=["Difficulty Stage 0 (D0)","Difficulty Stage 1 (D2)",
    "Difficulty Stage 2 (D6)","Difficulty Stage 3 (D10)"];
  const records=["P06","P07"].flatMap(person=>stages.map((stage,i)=>({
    ParticipantId:person,DifficultyStage:labels[i],
    Regression:{ReferencePrediction:1.2,Prediction:1.8,Measurements:names.map((Name,j)=>({
      Name,Unit:j===0?"/10":"seconds",Observed:observed[j],
      ShapContribution:observed[j]-means[j]
    }))}
  })));
  return {comparison,records};
}
test("accept development-only fixed-reference SHAP comparison",()=>{
  const {comparison,records}=fixture();
  assert.equal(validateShapComparison(comparison,records),comparison);
});
test("default population has 26 observations; all stages has 104",()=>{
  const {comparison}=fixture();
  assert.equal(shapComparisonPoints(comparison,"6","current").length,26);
  assert.equal(shapComparisonPoints(comparison,"6","all").length,104);
  assert.equal(shapComparisonPoints(comparison,"6","0").length,26);
});
test("stage filters never mutate the same selected-stage reference",()=>{
  const {comparison}=fixture();
  const before=JSON.stringify(comparison.byReferenceStage["6"]);
  shapComparisonPoints(comparison,"6","0");
  shapComparisonPoints(comparison,"6","all");
  assert.equal(JSON.stringify(comparison.byReferenceStage["6"]),before);
});
test("comparison accepts absent data for legacy A/B JSON",()=>{
  assert.equal(validateShapComparison(undefined,fixture().records),null);
});
test("reject incorrectly re-referenced cross-stage development dots",()=>{
  const {comparison,records}=fixture();
  comparison.byReferenceStage["6"].observations[0].contributions[0]=1;
  assert.throws(()=>validateShapComparison(comparison,records),/wrong reference/);
});
test("reject a wrong original waterfall reference",()=>{
  const {comparison,records}=fixture();
  records[0].Regression.ReferencePrediction=1.3;
  assert.throws(()=>validateShapComparison(comparison,records),/waterfall reference/);
});
test("reject a changed selected-trial SHAP feature value",()=>{
  const {comparison,records}=fixture();
  records[0].Regression.Measurements[0].ShapContribution=.8;
  assert.throws(()=>validateShapComparison(comparison,records),/mismatches comparison/);
});
test("reject test-like or identifying comparison records",()=>{
  const {comparison,records}=fixture();
  comparison.byReferenceStage["0"].observations[0].participant_id="P06";
  assert.throws(()=>validateShapComparison(comparison,records),/identifying/);
});
test("reject population and participant count modifications",()=>{
  const {comparison,records}=fixture();
  comparison.developmentParticipants=32;
  assert.throws(()=>validateShapComparison(comparison,records),/provenance/);
});
test("reject missing development observations in a stage",()=>{
  const {comparison,records}=fixture();
  comparison.byReferenceStage["2"].observations.pop();
  assert.throws(()=>validateShapComparison(comparison,records),/Missing development/);
});
test("reject malformed stage filter",()=>{
  const {comparison}=fixture();
  assert.throws(()=>shapComparisonPoints(comparison,"6","prediction:high"),/Unknown SHAP comparison/);
});
test("SHAP labels exactly match current C/D names and units",()=>{
  const {records}=fixture();
  const exact=[
    "Subjective mental demand rating",
    "Time interval between grabs",
    "Duration to locate a target product",
    "Total Head rotation while locating a product"
  ];
  records[0].Regression.Measurements.forEach((feature,i)=>{
    assert.equal(shapFeatureLabel(feature.Name),exact[i]);
    assert.ok(!shapFeatureLabel(feature.Name).includes("\n"));
  });
  assert.equal(shapMeasuredValue(records[0].Regression.Measurements[0]),"5 /10");
  assert.equal(shapMeasuredValue(records[0].Regression.Measurements[1]),"4.5 s");
  assert.equal(shapMeasuredValue(records[0].Regression.Measurements[3]),"16 °");
});

function withSvgDocument(run){
  const previous=global.document;
  const node=tag=>({tag,attributes:{},children:[],handlers:{},textContent:"",
    setAttribute(k,v){this.attributes[k]=String(v)},
    append(...nodes){this.children.push(...nodes)},
    addEventListener(k,fn){this.handlers[k]=fn}
  });
  global.document={createElementNS:(_ns,tag)=>node(tag)};
  try{return run()}finally{global.document=previous}
}
test("swarm offsets are deterministic and avoid overplotting colliding dots",()=>{
  const positions=[100,100,100,112,112];
  const result=swarmOffsets(positions);
  assert.deepEqual(result,swarmOffsets(positions));
  assert.equal(result.length,5);
  assert.ok(new Set(result.slice(0,3)).size===3);
});
test("waterfall feature can focus its matching comparison row",()=>{
  const {records}=fixture();
  withSvgDocument(()=>{
    let focus=null;
    const svg=makeWaterfall(records[0].Regression,name=>{focus=name});
    const first=svg.children.find(node=>node.tag==="rect"&&node.attributes["data-shap-feature"]);
    assert.equal(first.attributes["data-shap-feature"],names[0]);
    first.handlers.click();
    assert.equal(focus,names[0]);
    assert.equal(first.attributes.role,"button");
  });
});
test("beeswarm includes four highlighted current-trial markers",()=>{
  const {records,comparison}=fixture();
  withSvgDocument(()=>{
    const chart=makeShapBeeswarm(records[0].Regression,comparison,"6","current",names[1]);
    const current=chart.children.filter(node=>node.tag==="circle"&&
      node.attributes.stroke==="#7052b4");
    assert.equal(current.length,4);
    assert.match(chart.attributes["aria-label"],/26 other development observations/);
    assert.equal(chart.children.filter(node=>node.attributes.class==="shap-row-selected").length,1);
  });
});

test("waterfall uses green/red bars and labels the population prediction reference",()=>{
  const {records}=fixture();
  withSvgDocument(()=>{
    const chart=makeWaterfall(records[0].Regression);
    const bars=chart.children.filter(n=>n.tag==="rect"&&n.attributes["data-shap-feature"]);
    assert.equal(bars.length,4);
    assert.ok(bars.some(n=>n.attributes.fill==="#25835c"));
    assert.ok(bars.some(n=>n.attributes.fill==="#c04c46"));
    const text=chart.children.filter(n=>n.tag==="text").map(n=>n.textContent);
    assert.ok(text.includes("Population difficulty level"));
    assert.ok(text.includes("prediction reference"));
    assert.ok(text.includes("Model-predicted difficulty level"));
    names.forEach(name=>assert.ok(text.includes(shapFeatureLabel(name))));
  });
});
test("beeswarm has explicit zero tick and a Feature contribution axis",()=>{
  const {records,comparison}=fixture();
  withSvgDocument(()=>{
    const chart=makeShapBeeswarm(records[0].Regression,comparison,"6","all",null);
    const text=chart.children.filter(n=>n.tag==="text").map(n=>n.textContent);
    assert.ok(text.includes("0"));
    assert.ok(text.includes("Feature contribution"));
    assert.equal(text.filter(x=>x==="Feature contribution").length,1);
    names.forEach(name=>assert.ok(text.includes(shapFeatureLabel(name))));
  });
});

test("beeswarm contribution ticks use explicit signs and a single zero",()=>{
  assert.equal(signTick(-0.5),"−0.5");
  assert.equal(signTick(0),"0");
  assert.equal(signTick(0.5),"+0.5");
  assert.equal(signTick(0.0025,4),"+0.0025");
  const {records,comparison}=fixture();
  withSvgDocument(()=>{
    const chart=makeShapBeeswarm(records[0].Regression,comparison,"6","all",null);
    const texts=chart.children.filter(n=>n.tag==="text").map(n=>n.textContent);
    assert.ok(texts.some(x=>/^\+0[.]\d+/.test(x)));
    assert.ok(texts.some(x=>/^−0[.]\d+/.test(x)));
    assert.equal(texts.filter(x=>x==="0").length,1);
  });
});
test("beeswarm quarter-step ticks include negative values without clipping points",()=>{
  const {records,comparison}=fixture();
  withSvgDocument(()=>{
    const chart=makeShapBeeswarm(records[0].Regression,comparison,"6","all",null);
    const axisText=chart.children.filter(n=>n.tag==="text"&&
      n.attributes.class==="axis-label").map(n=>n.textContent);
    const labels=axisText.filter(t=>t==="0" || /^[+−]\d+\.\d{2}$/.test(t));
    assert.ok(labels.includes("−0.25"));
    assert.ok(labels.includes("0"));
    assert.ok(labels.includes("+0.25"));
    assert.equal(labels.filter(t=>t==="0").length,1);
    const numbers=labels.map(t=>t==="0"?0:Number(t.replace("−","-")));
    for(let i=1;i<numbers.length;i++){
      assert.ok(Math.abs(numbers[i]-numbers[i-1]-.25)<1e-10);
    }
    const dots=chart.children.filter(n=>n.tag==="circle"&&n.attributes.opacity);
    assert.ok(dots.every(n=>Number(n.attributes.cx)>=370&&
      Number(n.attributes.cx)<=1030));
    assert.ok(dots.every(n=>Number(n.attributes.opacity)>=.9));
  });
});
test("subtle signed contributions are visibly coloured against grey zero",()=>{
  const grey=shapContributionColor(0,1);
  const weakNegative=shapContributionColor(-.01,1);
  const weakPositive=shapContributionColor(.01,1);
  assert.notEqual(weakNegative,grey);
  assert.notEqual(weakPositive,grey);
  const channels=value=>value.match(/\d+/g).map(Number);
  const [redR,redG]=channels(weakNegative);
  const [greenR,greenG]=channels(weakPositive);
  assert.ok(redR-redG>30);
  assert.ok(greenG-greenR>10);
});
test("beeswarm colours are based on signed SHAP contributions, not raw measurements",()=>{
  assert.equal(shapContributionColor(0,1),"rgb(156,163,175)");
  assert.equal(shapContributionColor(-1,1),"rgb(192,76,70)");
  assert.equal(shapContributionColor(1,1),"rgb(37,131,92)");
  const {records,comparison}=fixture();
  withSvgDocument(()=>{
    const chart=makeShapBeeswarm(records[0].Regression,comparison,"6","current",null);
    const dots=chart.children.filter(n=>n.tag==="circle"&&n.attributes.opacity);
    assert.ok(dots.length>=26);
    // This synthetic background has equal zero SHAP contributions in every row.
    assert.ok(dots.every(n=>n.attributes.fill==="rgb(156,163,175)"));
    const titles=chart.children.filter(n=>n.tag==="circle").flatMap(n=>n.children||[])
      .filter(n=>n.tag==="title").map(n=>n.textContent);
    assert.ok(titles.some(s=>s.startsWith("Current observation;")));
  });
});
