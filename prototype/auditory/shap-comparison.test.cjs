"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const {validateShapComparison,shapComparisonPoints,shapInterpretation,
  makeWaterfall,makeShapBeeswarm,swarmOffsets} = require("./viewer.js");

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
test("interpretation remains descriptive, not clinical instruction",()=>{
  const {records}=fixture();
  const explanation=shapInterpretation(records[0].Regression);
  assert.match(explanation,/fitted prediction/);
  assert.match(explanation,/not effects of changing a measurement/);
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
    assert.match(chart.attributes["aria-label"],/26 development trials/);
    assert.equal(chart.children.filter(node=>node.attributes.class==="shap-row-selected").length,1);
  });
});
