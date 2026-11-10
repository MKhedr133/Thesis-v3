"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const {validateShapComparison,shapComparisonPoints,shapInterpretation} = require("./viewer.js");

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
