"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {finite, stageCode, displayStage, validateDataset, featureLabel, ruleSentence, ruleSummary} = require("./viewer.js");

function sampleDataset() {
  const features = [
    ["Mental demand", "score /10", 5, 0.4],
    ["Time between qualifying grabs", "seconds", 4.5, -0.1],
    ["Time to locate target", "seconds", 3, 0.2],
    ["Head turning", "degrees", 16, 0.1]
  ];
  const stages = [
    "Difficulty Stage 0 (D0)", "Difficulty Stage 1 (D2)",
    "Difficulty Stage 2 (D6)", "Difficulty Stage 3 (D10)"
  ];
  const rules = ["D1","D2","D3","D4","D5","I1","I2","I3","I4","I5"]
    .map(id => ({Rule:id, RuleName:id, Action:id[0]==="I"?"Increase":"Decrease",
      Activation:"0,3", Status:"candidate"}));
  return ["A","B"].flatMap((caseKey,person) =>
    stages.map(DifficultyStage => ({
      Case: caseKey, ParticipantId: person===0?"Synthetic01":"Synthetic02",
      DifficultyStage,
      Regression: {
        ReferencePrediction: 1.2, Prediction:1.8,
        Measurements:features.map(([Name,Unit,Observed,ShapContribution])=>({
          Name,Unit,Observed,ShapContribution,OutOfCalibration:"False",
          Membership:{Low:.2,Medium:.5,High:.3}
        }))
      },
      Fuzzy:{
        Recommendations:[.1,.2].map(Threshold => ({
          Threshold,IncreaseSupport:"0,4",DecreaseSupport:"0,2",
          FinalAction:"Maintain",RawAction:"Maintain",Status:"issued",
          ProposedLevel:"0",StrongestRules:"",ReviewFlags:"",DiagnosticFlags:""
        })),
        Rules:structuredClone(rules)
      }
    }))
  );
}
test("parse decimal comma safely, never as thousands separator",()=>{
  assert.equal(finite("4,39999999999997"), 4.39999999999997);
  assert.equal(finite("0,1"),.1);
  assert.throws(()=>finite("4.123,4"),/Invalid numeric/);
  assert.equal(finite("NaN",true),null);
  assert.throws(()=>finite("NaN"),/Invalid numeric/);
});
test("preserve source difficulty coding",()=>{
  assert.equal(stageCode("Difficulty Stage 2 (D6)"),"6");
  assert.equal(displayStage("10"),"Stage 3 (D10)");
});
test("accept eight internally coherent synthetic observations",()=>{
  assert.equal(validateDataset(sampleDataset()).length,8);
});
test("reject duplicate case-stage pair",()=>{
  const data=sampleDataset();data[1].DifficultyStage=data[0].DifficultyStage;
  assert.throws(()=>validateDataset(data),/Repeated/);
});
test("reject missing local participant ID",()=>{
  const data=sampleDataset();delete data[0].ParticipantId;
  assert.throws(()=>validateDataset(data),/ParticipantId missing/);
});
test("reject participant-ID inconsistencies",()=>{
  const data=sampleDataset();data[1].ParticipantId="Unexpected";
  assert.throws(()=>validateDataset(data),/Inconsistent/);
});
test("reject incorrect SHAP reconstruction",()=>{
  const data=sampleDataset();data[0].Regression.Prediction=2.6;
  assert.throws(()=>validateDataset(data),/reconstruction/);
});
test("accept MOD-18 decimal-encoded stage codes without changing source values",()=>{
  const data=sampleDataset();
  const levels=["0.0","2.0","6.0","10.0"];
  for (let i=0;i<4;i++) {
    data[i].Fuzzy.Recommendations[0].ProposedLevel=levels[i];
    data[i].Fuzzy.Recommendations[1].ProposedLevel=levels[i];
  }
  assert.equal(validateDataset(data).length,8);
  for (let i=0;i<4;i++) {
    assert.equal(data[i].Fuzzy.Recommendations[0].ProposedLevel,levels[i]);
  }
  assert.equal(displayStage("6.0"),"Stage 2 (D6)");
  assert.equal(displayStage(""),"Not proposed");
});
test("reject invalid fractional or unsupported proposed stage codes",()=>{
  for (const level of ["2.5","3.0","10.5","NaN","Infinity"]) {
    const data=sampleDataset();
    data[0].Fuzzy.Recommendations[0].ProposedLevel=level;
    assert.throws(()=>validateDataset(data),/Unsupported proposed stage code/);
  }
});
test("reject missing or repeated feature",()=>{
  const data=sampleDataset();data[0].Regression.Measurements[2].Name="Mental demand";
  assert.throws(()=>validateDataset(data),/duplicate Auditory feature/);
});
test("reject an altered fuzzy threshold",()=>{
  const data=sampleDataset();data[0].Fuzzy.Recommendations[1].Threshold=.3;
  assert.throws(()=>validateDataset(data),/thresholds/);
});
test("preserve Review and Unavailable when the status matches",()=>{
  const data=sampleDataset();
  data[0].Fuzzy.Recommendations[0].FinalAction="Review";
  data[0].Fuzzy.Recommendations[0].Status="boundary_review";
  data[0].Fuzzy.Recommendations[0].ProposedLevel="";
  data[0].Fuzzy.Recommendations[1].FinalAction="Unavailable";
  data[0].Fuzzy.Recommendations[1].Status="data_unavailable";
  data[0].Fuzzy.Recommendations[1].ProposedLevel="";
  assert.equal(validateDataset(data).length,8);
});
test("reject a misleading Review status",()=>{
  const data=sampleDataset();data[0].Fuzzy.Recommendations[0].FinalAction="Review";
  assert.throws(()=>validateDataset(data),/Inconsistent fuzzy/);
});
test("reject invalid membership values or stages",()=>{
  const data=sampleDataset();data[0].Regression.Measurements[0].Membership.High=1.3;
  assert.throws(()=>validateDataset(data),/outside/);
  const data2=sampleDataset();data2[1].Fuzzy.Recommendations[0].ProposedLevel="3";
  assert.throws(()=>validateDataset(data2),/stage code/);
});
test("reject duplicate or missing rule IDs",()=>{
  const data=sampleDataset();data[0].Fuzzy.Rules[1].Rule="D1";
  assert.throws(()=>validateDataset(data),/Duplicate or unknown fuzzy rule/);
});

test("use requested therapist-facing labels without changing stored feature keys",()=>{
  assert.equal(featureLabel("Mental demand"),"Subjective mental demand rating");
  assert.equal(featureLabel("Time between qualifying grabs"),"Time interval between grabs");
  assert.equal(featureLabel("Time to locate target"),"Duration to locate a target product");
  assert.equal(featureLabel("Head turning"),"Head rotation while locating a product");
  assert.throws(()=>featureLabel("Visual irrelevant focus"),/Unknown Auditory/);
});
test("render exact MOD-18 auditory antecedents as full if-then statements",()=>{
  assert.equal(ruleSentence("D1"),
    "If subjective mental demand rating is high AND time interval between grabs is high, then decrease difficulty.");
  assert.equal(ruleSentence("I5"),
    "If subjective mental demand rating is low AND time interval between grabs is low AND head rotation while locating a product is medium, then increase difficulty.");
  assert.equal(ruleSentence("I4"),
    "If subjective mental demand rating is low AND duration to locate a target product is low AND time interval between grabs is medium, then increase difficulty.");
  assert.throws(()=>ruleSentence("V1"),/Unknown Auditory rule/);
});
test("show strongest positive rule signal without choosing a policy threshold",()=>{
  const data=sampleDataset();
  const record=data[1];
  record.Fuzzy.Rules.find(r=>r.Rule==="I5").Activation="0.2";
  record.Fuzzy.Recommendations[0].FinalAction="Increase";
  record.Fuzzy.Recommendations[1].FinalAction="Maintain";
  // P07 Stage 1 test-style disagreement: no threshold is chosen in the rule-level summary.
  const summary=ruleSummary(record);
  assert.equal(summary.suggestion,"Increase");
  assert.equal(summary.main[0].Rule,"I5");
  assert.equal(summary.other.length,9);
  assert.equal(record.Fuzzy.Recommendations[1].FinalAction,"Maintain");
});
test("show no directional suggestion when all rule activations are zero",()=>{
  const record=sampleDataset()[0];
  record.Fuzzy.Rules.forEach(r=>{r.Activation=0;});
  assert.equal(ruleSummary(record).suggestion,"No directional suggestion");
});
test("preserve boundary Review and missing-data Unavailable in rule summary",()=>{
  const boundary=sampleDataset()[0];
  boundary.Fuzzy.Rules.forEach(r=>{r.Activation=0;});
  boundary.Fuzzy.Rules.find(r=>r.Rule==="D1").Activation=.8;
  assert.equal(ruleSummary(boundary).suggestion,"Review");
  const unavailable=sampleDataset()[3];
  unavailable.Fuzzy.Recommendations[1].Status="data_unavailable";
  unavailable.Fuzzy.Recommendations[1].FinalAction="Unavailable";
  assert.equal(ruleSummary(unavailable).suggestion,"Unavailable");
});
test("do not hide equally strong opposing rules",()=>{
  const record=sampleDataset()[1];
  record.Fuzzy.Rules.forEach(r=>{r.Activation=0;});
  record.Fuzzy.Rules.find(r=>r.Rule==="D1").Activation=.5;
  record.Fuzzy.Rules.find(r=>r.Rule==="I1").Activation=.5;
  const summary=ruleSummary(record);
  assert.equal(summary.suggestion,"Mixed indications");
  assert.equal(summary.main.length,2);
});
