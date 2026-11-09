"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const {validateUnifiedDataset,validateDataset}=require("./viewer.js");

const FIELDS=["Mental demand","Time between qualifying grabs",
  "Time to locate target","Head turning"];
const STAGES=[
  ["0","Difficulty Stage 0 (D0)"],["2","Difficulty Stage 1 (D2)"],
  ["6","Difficulty Stage 2 (D6)"],["10","Difficulty Stage 3 (D10)"]
];
function bundle(){
  const rules=Array.from({length:10},(_,i)=>{
    const name=(i<5?"D":"I")+(i%5+1);
    return {Rule:name,RuleName:name,Action:i<5?"Decrease":"Increase",
      Activation:0.3,Status:"candidate"};
  });
  const records=["P06","P07"].flatMap((who,i)=>STAGES.map(([code,label])=>({
    Case:i===0?"A":"B",ParticipantId:who,DifficultyStage:label,
    Regression:{
      ReferencePrediction:1.2,Prediction:1.8,
      Measurements:FIELDS.map((name,j)=>({
        Name:name,Unit:["score /10","seconds","seconds","degrees"][j],
        Observed:[5,4.5,3,16][j],
        ShapContribution:[.4,-.1,.2,.1][j],
        Membership:{Low:.2,Medium:.5,High:.3}
      }))
    },
    Fuzzy:{
      Recommendations:[.1,.2].map(threshold=>({
        Threshold:threshold,IncreaseSupport:.4,DecreaseSupport:.2,
        FinalAction:"Maintain",RawAction:"Maintain",
        Status:"issued",ProposedLevel:"0",StrongestRules:"",
        ReviewFlags:"",DiagnosticFlags:""
      })),
      Rules:structuredClone(rules)
    }
  })));
  const cd={
    population:"reserved_test_exploratory",
    testParticipantsDisplayed:2,testParticipantsUsedForFitting:0,
    cases:["P06","P07"].flatMap(who=>STAGES.map(([stage])=>({
      example:who,stage,originalPrediction:1.8,
      measurements:[5,4.5,3,16]
    })))
  };
  return {
    schema:"auditory-abcd-unified-exploratory-v1",
    scientificStatus:"exploratory_previously_inspected_reserved_test_cases",
    participants:["P06","P07"],records,cd
  };
}
const validateCD=cd=>cd;
test("original A/B records retain all previous validation constraints",()=>{
  assert.equal(validateDataset(bundle().records).length,8);
});
test("accept one file for exactly P06/P07 with all 4 stages and matched predictions",()=>{
  const x=validateUnifiedDataset(bundle(),validateCD);
  assert.equal(x.records.length,8);
  assert.equal(x.cd.cases.length,8);
});
test("require explicit exploratory source status",()=>{
  const x=bundle();x.scientificStatus="untouched_independent_test";
  assert.throws(()=>validateUnifiedDataset(x,validateCD),/exploratory bundle/);
});
test("reject unknown participant identity",()=>{
  const x=bundle();x.participants=["P06","P08"];
  assert.throws(()=>validateUnifiedDataset(x,validateCD),/exploratory bundle/);
});
test("reject mixing a P06 test case with a P07 fuzzy output",()=>{
  const x=bundle();x.cd.cases[0].example="P08";
  assert.throws(()=>validateUnifiedDataset(x,validateCD),/Missing C\/D result/);
});
test("reject mismatched original MLR predictions between A/B and C/D",()=>{
  const x=bundle();x.cd.cases[0].originalPrediction+=0.005;
  assert.throws(()=>validateUnifiedDataset(x,validateCD),/regression predictions differ/);
});
test("reject mismatched mental-demand input between methods",()=>{
  const x=bundle();x.cd.cases[0].measurements[0]+=0.01;
  assert.throws(()=>validateUnifiedDataset(x,validateCD),/measured inputs differ/);
});
test("reject mismatched head-turning measurement between methods",()=>{
  const x=bundle();x.cd.cases[0].measurements[3]+=0.01;
  assert.throws(()=>validateUnifiedDataset(x,validateCD),/measured inputs differ/);
});
test("do not accept a C/D inference that claims fitting on test people",()=>{
  const x=bundle();x.cd.testParticipantsUsedForFitting=2;
  assert.throws(()=>validateUnifiedDataset(x,validateCD),/frozen development-fitted models/);
});
test("reject missing an observation from A or B",()=>{
  const x=bundle();x.records.pop();
  assert.throws(()=>validateUnifiedDataset(x,validateCD),/eight Auditory observations/);
});
test("reject legacy SHAP reconstruction failure inside the unified bundle",()=>{
  const x=bundle();x.records[0].Regression.Prediction=1.6;
  assert.throws(()=>validateUnifiedDataset(x,validateCD),/SHAP reconstruction/);
});
test("call the C/D numerical validator, not simply trust its provenance label",()=>{
  const x=bundle();
  assert.throws(()=>validateUnifiedDataset(x,()=>{throw new Error("modified centroid");}),
    /modified centroid/);
});
