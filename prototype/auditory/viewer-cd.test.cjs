"use strict";
const test=require("node:test");
const assert=require("node:assert/strict");
const CD=require("./viewer-cd.js");
const KEYS=CD?[
  "mental_demand_score_0_to_10",
  "median_time_between_qualifying_grabs_seconds",
  "median_time_to_target_seconds",
  "median_head_turning_degrees"
]:[];
function tree(){
  let leaf=0;
  function make(depth){
    if(depth===3)return {leaf:leaf++,value:1.2};
    return {feature:KEYS[depth],threshold:5,left:make(depth+1),right:make(depth+1)};
  }
  return make(0);
}
function fixture(){
  const structure=tree();
  const profiles=[1,2,3].map((cluster,i)=>({
    cluster,constant:cluster,centre:[i+2,i+4,i+1,40+i*10]
  }));
  const memberships=[0.2,0.3,0.5],contributions=[.2,.6,1.5];
  const map={type:"pca-development",explainedVariance:[.62,.25],
    points:Array.from({length:104},(_,i)=>({x:i/30,y:(i%17)/8,memberships:[...memberships]})),
    centres:[{x:.5,y:.5},{x:2,y:1},{x:3,y:2}]
  };
  const curve={x:Array.from({length:101},(_,i)=>i*3/100),
    y:Array.from({length:101},()=>1)};
  const features=KEYS.map((key,i)=>({key,maximum:[10,9,3,180][i]}));
  const measurements=[4,3,2,40];
  const crisp=CD.route(structure,measurements).leaf;
  const cases=["Development example 1","Development example 2"].flatMap((example,j)=>
    ["0","2","6","10"].map((stage,i)=>({
      example,stage,originalPrediction:1.7,measurements:[...measurements],
      mapPointIndex:j*4+i,
      mamdani:{status:"OK",prediction:1.5,outputTerms:5,crispLeaf:crisp,
        rules:Array.from({length:8},(_,id)=>({leaf:id,activation:.2,
          consequent:"Medium",antecedent:"IF feature is Medium THEN Medium"})),
        outputCurve:structuredClone(curve)},
      sugeno:{status:"OK",prediction:2.3,memberships:[...memberships],
        contributions:[...contributions]}
    }))
  );
  return {schema:"auditory-cd-development-v1",population:"development",
    reservedTestParticipantsUsed:0,participants:26,observations:104,
    features,tree:structure,profiles,clusterMap:map,cases};
}
test("accept complete anonymised 26-person development-only C/D bundle",()=>{
  assert.equal(CD.validate(fixture()).cases.length,8);
});
test("verify decision tree path for the selected four measured values",()=>{
  const d=fixture();const outcome=CD.route(d.tree,d.cases[0].measurements);
  assert.equal(outcome.leaf,d.cases[0].mamdani.crispLeaf);
  assert.equal(outcome.path.length,4);
});
test("recover centroid from actual curve without confidence claims",()=>{
  assert.ok(Math.abs(CD.centroid([0,1,2,3],[1,1,1,1])-1.5)<1e-10);
  assert.equal(CD.centroid([0,1,2],[0,0,0]),null);
});
test("reject reserved-test provenance",()=>{
  const d=fixture();d.population="test";
  assert.throws(()=>CD.validate(d),/development provenance/);
});
test("reject test participants used flag even with development label",()=>{
  const d=fixture();d.reservedTestParticipantsUsed=1;
  assert.throws(()=>CD.validate(d),/development provenance/);
});
test("reject wrong participant or observation count",()=>{
  const d=fixture();d.participants=32;
  assert.throws(()=>CD.validate(d),/development provenance/);
});
test("reject participant identity and stage mismatch",()=>{
  const d=fixture();d.cases[0].stage=d.cases[1].stage;
  assert.throws(()=>CD.validate(d),/duplicate case/);
});
test("reject missing stage and incomplete case collection",()=>{
  const d=fixture();d.cases.pop();
  assert.throws(()=>CD.validate(d),/eight development/);
});
test("reject changed four-feature ordering",()=>{
  const d=fixture();d.features.reverse();
  assert.throws(()=>CD.validate(d),/four Auditory features/);
});
test("reject missing tree leaf",()=>{
  const d=fixture();d.tree.left.left.left={leaf:25,value:1.2};
  assert.throws(()=>CD.validate(d),/invalid Mamdani case/);
});
test("reject wrong highlighted crisp route",()=>{
  const d=fixture();d.cases[0].mamdani.crispLeaf=7;
  assert.throws(()=>CD.validate(d),/invalid Mamdani case/);
});
test("reject wrong Mamdani centroid",()=>{
  const d=fixture();d.cases[0].mamdani.prediction=1.9;
  assert.throws(()=>CD.validate(d),/centroid mismatch/);
});
test("preserve Unavailable when no valid Mamdani score exists",()=>{
  const d=fixture();d.cases[0].mamdani={...d.cases[0].mamdani,
    status:"Unavailable",prediction:null,outputCurve:null};
  assert.equal(CD.validate(d).cases[0].mamdani.status,"Unavailable");
});
test("reject fabricated value for Unavailable result",()=>{
  const d=fixture();d.cases[0].mamdani.status="Unavailable";
  assert.throws(()=>CD.validate(d),/prediction is not null/);
});
test("reject altered membership proportions",()=>{
  const d=fixture();d.cases[0].sugeno.memberships=[.2,.3,.49];
  assert.throws(()=>CD.validate(d),/membership/);
});
test("reject selected point belonging to another cluster profile",()=>{
  const d=fixture();d.clusterMap.points[0].memberships=[.6,.2,.2];
  assert.throws(()=>CD.validate(d),/membership\/contribution/);
});
test("reject modified Sugeno weighted contribution",()=>{
  const d=fixture();d.cases[0].sugeno.contributions[0]=.22;
  assert.throws(()=>CD.validate(d),/membership\/contribution/);
});
test("reject modified Sugeno prediction",()=>{
  const d=fixture();d.cases[0].sugeno.prediction=2.4;
  assert.throws(()=>CD.validate(d),/weighted sum mismatch/);
});
test("reject made-up display PCA coordinates",()=>{
  const d=fixture();d.clusterMap.points[0].x=Infinity;
  assert.throws(()=>CD.validate(d),/projected point/);
});
