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

function exploratoryFixture() {
  const d=fixture();
  d.schema="auditory-cd-exploratory-v1";
  d.population="reserved_test_exploratory";
  d.testParticipantsDisplayed=2;
  d.testParticipantsUsedForFitting=0;
  d.clusterMap.selectedPoints=d.clusterMap.points.slice(0,8).map(p=>structuredClone(p));
  d.cases.forEach((c,i)=>{
    c.example=i<4?"P06":"P07";
    c.mapPointIndex=i;
  });
  return d;
}
test("accept explicitly exploratory P06/P07 cases without changing the 104 development cloud",()=>{
  const d=exploratoryFixture();
  assert.equal(CD.validate(d).clusterMap.points.length,104);
  assert.equal(CD.validate(d).clusterMap.selectedPoints.length,8);
});
test("reject claiming P06/P07 were used to fit the development model",()=>{
  const d=exploratoryFixture();d.testParticipantsUsedForFitting=2;
  assert.throws(()=>CD.validate(d),/exploratory P06\/P07 provenance/);
});
test("reject passing development-example labels as test participant identities",()=>{
  const d=exploratoryFixture();d.cases[0].example="Development example 1";
  assert.throws(()=>CD.validate(d),/invalid case/);
});
test("reject using another test participant in exploratory display",()=>{
  const d=exploratoryFixture();d.cases[0].example="P08";
  assert.throws(()=>CD.validate(d),/invalid case/);
});
test("reject mixing a selected test point with a different membership",()=>{
  const d=exploratoryFixture();d.clusterMap.selectedPoints[0].memberships=[.1,.2,.7];
  assert.throws(()=>CD.validate(d),/membership\/contribution mismatch/);
});
test("reject including test points in the 104-observation development cloud",()=>{
  const d=exploratoryFixture();d.clusterMap.points.push(d.clusterMap.selectedPoints[0]);
  assert.throws(()=>CD.validate(d),/invalid PCA map/);
});

function withMockSvgDom(callback){
  const previous=global.document;
  const node=(tag)=>({
    tag,attributes:{},children:[],textContent:"",
    setAttribute(key,value){this.attributes[key]=String(value);},
    append(...children){this.children.push(...children);}
  });
  global.document={createElementNS:(_namespace,tag)=>node(tag),
    createElement:(tag)=>node(tag)};
  try{return callback();}finally{global.document=previous;}
}
function walkSvg(root){return [root,...root.children.flatMap(walkSvg)];}
test("tree zoom increases and resets without changing the tree source",()=>{
  assert.equal(CD.zoomStep(1,"in"),1.25);
  assert.equal(CD.zoomStep(1.25,"out"),1);
  assert.equal(CD.zoomStep(3.2,"in"),3.2);
  assert.equal(CD.zoomStep(.75,"out"),.75);
  assert.equal(CD.zoomStep(2.1,"reset"),1);
});
test("replace Medium only in therapist-visible wording",()=>{
  assert.equal(CD.linguisticLabel("Very low, Medium or High"),"Very low, Intermediate or High");
  assert.equal(CD.linguisticLabel("Mental demand is medium"),
    "Mental demand is intermediate");
  assert.equal(CD.linguisticLabel("Medium"),"Intermediate");
});
test("shared prediction axis contains both original MLR and centroid even if regression exceeds output universe",()=>{
  const bounds=CD.plotBounds([.46,.9,1.4,3.23],3.7,1.389);
  assert.ok(bounds.lower<.46);
  assert.ok(bounds.upper>3.7);
});
test("Mamdani SVG overlays original regression and centroid on one axis with shaded difference",()=>{
  const obs={
    originalPrediction:1.29,
    mamdani:{status:"OK",prediction:1.389,outputCurve:{
      x:[.46,.9,1.5,2,2.5,3.23],y:[0,.69,.69,.25,.25,0]
    }}
  };
  withMockSvgDom(()=>{
    const figure=CD.outputChart(obs),nodes=walkSvg(figure);
    assert.equal(figure.tag,"svg");
    assert.match(figure.attributes["aria-label"],/Regression prediction/i);
    assert.match(figure.attributes["aria-label"],/1.389/);
    assert.ok(nodes.some(n=>n.tag==="polygon"&&n.attributes.fill==="#7850c8"));
    assert.ok(nodes.some(n=>n.tag==="polyline"&&n.attributes.stroke==="#7147bf"));
    const orig=nodes.find(n=>n.tag==="line"&&n.attributes.stroke==="#687a90");
    const centroid=nodes.find(n=>n.tag==="line"&&n.attributes.stroke==="#173f6c");
    assert.ok(orig&&centroid);
    assert.ok(Number(orig.attributes.x1)<Number(centroid.attributes.x1));
    assert.ok(nodes.some(n=>n.tag==="rect"&&n.attributes.fill==="#c7d8f5"));
    assert.ok(nodes.some(n=>n.textContent==="Predicted difficulty level"));
    assert.ok(nodes.some(n=>n.textContent==="1.29"));
    assert.ok(nodes.some(n=>n.textContent==="1.389"));
    assert.ok(!nodes.some(n=>/^Regression |^Centroid /.test(n.textContent)));
    assert.equal(nodes.filter(n=>n.tag==="polyline"&&
      n.attributes.stroke==="#9bb1c5").length,5);
    for(const label of ["Very low","Low","Intermediate","High","Very high"])
      assert.ok(nodes.some(n=>n.textContent===label));
  });
});
test("Mamdani chart preserves unavailable without plotting fabricated markers",()=>{
  withMockSvgDom(()=>{
    const result=CD.outputChart({originalPrediction:1.29,
      mamdani:{status:"Unavailable",prediction:null,outputCurve:null}});
    assert.equal(result.tag,"p");
    assert.match(result.textContent,/Unavailable/);
  });
});

test("decision tree zoom controls change SVG width and reset scroll position",()=>{
  const old=global.document;
  const drawing={style:{}};
  const viewport={clientWidth:720,scrollLeft:55,scrollTop:38,
    querySelector:(name)=>name==="svg"?drawing:null};
  const percentage={textContent:""};
  global.document={getElementById:(id)=>{
    if(id==="cd-tree")return viewport;
    if(id==="cd-tree-zoom-value")return percentage;
    throw new Error("Unexpected DOM ID "+id);
  }};
  try{
    CD.zoomTree("reset");
    assert.equal(percentage.textContent,"100%");
    assert.equal(viewport.scrollLeft,0);
    const before=parseFloat(drawing.style.width);
    CD.zoomTree("in");
    assert.equal(percentage.textContent,"125%");
    assert.ok(parseFloat(drawing.style.width)>before);
    CD.zoomTree("reset");
    assert.equal(parseFloat(drawing.style.width),before);
  }finally{global.document=old;}
});


test("display rename changes rule prose without altering saved Medium consequent",()=>{
  assert.equal(
    CD.linguisticLabel("IF Mental demand is Medium THEN difficulty term Medium"),
    "IF Mental demand is Intermediate THEN difficulty level prediction is Intermediate"
  );
});
test("five output categories exactly reproduce original triangular and shoulder knots",()=>{
  const c=CD.outputCategories([0,1,2,3,4]);
  assert.deepEqual(c.centres,[0,1,2,3,4]);
  assert.deepEqual(c.names,["Very low","Low","Intermediate","High","Very high"]);
  for(let i=0;i<5;i++){
    assert.equal(c.values[i][i],1);
    assert.equal(c.values.reduce((sum,curve)=>sum+curve[i],0),1);
  }
  assert.deepEqual(c.values[0],[1,0,0,0,0]);
  assert.deepEqual(c.values[4],[0,0,0,0,1]);
  const dense=CD.outputCategories([0,.5,1,1.5,2,2.5,3,3.5,4]);
  assert.equal(dense.values[0][1],.5);
  assert.equal(dense.values[1][1],.5);
  assert.equal(dense.values[2][3],.5);
});
test("rule path map includes every positively activated fuzzy leaf and zeroes inactive branches",()=>{
  const d=fixture(),obs=d.cases[0],crisp=obs.mamdani.crispLeaf;
  const other=crisp===0?4:0;
  obs.mamdani.rules.forEach(r=>{r.activation=r.leaf===crisp?.69:r.leaf===other?.31:0;});
  const strengths=CD.activeTreeRoutes(d.tree,obs.mamdani.rules);
  assert.equal(strengths.get(d.tree),.69);
  const treeSvg=withMockSvgDom(()=>CD.treeChart(d.tree,obs));
  const nodes=walkSvg(treeSvg);
  assert.ok(nodes.some(n=>n.tag==="path"&&n.attributes.stroke==="#714bc3"));
  assert.ok(nodes.some(n=>n.tag==="path"&&n.attributes.stroke==="#198a9c"));
  assert.ok(nodes.some(n=>n.tag==="path"&&n.attributes.stroke==="#c7d2de"));
  assert.equal(nodes.filter(n=>n.tag==="rect"&&n.attributes.stroke==="#198a9c").length>0,true);
  assert.ok(nodes.some(n=>n.tag==="text"&&
    n.textContent.includes("Leaf "+other+" · 0.31")));
});
test("original fuzzy membership functions remain independent of selected regression marker",()=>{
  const a=CD.outputCategories([.46,1.1525,1.845,2.5375,3.23]);
  const b=CD.outputCategories([.46,1.1525,1.845,2.5375,3.23]);
  assert.deepEqual(a,b);
  const bounds=CD.plotBounds([.46,1.1525,1.845,2.5375,3.23],3.75,1.389);
  assert.ok(bounds.upper>3.75);
  assert.equal(a.centres.length,5);
});

test("Sugeno screen uses clusters, no visible case caption, and unchanged membership arithmetic",()=>{
  const d=exploratoryFixture();
  const prev=global.document;
  const elements=new Map();
  function element(tag){
    return {
      tag,attributes:{},children:[],style:{},textContent:"",className:"",
      setAttribute(name,value){this.attributes[name]=String(value);},
      append(...children){this.children.push(...children);},
      replaceChildren(...children){this.children=[...children];},
      addEventListener(name,callback){this.listeners??={};this.listeners[name]=callback;}
    };
  }
  global.document={
    createElement:element,
    createElementNS:(_ns,tag)=>element(tag),
    getElementById:(name)=>{
      if(!elements.has(name))elements.set(name,element("div"));
      return elements.get(name);
    }
  };
  try{
    CD.show(d,"P06","0","sugeno");
    const all=(root)=>walkSvg(root);
    const plot=elements.get("cd-scatter").children[0];
    assert.equal(plot.tag,"svg");
    const points=all(plot);
    assert.equal(points.filter(x=>x.tag==="circle").length,108);
    assert.ok(points.some(x=>x.tag==="title"&&x.textContent==="Current session"));
    assert.ok(!points.some(x=>x.tag==="text"&&x.textContent.includes("exploratory test case")));
    assert.ok(points.some(x=>x.tag==="text"&&x.textContent.startsWith("PC1")));
    assert.ok(points.some(x=>x.tag==="text"&&x.textContent.startsWith("PC2")));

    const membershipTexts=all(elements.get("cd-memberships")).map(n=>n.textContent);
    for(const cluster of ["Cluster 1","Cluster 2","Cluster 3"])
      assert.ok(membershipTexts.includes(cluster));
    assert.ok(!membershipTexts.some(t=>t.includes("Profile")));

    const comparison=elements.get("cd-profile-detail");
    assert.match(comparison.attributes["aria-label"],/^Cluster 3 recorded measurement comparison$/);
    const comparisonText=all(comparison).map(n=>n.textContent).filter(Boolean);
    assert.ok(!comparisonText.some(t=>t.includes("Profile")||
      t.includes("Each feature has its own scale")||
      t.includes("measurement comparison")||
      t.includes("Session value / learned")));
    assert.ok(comparisonText.includes("Solid = current session · outline = cluster center."));
    assert.equal(all(comparison).filter(n=>n.className==="cd-feature").length,4);

    const calculation=all(elements.get("cd-calculation")).map(n=>n.textContent);
    for(const cluster of ["Cluster 1","Cluster 2","Cluster 3"])
      assert.ok(calculation.includes(cluster));
    assert.ok(calculation.includes("Combined prediction = 2.3"));
    assert.ok(calculation.includes("0.2 × 1 = 0.2"));
    assert.ok(calculation.includes("0.3 × 2 = 0.6"));
    assert.ok(calculation.includes("0.5 × 3 = 1.5"));
    assert.ok(!calculation.some(t=>t.includes("Profile")));
  }finally{global.document=prev;}
});
