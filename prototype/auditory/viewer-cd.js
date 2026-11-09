"use strict";
/* Browser-only renderer: frozen development-fitted C/D, with optional explicitly exploratory reserved-test display. Never fits models. */
(function (root) {
  const NS = "http://www.w3.org/2000/svg";
  const KEYS = ["mental_demand_score_0_to_10","median_time_between_qualifying_grabs_seconds","median_time_to_target_seconds","median_head_turning_degrees"];
  const LABELS = ["Mental demand","Grab interval","Time to target","Head turning"];
  const UNITS = ["/10","s","s","°"];
  const COLORS = ["#3576a5","#228d85","#ae7c35"];
  const linguisticLabel = value => String(value).replace(/\bMedium\b/g,"Intermediate")
    .replace(/\bmedium\b/g,"intermediate");
  function plotBounds(grid, regression, prediction) {
    insist(grid.length>=3 && finite(regression) && finite(prediction),
      "incomplete prediction-axis inputs");
    const lower=Math.min(grid[0],regression,prediction);
    const upper=Math.max(grid[grid.length-1],regression,prediction);
    const pad=Math.max((upper-lower)*.04,.015);
    return {lower:lower-pad,upper:upper+pad};
  }
  const zoomStep = (value,action) => action==="reset"?1:
    Math.max(.75,Math.min(3.2,value*(action==="in"?1.25:action==="out"?.8:1)));
  let currentTreeZoom=1;
  function zoomTree(action) {
    currentTreeZoom=zoomStep(currentTreeZoom,action);
    const viewport=document.getElementById("cd-tree");
    const drawing=viewport.querySelector("svg");
    if(drawing){
      drawing.style.width=Math.round(Math.max(440,viewport.clientWidth-16)*currentTreeZoom)+"px";
      drawing.style.minWidth="0";
      drawing.style.maxWidth="none";
    }
    document.getElementById("cd-tree-zoom-value").textContent=
      Math.round(currentTreeZoom*100)+"%";
    if(action==="reset"){viewport.scrollLeft=0;viewport.scrollTop=0;}
    return currentTreeZoom;
  }
  const finite = v => typeof v === "number" && Number.isFinite(v);
  const near = (a,b,t=1e-8) => Math.abs(a-b)<=t;
  function insist(v,msg){if(!v)throw new Error("C/D development data: "+msg);}
  function f(v,n=3){return v===null?"Unavailable":Number(v).toLocaleString("en-GB",{maximumFractionDigits:n});}
  function html(tag,cls,txt){const x=document.createElement(tag);if(cls)x.className=cls;if(txt!==undefined)x.textContent=txt;return x;}
  function svg(tag,attrs,txt){const x=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs||{}))x.setAttribute(k,String(v));if(txt!==undefined)x.textContent=txt;return x;}
  function leavesOf(node,seen=new Set(),depth=0){
    insist(node&&depth<7,"invalid decision tree");
    if(Object.hasOwn(node,"leaf")){insist(Number.isInteger(node.leaf)&&!seen.has(node.leaf)&&finite(node.value),"invalid leaf");seen.add(node.leaf);return seen;}
    insist(KEYS.includes(node.feature)&&finite(node.threshold)&&node.left&&node.right,"invalid tree split");
    leavesOf(node.left,seen,depth+1);leavesOf(node.right,seen,depth+1);return seen;
  }
  function route(tree,values){
    const path=[];let n=tree;
    while(n){path.push(n);if(Object.hasOwn(n,"leaf"))return {leaf:n.leaf,path};
      const i=KEYS.indexOf(n.feature);insist(i>=0&&path.length<8,"invalid tree path");
      n=values[i]<=n.threshold?n.left:n.right;
    }throw new Error("Missing decision tree leaf");
  }
  function centroid(x,y){
    insist(x.length===y.length&&x.length>=3,"invalid fuzzy curve");
    let a=0,m=0;
    for(let i=1;i<x.length;i++){const dx=x[i]-x[i-1];
      insist(finite(dx)&&dx>0&&finite(y[i])&&y[i]>=0&&y[i]<=1,"invalid curve coordinate");
      a+=dx*(y[i-1]+y[i])/2;m+=dx*(x[i-1]*y[i-1]+x[i]*y[i])/2;
    }return a>1e-12?m/a:null;
  }
  function validate(data){
    const exploratory = !!data && data.schema==="auditory-cd-exploratory-v1";
    if(exploratory)insist(data.population==="reserved_test_exploratory"&&
      data.testParticipantsDisplayed===2&&data.testParticipantsUsedForFitting===0&&
      data.participants===26&&data.observations===104,
      "exploratory P06/P07 provenance is incorrect");
    else insist(data&&data.schema==="auditory-cd-development-v1"&&data.population==="development"&&
      data.reservedTestParticipantsUsed===0&&data.participants===26&&data.observations===104,
      "incorrect development provenance");
    insist(Array.isArray(data.features)&&data.features.length===4&&
      data.features.every((v,i)=>v.key===KEYS[i]&&finite(v.maximum)&&v.maximum>0),"wrong four Auditory features");
    const ids=leavesOf(data.tree);insist(ids.size===8,"expected eight CART leaves");
    insist(Array.isArray(data.profiles)&&data.profiles.length===3,"expected three Sugeno clusters");
    data.profiles.forEach((p,i)=>insist(p.cluster===i+1&&finite(p.constant)&&
      Array.isArray(p.centre)&&p.centre.length===4&&p.centre.every(finite),"invalid profile"));
    const map=data.clusterMap;
    insist(map&&map.type==="pca-development"&&Array.isArray(map.points)&&map.points.length===104&&
      Array.isArray(map.centres)&&map.centres.length===3&&
      Array.isArray(map.explainedVariance)&&map.explainedVariance.length===2&&
      map.explainedVariance.every(v=>finite(v)&&v>=0&&v<=1),"invalid PCA map");
    const selectedPoints = exploratory ? map.selectedPoints : map.points;
    if(exploratory)insist(Array.isArray(selectedPoints)&&selectedPoints.length===8,
      "expected eight projected exploratory cases apart from the development cloud");
    for(const point of [...map.points,...map.centres,...selectedPoints])
      insist(finite(point.x)&&finite(point.y),"invalid projected point");
    for(const point of [...map.points,...selectedPoints])
      insist(Array.isArray(point.memberships)&&point.memberships.length===3&&
        point.memberships.every(v=>finite(v)&&v>=0&&v<=1)&&
        near(point.memberships.reduce((a,b)=>a+b,0),1),"invalid membership");
    insist(Array.isArray(data.cases)&&data.cases.length===8,"expected eight development example stages or exploratory test cases");
    const pairs=new Set();
    for(const obs of data.cases){
      insist((exploratory ? ["P06","P07"].includes(obs.example) :
        /^Development example [12]$/.test(obs.example))&&["0","2","6","10"].includes(obs.stage),"invalid case");
      const pair=obs.example+"|"+obs.stage;insist(!pairs.has(pair),"duplicate case");pairs.add(pair);
      insist(finite(obs.originalPrediction)&&Array.isArray(obs.measurements)&&obs.measurements.length===4&&
        obs.measurements.every(finite),"missing measurements");
      insist(Number.isInteger(obs.mapPointIndex)&&obs.mapPointIndex>=0&&
        obs.mapPointIndex<selectedPoints.length,"invalid selected point");
      const m=obs.mamdani,s=obs.sugeno;
      insist(m&&["OK","Unavailable"].includes(m.status)&&m.outputTerms===5&&
        route(data.tree,obs.measurements).leaf===m.crispLeaf&&
        Array.isArray(m.rules)&&m.rules.length===8,"invalid Mamdani case");
      insist(new Set(m.rules.map(r=>r.leaf)).size===8&&m.rules.every(r=>ids.has(r.leaf)&&finite(r.activation)&&
        r.activation>=0&&r.activation<=1&&typeof r.consequent==="string"&&typeof r.antecedent==="string"),"invalid rule strengths");
      if(m.status==="OK")insist(finite(m.prediction)&&m.outputCurve&&
        near(centroid(m.outputCurve.x,m.outputCurve.y),m.prediction,2e-4),"Mamdani centroid mismatch");
      else insist(m.prediction===null,"unavailable Mamdani prediction is not null");
      insist(s&&s.status==="OK"&&finite(s.prediction)&&Array.isArray(s.memberships)&&
        s.memberships.length===3&&Array.isArray(s.contributions)&&s.contributions.length===3,"invalid Sugeno case");
      s.memberships.forEach((u,i)=>insist(finite(u)&&u>=0&&u<=1&&
        near(u,selectedPoints[obs.mapPointIndex].memberships[i])&&
        near(s.contributions[i],u*data.profiles[i].constant),"Sugeno membership/contribution mismatch"));
      insist(near(s.memberships.reduce((a,b)=>a+b,0),1)&&
        near(s.contributions.reduce((a,b)=>a+b,0),s.prediction),"Sugeno weighted sum mismatch");
    }
    for(const label of (exploratory?["P06","P07"]:["Development example 1","Development example 2"]))
      for(const stage of ["0","2","6","10"])insist(pairs.has(label+"|"+stage),"missing stage in development case");
    return data;
  }
  function cards(id,obs,method){
    const x=obs[method],labels=["Regression model predicted difficulty level",
      "Surrogate model predicted difficulty level","Difference"];
    const values=[f(obs.originalPrediction),x.status==="OK"?f(x.prediction):"Unavailable",
      x.status==="OK"?(x.prediction<obs.originalPrediction?"−":"+")+f(Math.abs(x.prediction-obs.originalPrediction)):"—"];
    document.getElementById(id).replaceChildren(...labels.map((label,i)=>{
      const b=html("div","cd-score");b.append(html("small","",label),html("strong","",values[i]));return b;
    }));
  }
  function treeChart(tree,obs){
    const active=new Set(route(tree,obs.measurements).path);let leaves=0,depthMax=0;
    function place(n,depth){depthMax=Math.max(depthMax,depth);
      if(Object.hasOwn(n,"leaf"))return {node:n,x:90+leaves++*140,y:45+depth*110};
      const left=place(n.left,depth+1),right=place(n.right,depth+1);
      return {node:n,x:(left.x+right.x)/2,y:45+depth*110,left,right};
    }
    const root=place(tree,0),chart=svg("svg",{viewBox:"0 0 "+Math.max(1200,leaves*140+180)+" "+(115+depthMax*110),
      role:"img","aria-label":"Eight-leaf decision tree with selected crisp path highlighted"});
    function links(p){for(const q of [p.left,p.right].filter(Boolean)){
      const chosen=active.has(p.node)&&active.has(q.node);
      chart.append(svg("path",{d:"M"+p.x+" "+(p.y+22)+" V"+((p.y+q.y)/2)+" H"+q.x+" V"+(q.y-22),
        fill:"none",stroke:chosen?"#714bc3":"#bdc8d3","stroke-width":chosen?4:2}));
      links(q);
    }}
    function nodes(p){const leaf=Object.hasOwn(p.node,"leaf"),chosen=active.has(p.node);
      const box=svg("rect",{x:p.x-62,y:p.y-23,width:124,height:46,rx:9,
        fill:chosen?"#eee7ff":"#f3f6fa",stroke:chosen?"#7048bd":"#c4d0dc","stroke-width":chosen?2.5:1});
      const name=leaf?"Leaf "+p.node.leaf:LABELS[KEYS.indexOf(p.node.feature)];
      const value=leaf?"Tree "+f(p.node.value):"≤ "+f(p.node.threshold);
      box.append(svg("title",{},leaf?"Crisp tree leaf, not the Mamdani prediction":
        name+"; threshold "+f(p.node.threshold)+"; observed "+f(obs.measurements[KEYS.indexOf(p.node.feature)])));
      chart.append(box,svg("text",{x:p.x,y:p.y-4,"text-anchor":"middle","font-size":13,"font-weight":600,fill:"#26354a"},name),
        svg("text",{x:p.x,y:p.y+13,"text-anchor":"middle","font-size":12,fill:"#52677c"},value));
      if(p.left)nodes(p.left);if(p.right)nodes(p.right);
    }
    links(root);nodes(root);return chart;
  }
  function ruleList(obs){
    const rules=obs.mamdani.rules.filter(r=>r.activation>0).sort((a,b)=>b.activation-a.activation);
    if(!rules.length)return [html("p","hint","No activated rules. Mamdani result unavailable.")];
    return rules.map(rule=>{
      const item=html("details","cd-rule"),head=html("summary");
      head.append(html("span",null,"Leaf "+rule.leaf+" · "+linguisticLabel(rule.consequent)),
        html("strong",null,f(rule.activation,2)));
      const track=html("div","cd-track"),fill=html("div","cd-fill");
      fill.style.width=(rule.activation*100)+"%";track.append(fill);
      item.append(head,track,html("p","cd-rule-text",linguisticLabel(rule.antecedent)));
      return item;
    });
  }
  function outputChart(obs){
    if(obs.mamdani.status!=="OK")
      return html("p","hint","Unavailable: no valid aggregated fuzzy output.");
    const curve=obs.mamdani.outputCurve,x=curve.x,y=curve.y;
    const original=obs.originalPrediction,centroidScore=obs.mamdani.prediction;
    const bounds=plotBounds(x,original,centroidScore);
    const X=v=>48+694*(v-bounds.lower)/(bounds.upper-bounds.lower);
    const Y=v=>202-137*v;
    const xa=X(original),xb=X(centroidScore);
    const chart=svg("svg",{viewBox:"0 0 790 273",role:"img",
      "aria-label":"Combined fuzzy membership shape with original regression prediction "+
        f(original)+" and Mamdani centroid "+f(centroidScore)+
        ". Approximation difference "+(centroidScore>=original?"+":"−")+
        f(Math.abs(centroidScore-original))+" predicted difficulty level points."});
    chart.append(svg("title",{},
      "Mamdani aggregation (purple), original regression (grey) and centroid (blue)"));
    chart.append(svg("line",{x1:48,y1:202,x2:742,y2:202,stroke:"#94a5b8","stroke-width":1.5}));
    chart.append(svg("text",{x:28,y:76,"text-anchor":"middle",fill:"#63788b","font-size":11},"1"));
    chart.append(svg("text",{x:28,y:204,"text-anchor":"middle",fill:"#63788b","font-size":11},"0"));
    // A light band makes the magnitude and direction of surrogate disagreement visible.
    const gap=Math.abs(xb-xa);
    if(gap>0.25)chart.append(svg("rect",{
      x:Math.min(xa,xb),y:57,width:gap,height:145,
      fill:"#c7d8f5",opacity:".42"
    }));
    const stride=Math.max(1,Math.floor((x.length-1)/350)),coords=[];
    for(let i=0;i<x.length;i+=stride)
      coords.push([X(x[i]),Y(y[i])]);
    if((x.length-1)%stride!==0)
      coords.push([X(x[x.length-1]),Y(y[y.length-1])]);
    const raw=coords.map(p=>p[0].toFixed(2)+","+p[1].toFixed(2)).join(" ");
    chart.append(svg("polygon",{
      points:X(x[0]).toFixed(2)+",202 "+raw+" "+
        X(x[x.length-1]).toFixed(2)+",202",
      fill:"#7850c8","fill-opacity":".13"
    }));
    chart.append(svg("polyline",{points:raw,fill:"none",stroke:"#7147bf",
      "stroke-width":3,"stroke-linejoin":"round"}));
    // Labels intentionally use different vertical lanes for nearby predictions.
    chart.append(svg("line",{x1:xa,y1:31,x2:xa,y2:202,stroke:"#687a90",
      "stroke-width":2,"stroke-dasharray":"5 5"}));
    chart.append(svg("line",{x1:xb,y1:51,x2:xb,y2:202,stroke:"#173f6c",
      "stroke-width":2.5,"stroke-dasharray":"7 4"}));
    chart.append(svg("text",{x:Math.max(98,Math.min(690,xa)),
      y:23,"text-anchor":"middle",fill:"#586a80","font-size":14,
      "font-weight":600},"Regression "+f(original)));
    chart.append(svg("text",{x:Math.max(98,Math.min(690,xb)),
      y:48,"text-anchor":"middle",fill:"#173f6c","font-size":14,
      "font-weight":700},"Centroid "+f(centroidScore)));
    for(let i=0;i<=4;i++){
      const value=bounds.lower+(bounds.upper-bounds.lower)*i/4;
      const px=X(value);
      chart.append(svg("line",{x1:px,y1:202,x2:px,y2:208,stroke:"#94a5b8"}));
      chart.append(svg("text",{x:px,y:228,"text-anchor":"middle",fill:"#63788b",
        "font-size":12},f(value,2)));
    }
    chart.append(svg("text",{x:395,y:257,"text-anchor":"middle",
      fill:"#425e77","font-size":14,"font-weight":600},"Predicted difficulty level"));
    return chart;
  }
  function scatter(map,obs){
    const current=map.selectedPoints?map.selectedPoints[obs.mapPointIndex]:map.points[obs.mapPointIndex];
    const all=[...map.points,...map.centres,current],xmin=Math.min(...all.map(p=>p.x)),xmax=Math.max(...all.map(p=>p.x)),
      ymin=Math.min(...all.map(p=>p.y)),ymax=Math.max(...all.map(p=>p.y));
    const X=v=>60+670*(v-xmin)/Math.max(xmax-xmin,1e-8);
    const Y=v=>420-360*(v-ymin)/Math.max(ymax-ymin,1e-8);
    const chart=svg("svg",{viewBox:"0 0 790 475",role:"img",
      "aria-label":"PCA overview fitted on 104 development Auditory observations, three training cluster centres and the selected session"});
    chart.append(svg("line",{x1:60,y1:420,x2:730,y2:420,stroke:"#9baabb"}),
      svg("line",{x1:60,y1:60,x2:60,y2:420,stroke:"#9baabb"}));
    map.points.forEach((p,i)=>{
      const dominant=p.memberships.indexOf(Math.max(...p.memberships));
      const isSelected=!map.selectedPoints&&i===obs.mapPointIndex;
      const dot=svg("circle",{cx:X(p.x),cy:Y(p.y),r:isSelected?10:5,
        fill:isSelected?"#172d56":COLORS[dominant],opacity:isSelected?1:.52,
        stroke:isSelected?"#ffffff":"none","stroke-width":isSelected?3:0});
      dot.append(svg("title",{},isSelected?"Selected development session":"Anonymised development observation"));
      chart.append(dot);
    });
    if(map.selectedPoints){
      chart.append(svg("circle",{cx:X(current.x),cy:Y(current.y),r:13,fill:"#172d56",
        stroke:"#ffffff","stroke-width":4}));
      chart.append(svg("text",{x:X(current.x)+14,y:Y(current.y)-11,
        fill:"#172d56","font-weight":700,"font-size":14},obs.example+" · exploratory test case"));
    }
    map.centres.forEach((p,i)=>{
      chart.append(svg("circle",{cx:X(p.x),cy:Y(p.y),r:14,fill:"#fff",stroke:COLORS[i],"stroke-width":3}),
        svg("text",{x:X(p.x),y:Y(p.y)+5,"text-anchor":"middle","font-weight":700,"font-size":15,fill:COLORS[i]},String(i+1)));
    });
    chart.append(svg("text",{x:395,y:464,"text-anchor":"middle","font-size":13,fill:"#63788b"},
      "PC1 · "+(100*map.explainedVariance[0]).toFixed(1)+"% variance"),
      svg("text",{x:17,y:238,transform:"rotate(-90 17 238)","text-anchor":"middle","font-size":13,fill:"#63788b"},
        "PC2 · "+(100*map.explainedVariance[1]).toFixed(1)+"% variance"));
    return chart;
  }
  function profileDetails(target,obs,profile,features){
    target.replaceChildren(html("h3",null,"Profile "+profile.cluster+" · measurement comparison"),
      html("p","hint","Session value / learned profile centre (original units)"));
    for(let i=0;i<4;i++){
      const row=html("div","cd-feature");
      row.append(html("span",null,LABELS[i]),html("strong",null,
        f(obs.measurements[i])+" "+UNITS[i]+" / "+f(profile.centre[i])+" "+UNITS[i]));
      const track=html("div","cd-feature-track"),dot1=html("span","cd-feature-dot"),
        dot2=html("span","cd-feature-dot cd-feature-centre");
      dot1.style.left=(100*Math.min(1,obs.measurements[i]/features[i].maximum))+"%";
      dot2.style.left=(100*Math.min(1,profile.centre[i]/features[i].maximum))+"%";
      track.append(dot1,dot2);row.append(track);target.append(row);
    }
    target.append(html("p","cd-note","Solid = session · outline = cluster centre. Each feature has its own scale."));
  }
  function show(data,example,stage,method){
    const obs=data.cases.find(r=>r.example===example&&r.stage===stage);
    insist(obs,"observation not found");
    const which=obs[method];
    const fields=[["Regression model predicted difficulty level",f(obs.originalPrediction)],
      ["Surrogate model predicted difficulty level",which.status==="OK"?f(which.prediction):"Unavailable"],
      ["Difference",which.status==="OK"?(which.prediction>=obs.originalPrediction?"+":"−")+
        f(Math.abs(which.prediction-obs.originalPrediction)):"—"]];
    const scores=document.getElementById("cd-"+method+"-scores");
    scores.replaceChildren(...fields.map(([name,value])=>{
      const card=html("div","cd-score");card.append(html("small",null,name),html("strong",null,value));return card;
    }));
    if(method==="mamdani"){
      document.getElementById("cd-tree").replaceChildren(treeChart(data.tree,obs));
      currentTreeZoom=1;
      zoomTree("reset");
      const recorded=document.getElementById("cd-recorded-rows");
      recorded.replaceChildren(...obs.measurements.map((value,i)=>{
        const row=html("tr");
        row.append(html("td",null,LABELS[i]),html("td",null,f(value)+" "+UNITS[i]));
        return row;
      }));
      document.getElementById("cd-rule-list").replaceChildren(...ruleList(obs));
      document.getElementById("cd-curve").replaceChildren(outputChart(obs));
      const diff=document.getElementById("cd-curve-difference");
      diff.textContent=obs.mamdani.status==="OK"?
        "Approximation difference: "+(obs.mamdani.prediction>=obs.originalPrediction?"+":"−")+
        f(Math.abs(obs.mamdani.prediction-obs.originalPrediction)):"Difference unavailable";
    } else {
      document.getElementById("cd-scatter").replaceChildren(scatter(data.clusterMap,obs));
      const list=document.getElementById("cd-memberships"),detail=document.getElementById("cd-profile-detail");
      const buttons=[],winner=obs.sugeno.memberships.indexOf(Math.max(...obs.sugeno.memberships));
      data.profiles.forEach((profile,i)=>{
        const b=html("button","cd-membership");b.type="button";
        b.setAttribute("aria-pressed",String(i===winner));
        const row=html("span","cd-membership-head");
        row.append(html("span",null,"Profile "+(i+1)),
          html("strong",null,(100*obs.sugeno.memberships[i]).toFixed(1)+"%"));
        const track=html("span","cd-track"),fill=html("span","cd-fill");
        fill.style.width=(100*obs.sugeno.memberships[i])+"%";fill.style.background=COLORS[i];
        track.append(fill);b.append(row,track);
        b.addEventListener("click",()=>{
          buttons.forEach(btn=>btn.setAttribute("aria-pressed","false"));
          b.setAttribute("aria-pressed","true");profileDetails(detail,obs,profile,data.features);
        });buttons.push(b);
      });
      list.replaceChildren(...buttons);profileDetails(detail,obs,data.profiles[winner],data.features);
      const calcs=document.getElementById("cd-calculation");
      calcs.replaceChildren(...data.profiles.map((profile,i)=>{
        const r=html("div","cd-term");r.append(html("span",null,"Profile "+(i+1)),
          html("strong",null,f(obs.sugeno.memberships[i])+" × "+f(profile.constant)+
          " = "+f(obs.sugeno.contributions[i])));return r;
      }),html("div","cd-total","Combined prediction = "+f(obs.sugeno.prediction)));
    }
  }
  const API={validate,route,centroid,show,zoomTree,zoomStep,plotBounds,linguisticLabel,outputChart};
  if(typeof module!=="undefined"&&module.exports)module.exports=API;
  if(root)root.AuditoryCD=API;
})(typeof window!=="undefined"?window:null);
