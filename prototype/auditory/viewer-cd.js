"use strict";
/* Development-only C/D viewer. No fitting, browser upload or test-person inference. */
(function (root) {
  const NS = "http://www.w3.org/2000/svg";
  const KEYS = ["mental_demand_score_0_to_10","median_time_between_qualifying_grabs_seconds","median_time_to_target_seconds","median_head_turning_degrees"];
  const LABELS = ["Mental demand","Grab interval","Time to target","Head turning"];
  const UNITS = ["/10","s","s","°"];
  const COLORS = ["#3576a5","#228d85","#ae7c35"];
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
    insist(data&&data.schema==="auditory-cd-development-v1"&&data.population==="development"&&
      data.reservedTestParticipantsUsed===0&&data.participants===26&&data.observations===104,"incorrect development provenance");
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
    for(const point of [...map.points,...map.centres])insist(finite(point.x)&&finite(point.y),"invalid projected point");
    for(const point of map.points)insist(Array.isArray(point.memberships)&&point.memberships.length===3&&
      point.memberships.every(v=>finite(v)&&v>=0&&v<=1)&&near(point.memberships.reduce((a,b)=>a+b,0),1),"invalid membership");
    insist(Array.isArray(data.cases)&&data.cases.length===8,"expected eight development example stages");
    const pairs=new Set();
    for(const obs of data.cases){
      insist(/^Development example [12]$/.test(obs.example)&&["0","2","6","10"].includes(obs.stage),"invalid case");
      const pair=obs.example+"|"+obs.stage;insist(!pairs.has(pair),"duplicate case");pairs.add(pair);
      insist(finite(obs.originalPrediction)&&Array.isArray(obs.measurements)&&obs.measurements.length===4&&
        obs.measurements.every(finite),"missing measurements");
      insist(Number.isInteger(obs.mapPointIndex)&&obs.mapPointIndex>=0&&obs.mapPointIndex<104,"invalid selected point");
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
        near(u,map.points[obs.mapPointIndex].memberships[i])&&
        near(s.contributions[i],u*data.profiles[i].constant),"Sugeno membership/contribution mismatch"));
      insist(near(s.memberships.reduce((a,b)=>a+b,0),1)&&
        near(s.contributions.reduce((a,b)=>a+b,0),s.prediction),"Sugeno weighted sum mismatch");
    }
    for(const label of ["Development example 1","Development example 2"])
      for(const stage of ["0","2","6","10"])insist(pairs.has(label+"|"+stage),"missing stage in development case");
    return data;
  }
  function cards(id,obs,method){
    const x=obs[method],labels=["Original regression",method==="mamdani"?"Mamdani score":"Sugeno score","Difference"];
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
      head.append(html("span",null,"Leaf "+rule.leaf+" · "+rule.consequent),
        html("strong",null,f(rule.activation,2)));
      const track=html("div","cd-track"),fill=html("div","cd-fill");
      fill.style.width=(rule.activation*100)+"%";track.append(fill);
      item.append(head,track,html("p","cd-rule-text",rule.antecedent));
      return item;
    });
  }
  function outputChart(obs){
    if(obs.mamdani.status!=="OK")return html("p","hint","Unavailable: no valid aggregated fuzzy output.");
    const curve=obs.mamdani.outputCurve,x=curve.x,y=curve.y,lo=x[0],hi=x[x.length-1];
    const X=v=>40+700*(v-lo)/(hi-lo),Y=v=>170-135*v;
    const chart=svg("svg",{viewBox:"0 0 790 205",role:"img",
      "aria-label":"Aggregated fuzzy output and centroid"});
    chart.append(svg("line",{x1:40,y1:170,x2:740,y2:170,stroke:"#9baabb"}));
    const step=Math.max(1,Math.floor(x.length/350)),points=[];
    for(let i=0;i<x.length;i+=step)points.push(X(x[i]).toFixed(1)+","+Y(y[i]).toFixed(1));
    points.push(X(x[x.length-1]).toFixed(1)+","+Y(y[y.length-1]).toFixed(1));
    chart.append(svg("polyline",{points:points.join(" "),fill:"none",stroke:"#7048bd","stroke-width":3}));
    const cp=X(obs.mamdani.prediction);
    chart.append(svg("line",{x1:cp,y1:27,x2:cp,y2:170,stroke:"#223f65","stroke-width":2,"stroke-dasharray":"6 4"}),
      svg("text",{x:cp+6,y:24,fill:"#223f65","font-weight":700,"font-size":16},f(obs.mamdani.prediction)),
      svg("text",{x:40,y:195,"font-size":12,fill:"#617589"},f(lo,2)),
      svg("text",{x:740,y:195,"text-anchor":"end","font-size":12,fill:"#617589"},f(hi,2)));
    return chart;
  }
  function scatter(map,obs){
    const all=[...map.points,...map.centres],xmin=Math.min(...all.map(p=>p.x)),xmax=Math.max(...all.map(p=>p.x)),
      ymin=Math.min(...all.map(p=>p.y)),ymax=Math.max(...all.map(p=>p.y));
    const X=v=>60+670*(v-xmin)/Math.max(xmax-xmin,1e-8);
    const Y=v=>420-360*(v-ymin)/Math.max(ymax-ymin,1e-8);
    const chart=svg("svg",{viewBox:"0 0 790 475",role:"img",
      "aria-label":"Development-only PCA overview of 104 Auditory observations and three cluster centres"});
    chart.append(svg("line",{x1:60,y1:420,x2:730,y2:420,stroke:"#9baabb"}),
      svg("line",{x1:60,y1:60,x2:60,y2:420,stroke:"#9baabb"}));
    map.points.forEach((p,i)=>{
      const dominant=p.memberships.indexOf(Math.max(...p.memberships)),current=i===obs.mapPointIndex;
      const dot=svg("circle",{cx:X(p.x),cy:Y(p.y),r:current?10:5,fill:current?"#172d56":COLORS[dominant],
        opacity:current?1:.52,stroke:current?"white":"none","stroke-width":current?3:0});
      dot.append(svg("title",{},current?"Selected development session":"Anonymised development observation"));
      chart.append(dot);
    });
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
    const title=method==="mamdani"?"Mamdani score":"Sugeno score";
    const fields=[["Original regression",f(obs.originalPrediction)],[title,which.status==="OK"?f(which.prediction):"Unavailable"],
      ["Difference",which.status==="OK"?(which.prediction>=obs.originalPrediction?"+":"−")+
        f(Math.abs(which.prediction-obs.originalPrediction)):"—"]];
    const scores=document.getElementById("cd-"+method+"-scores");
    scores.replaceChildren(...fields.map(([name,value])=>{
      const card=html("div","cd-score");card.append(html("small",null,name),html("strong",null,value));return card;
    }));
    document.getElementById("cd-case-caption").textContent=example+" · Stage "+["0","2","6","10"].indexOf(stage)+
      " (D"+stage+") · Development fit, Auditory";
    if(method==="mamdani"){
      document.getElementById("cd-tree").replaceChildren(treeChart(data.tree,obs));
      document.getElementById("cd-rule-list").replaceChildren(...ruleList(obs));
      document.getElementById("cd-curve").replaceChildren(outputChart(obs));
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
  const API={validate,route,centroid,show};
  if(typeof module!=="undefined"&&module.exports)module.exports=API;
  if(root)root.AuditoryCD=API;
})(typeof window!=="undefined"?window:null);
