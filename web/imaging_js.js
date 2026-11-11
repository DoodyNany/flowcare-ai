/* ---------- 4 · chest x-ray ---------- */
var XN=new FC.XrayNet(DATA.xray), XS=DATA.samples, XTHR=DATA.xray.threshold, XSZ=DATA.xray.size||64;
var xitems=XS.images.map(function(px,k){return {id:"XR-"+(1041+k),pix:px,label:XS.labels[k],res:null,dec:null,arr:0};});
(function(){var order=[4,5,0,6,7,11,8,1,9,2,10,3].filter(function(k){return k<xitems.length;});order.forEach(function(k,n){xitems[k].arr=7*60+30+n*6;});})();
xitems.forEach(function(it){it.res=XN.run(it.pix);});
var xsort="ai", xcur=null, xc=$("#xcanvas"), xp=$("#xplain"), xmag=$("#xmag"), xlastMouse=null;
function hhmm(m){return ("0"+Math.floor(m/60)).slice(-2)+":"+("0"+(m%60)).slice(-2);}
function verdict(p,thr){return p>=Math.min(0.97,thr+0.15)?["crit","Strong pattern"]:p>=thr?["crit","Flagged"]:p>=thr-0.12?["warn","Borderline"]:["ok","No pattern"];}
function xsorted(){var a=xitems.slice();if(xsort==="ai")a.sort(function(p,q){var fp=p.res.flag?1:0,fq=q.res.flag?1:0;return fq-fp||(fp?q.res.prob-p.res.prob:p.arr-q.arr);});else a.sort(function(p,q){return p.arr-q.arr;});return a;}
function xRenderList(){
  var a=xsorted();
  $("#xlist").innerHTML=a.map(function(it,n){var v=verdict(it.res.prob,XTHR);
    return '<button class="wrow'+(it===xcur?" on":"")+'" data-xi="'+xitems.indexOf(it)+'"'+tip(it.id+" · arrived "+hhmm(it.arr),"AI score "+it.res.prob.toFixed(2)+" ("+v[1].toLowerCase()+"). "+(it.dec?"Physician: "+it.dec+".":"Not read yet."))+'>'+
      '<span class="pos">'+(n+1)+'</span><canvas width="40" height="40"></canvas><span class="wmeta"><b>'+it.id+'</b><span class="small muted">'+(it.arr?"arrived "+hhmm(it.arr):"uploaded")+'</span></span>'+
      '<span class="wst"><span class="chip '+v[0]+'">'+(it.res.flag?"Urgent":"Routine")+'</span>'+(it.dec?'<span class="small" style="color:var(--ok)">✓ read</span>':'')+'</span></button>';}).join("");
  $$("#xlist .wrow").forEach(function(b){var it=xitems[+b.dataset.xi];paintGray(it.pix,XSZ,b.querySelector("canvas"),0,100);b.addEventListener("click",function(){xOpen(it);});});
  var fl=xitems.filter(function(it){return it.res.flag&&it.arr;}),ai=xsorted(),save=0;
  if(xsort!=="ai"){xsort="ai";ai=xsorted();xsort="time";}
  var tm=xitems.filter(function(it){return it.arr;}).sort(function(p,q){return p.arr-q.arr;});
  fl.forEach(function(it){save+=tm.indexOf(it)-ai.indexOf(it);});
  $("#xgain").textContent=fl.length?"With AI sorting, the "+fl.length+" flagged X-rays are read on average "+(save/fl.length).toFixed(1)+" places earlier.":"";
}
function paintGray(pix,S,cv,bright,contrast){var im=new ImageData(S,S),c=contrast/100;
  for(var k=0;k<S*S;k++){var v=(pix[k]-128)*c+128+bright;v=v<0?0:v>255?255:v;im.data[4*k]=im.data[4*k+1]=im.data[4*k+2]=v;im.data[4*k+3]=255;}
  var t=document.createElement("canvas");t.width=t.height=S;t.getContext("2d").putImageData(im,0,0);var g=cv.getContext("2d");g.imageSmoothingEnabled=true;g.imageSmoothingQuality="high";g.drawImage(t,0,0,cv.width,cv.height);}
function heat(cam,n,alpha){var mx=0;for(var k=0;k<cam.length;k++)if(cam[k]>mx)mx=cam[k];var t=document.createElement("canvas");t.width=t.height=n;if(mx<=0)return t;
  var im=new ImageData(n,n);for(var q=0;q<n*n;q++){var v=Math.max(0,cam[q])/mx;im.data[4*q]=255;im.data[4*q+1]=Math.round(170*(1-v));im.data[4*q+2]=0;im.data[4*q+3]=Math.round(255*alpha*v*v);}
  t.getContext("2d").putImageData(im,0,0);return t;}
function xDraw(){if(!xcur)return;var b=+$("#xbright").value,c=+$("#xcontrast").value,op=+$("#xop").value/100;
  paintGray(xcur.pix,XSZ,xc,b,c);if($("#xheat").checked){var g=xc.getContext("2d");g.imageSmoothingEnabled=true;g.drawImage(heat(xcur.res.cam,xcur.res.camSize,op),0,0,xc.width,xc.height);}
  var cmp=$("#xcmp").checked;$("#xplainwrap").hidden=!cmp;if(cmp)paintGray(xcur.pix,XSZ,xp,b,c);
  xc.parentNode.querySelector(".vlab").textContent=$("#xheat").checked?"AI heat map":"X-ray";xMag(xlastMouse);}
var xmagSrc=document.createElement("canvas");xmagSrc.width=xmagSrc.height=384;
function xMag(pt){if(xcur)paintGray(xcur.pix,XSZ,xmagSrc,+$("#xbright").value,+$("#xcontrast").value);var g=xmag.getContext("2d"),w=xmag.width,h=xmag.height,src=xmagSrc,cx=pt?pt.x:xc.width/2,cy=pt?pt.y:xc.height/2,s=3;
  g.fillStyle="#000";g.fillRect(0,0,w,h);g.imageSmoothingEnabled=true;g.drawImage(src,cx-w/(2*s),cy-h/(2*s),w/s,h/s,0,0,w,h);}
[xc,xp].forEach(function(cv){cv.addEventListener("mousemove",function(e){var r=cv.getBoundingClientRect();xlastMouse={x:(e.clientX-r.left)*cv.width/r.width,y:(e.clientY-r.top)*cv.height/r.height};xMag(xlastMouse);});});
function xOpen(it){xcur=it;var r=it.res,v=verdict(r.prob,XTHR);
  $("#xtitle").textContent=it.id+(it.arr?" · arrived "+hhmm(it.arr):" · uploaded");
  $("#xscore").textContent=r.prob.toFixed(2);$("#xverdict").className="chip "+v[0];$("#xverdict").textContent=v[1];
  $("#xmark").style.left="calc("+(100*r.prob)+"% - 2px)";
  var pos=xsorted().indexOf(it)+1,t0=performance.now();XN.run(it.pix);var ms=Math.max(1,Math.round(performance.now()-t0));
  $("#xqueue").textContent=(r.flag?"Moved to position "+pos+" on the reading list (urgent).":"Position "+pos+" on the reading list (routine order).")+" Analysed on this computer in "+ms+" ms.";
  if(r.flag)setAnswer("x","crit",it.id+": possible pneumonia. A physician should read this X-ray first","The AI found a pattern often seen in pneumonia (red area). It moved this X-ray up the reading list. It does not diagnose.");
  else if(v[0]==="warn")setAnswer("x","warn",it.id+": borderline, read carefully","The score is just under the flag line, so the AI is unsure. It stays in routine order; the physician decides.");
  else setAnswer("x","ok",it.id+": no pneumonia pattern found","Stays in routine order. A physician still reads every X-ray.");
  $("#xstatus").className="chip "+(it.dec?"ok":"warn");$("#xstatus").textContent=it.dec?"Physician: "+it.dec:"Waiting for physician";
  $("#xtruth").textContent=it.dec&&it.label!=null?"Real answer: "+(it.label?"pneumonia":"normal")+".":"";
  if(!it.logged){it.logged=1;LOG("Chest X-ray","AI","X-ray analysed",it.id+": score "+r.prob.toFixed(3)+" ("+v[1]+"), "+(r.flag?"flagged urgent":"routine")+", reading-list position "+pos);}
  xRenderList();xDraw();}
function xAudit(){var d=xitems.filter(function(it){return it.dec&&it.dec!=="More tests";}),ag=d.filter(function(it){return (it.dec==="Pneumonia")===it.res.flag;}).length;
  $("#xaudit").innerHTML=d.length?'<b>'+ag+' of '+d.length+'</b> physician decisions agreed with the AI flag this session.':'<span class="muted">No X-rays read yet this session.</span>';}
$$("[data-xdec]").forEach(function(b){b.addEventListener("click",function(){if(!xcur)return;var who=$("#xdoc").value.trim()||"Physician";xcur.dec=b.dataset.xdec;
  $("#xstatus").className="chip ok";$("#xstatus").textContent=who+": "+xcur.dec;
  if(xcur.label!=null)$("#xtruth").textContent="Real answer for this test X-ray: "+(xcur.label?"pneumonia":"normal")+". The AI "+(xcur.res.flag===!!xcur.label?"was right.":"was wrong here, which is why a physician always decides.");
  LOG("Chest X-ray","Decision","Physician decision",who+": "+xcur.dec+" for "+xcur.id+" (AI "+(xcur.res.flag?"flagged":"did not flag")+", score "+xcur.res.prob.toFixed(2)+")");
  xRenderList();xAudit();});});
$("#xs-ai").addEventListener("click",function(){xsort="ai";$("#xs-ai").classList.add("on");$("#xs-time").classList.remove("on");xRenderList();});
$("#xs-time").addEventListener("click",function(){xsort="time";$("#xs-time").classList.add("on");$("#xs-ai").classList.remove("on");xRenderList();});
["#xheat","#xcmp","#xbright","#xcontrast","#xop"].forEach(function(s){$(s).addEventListener("input",xDraw);$(s).addEventListener("change",xDraw);});
$("#xreset").addEventListener("click",function(){$("#xbright").value=0;$("#xcontrast").value=100;$("#xop").value=65;xDraw();});
$("#xthrline").style.left=(100*XTHR)+"%";$("#xzone").style.left=(100*(XTHR-0.12))+"%";$("#xzone").style.width=(12)+"%";$("#xthr").textContent="flag at "+XTHR.toFixed(2);
$("#xfile").addEventListener("change",function(e){var f=e.target.files&&e.target.files[0];if(!f)return;var img=new Image(),url=URL.createObjectURL(f);
  img.onload=function(){var cv=document.createElement("canvas");cv.width=cv.height=XSZ;var c=cv.getContext("2d"),s=Math.min(img.width,img.height);
    c.drawImage(img,(img.width-s)/2,(img.height-s)/2,s,s,0,0,XSZ,XSZ);var d=c.getImageData(0,0,XSZ,XSZ).data,pix=new Array(XSZ*XSZ);
    for(var k=0;k<XSZ*XSZ;k++)pix[k]=Math.round(0.299*d[4*k]+0.587*d[4*k+1]+0.114*d[4*k+2]);URL.revokeObjectURL(url);
    var it={id:"Upload "+(xitems.filter(function(i){return !i.arr;}).length+1),pix:pix,label:null,res:XN.run(pix),dec:null,arr:0};xitems.push(it);xOpen(it);};
  img.onerror=function(){setAnswer("x","warn","That file couldn't be opened","Try a PNG or JPG image.");};img.src=url;});
xAudit();xOpen(xsorted()[0]);

/* ---------- 5 · CT ---------- */
var CN=new FC.CTNet(DATA.ct), CS=DATA.ctSamples, CTHR=DATA.ct.threshold, CD=28;
var citems=CS.scans.map(function(v,k){return {id:"CT-"+(2201+k),vox:v,label:CS.labels[k],res:null,cam:null,dec:null};});
var corder=[0,5,1,6,7,2,8,3,9,4].map(function(k){return citems[k];});
var ccur=null, sl={ax:14,co:14,sa:14}, cplay=null;
function camVolume(r){var n=r.camSize,f=n/CD,v=new Float32Array(CD*CD*CD),mx=0;
  function at(z,y,x){z=Math.max(0,Math.min(n-1,z));y=Math.max(0,Math.min(n-1,y));x=Math.max(0,Math.min(n-1,x));return r.cam[(z*n+y)*n+x];}
  for(var z=0;z<CD;z++)for(var y=0;y<CD;y++)for(var x=0;x<CD;x++){
    var fz=(z+.5)*f-.5,fy=(y+.5)*f-.5,fx=(x+.5)*f-.5,z0=Math.floor(fz),y0=Math.floor(fy),x0=Math.floor(fx),dz=fz-z0,dy=fy-y0,dx=fx-x0,s=0;
    for(var a=0;a<2;a++)for(var b=0;b<2;b++)for(var c=0;c<2;c++)s+=at(z0+a,y0+b,x0+c)*(a?dz:1-dz)*(b?dy:1-dy)*(c?dx:1-dx);
    v[(z*CD+y)*CD+x]=s;if(s>mx)mx=s;}
  for(var i=0;i<v.length;i++)v[i]=mx>0?Math.max(0,v[i])/mx:0;return v;}
function cScoreAll(k){if(k>=corder.length){corder.sort(function(p,q){return q.res.prob-p.res.prob;});cOpen(corder[0]);
    LOG("CT scan","AI","Worklist scored",corder.length+" CT scans analysed; "+corder.filter(function(i){return i.res.flag;}).length+" flagged for the radiologist");return;}
  var it=corder[k];it.res=CN.run(it.vox);it.cam=camVolume(it.res);cRenderList();setTimeout(function(){cScoreAll(k+1);},10);}
function cRenderList(){$("#clist").innerHTML=corder.map(function(it){var r=it.res,v=r?verdict(r.prob,CTHR):["plain","…"];
  return '<button class="wrow'+(it===ccur?" on":"")+'" data-ci="'+citems.indexOf(it)+'"'+tip(it.id,(r?"AI score "+r.prob.toFixed(2)+" ("+v[1].toLowerCase()+"). ":"Scoring… ")+(it.dec?"Radiologist: "+it.dec+".":"Not reviewed yet."))+'>'+
    '<canvas width="40" height="40"></canvas><span class="wmeta"><b>'+it.id+'</b><span class="small muted">Chest CT · lung nodule</span></span><span class="wst"><span class="chip '+v[0]+'">'+(r?(r.flag?"Suspicious":"Low"):"…")+'</span>'+(it.dec?'<span class="small" style="color:var(--ok)">✓ reviewed</span>':'')+'</span></button>';}).join("");
  $$("#clist .wrow").forEach(function(b){var it=citems[+b.dataset.ci];drawSlice(b.querySelector("canvas"),it,"mip",0,false);b.addEventListener("click",function(){if(it.res)cOpen(it);});});}
function vox(it,z,y,x){return it.vox[(z*CD+y)*CD+x];}
function drawSlice(cv,it,view,s,withHeat){var im=new ImageData(CD,CD),hm=new ImageData(CD,CD),cam=it.cam;
  for(var r=0;r<CD;r++)for(var c=0;c<CD;c++){var v,h=0,k=r*CD+c;
    if(view==="ax"){v=vox(it,s,r,c);if(cam)h=cam[(s*CD+r)*CD+c];}
    else if(view==="co"){v=vox(it,r,s,c);if(cam)h=cam[(r*CD+s)*CD+c];}
    else if(view==="sa"){v=vox(it,r,c,s);if(cam)h=cam[(r*CD+c)*CD+s];}
    else{v=0;for(var z=0;z<CD;z++){var q=vox(it,z,r,c);if(q>v)v=q;if(cam){var w=cam[(z*CD+r)*CD+c];if(w>h)h=w;}}}
    im.data[4*k]=im.data[4*k+1]=im.data[4*k+2]=v;im.data[4*k+3]=255;hm.data[4*k]=255;hm.data[4*k+1]=Math.round(170*(1-h));hm.data[4*k+2]=0;hm.data[4*k+3]=Math.round(170*h*h);}
  var t=document.createElement("canvas");t.width=t.height=CD;var g=cv.getContext("2d");g.imageSmoothingEnabled=true;g.imageSmoothingQuality="high";
  t.getContext("2d").putImageData(im,0,0);g.drawImage(t,0,0,cv.width,cv.height);
  if(withHeat&&cam){t.getContext("2d").putImageData(hm,0,0);g.drawImage(t,0,0,cv.width,cv.height);}}
function cross(cv,px,py){var g=cv.getContext("2d"),u=cv.width/CD;g.strokeStyle="rgba(255,214,10,.85)";g.lineWidth=1;g.setLineDash([4,4]);g.beginPath();
  if(px!=null){g.moveTo((px+.5)*u,0);g.lineTo((px+.5)*u,cv.height);}if(py!=null){g.moveTo(0,(py+.5)*u);g.lineTo(cv.width,(py+.5)*u);}g.stroke();g.setLineDash([]);}
function cDraw(){if(!ccur)return;var hh=$("#cheat").checked;
  drawSlice($("#cv-ax"),ccur,"ax",sl.ax,hh);cross($("#cv-ax"),sl.sa,sl.co);
  drawSlice($("#cv-co"),ccur,"co",sl.co,hh);cross($("#cv-co"),sl.sa,sl.ax);
  drawSlice($("#cv-sa"),ccur,"sa",sl.sa,hh);cross($("#cv-sa"),sl.co,sl.ax);
  drawSlice($("#cv-mip"),ccur,"mip",0,hh);
  ["ax","co","sa"].forEach(function(k){$("#cs-"+k).value=sl[k];$("#cn-"+k).textContent=(sl[k]+1)+"/28";});}
function hotspot(it){var m=-1,b=0;for(var i=0;i<it.cam.length;i++)if(it.cam[i]>m){m=it.cam[i];b=i;}return {z:Math.floor(b/(CD*CD)),y:Math.floor(b/CD)%CD,x:b%CD};}
function cOpen(it){ccur=it;var r=it.res,v=verdict(r.prob,CTHR),h=hotspot(it);sl={ax:h.z,co:h.y,sa:h.x};
  $("#ctitle").textContent=it.id+" · chest CT, lung nodule";
  $("#cscore").textContent=r.prob.toFixed(2);$("#cverdict").className="chip "+v[0];$("#cverdict").textContent=v[1];$("#cmark").style.left="calc("+(100*r.prob)+"% - 2px)";
  var t0=performance.now();CN.run(it.vox);var ms=Math.max(1,Math.round(performance.now()-t0));
  $("#cinfo").textContent="Views opened at the AI hotspot (slice "+(h.z+1)+"). The 3D scan was analysed on this computer in "+ms+" ms.";
  if(r.flag)setAnswer("c","crit",it.id+": suspicious nodule. A radiologist should review this scan first","The AI found features often seen in malignant nodules (red area). It moves this scan up the list. It does not diagnose cancer.");
  else if(v[0]==="warn")setAnswer("c","warn",it.id+": borderline nodule","The AI is unsure. Follow the hospital's nodule guidelines; the radiologist decides.");
  else setAnswer("c","ok",it.id+": nodule looks low-risk to the AI","Stays in routine order. A radiologist still reviews every scan.");
  $("#cstatus").className="chip "+(it.dec?"ok":"warn");$("#cstatus").textContent=it.dec?"Radiologist: "+it.dec:"Waiting for radiologist";
  $("#ctruth").textContent=it.dec?"Radiologists' rating in the dataset: "+(it.label?"suspicious for malignancy":"benign")+".":"";
  if(!it.logged){it.logged=1;LOG("CT scan","AI","CT analysed",it.id+": suspicion score "+r.prob.toFixed(3)+" ("+v[1]+"), hotspot at slice "+(h.z+1)+", "+(r.flag?"flagged for radiologist":"routine"));}
  cRenderList();cDraw();}
function cAudit(){var d=citems.filter(function(it){return it.dec&&it.dec.indexOf("3 months")<0;}),ag=d.filter(function(it){return (it.dec.indexOf("MDT")>=0)===it.res.flag;}).length;
  $("#caudit").innerHTML=d.length?'<b>'+ag+' of '+d.length+'</b> radiologist decisions agreed with the AI flag this session.':'<span class="muted">No scans reviewed yet this session.</span>';}
$$("[data-cdec]").forEach(function(b){b.addEventListener("click",function(){if(!ccur)return;ccur.dec=b.dataset.cdec;$("#cstatus").className="chip ok";$("#cstatus").textContent="Radiologist: "+ccur.dec;
  $("#ctruth").textContent="Radiologists' rating in the dataset: "+(ccur.label?"suspicious for malignancy":"benign")+". The AI "+(ccur.res.flag===!!ccur.label?"agreed.":"disagreed, which is why a radiologist always decides.");LOG("CT scan","Decision","Radiologist decision",ccur.dec+" for "+ccur.id+" (AI "+(ccur.res.flag?"flagged":"did not flag")+", score "+ccur.res.prob.toFixed(2)+")");cRenderList();cAudit();});});
["ax","co","sa"].forEach(function(k){$("#cs-"+k).addEventListener("input",function(){sl[k]=+this.value;cDraw();});
  var cv=$("#cv-"+k);cv.addEventListener("wheel",function(e){e.preventDefault();sl[k]=Math.max(0,Math.min(CD-1,sl[k]+(e.deltaY>0?1:-1)));cDraw();},{passive:false});
  cv.addEventListener("click",function(e){var r=cv.getBoundingClientRect(),px=Math.floor((e.clientX-r.left)/r.width*CD),py=Math.floor((e.clientY-r.top)/r.height*CD);
    if(k==="ax"){sl.sa=px;sl.co=py;}else if(k==="co"){sl.sa=px;sl.ax=py;}else{sl.co=px;sl.ax=py;}cDraw();});});
$("#cheat").addEventListener("change",cDraw);
$("#chot").addEventListener("click",function(){if(!ccur)return;var h=hotspot(ccur);sl={ax:h.z,co:h.y,sa:h.x};cDraw();});
$("#cplay").addEventListener("click",function(){if(cplay){clearInterval(cplay);cplay=null;$("#cplay").textContent="▶ Scroll slices";return;}
  $("#cplay").textContent="❚❚ Stop";sl.ax=0;cplay=setInterval(function(){sl.ax++;if(sl.ax>=CD){clearInterval(cplay);cplay=null;$("#cplay").textContent="▶ Scroll slices";sl.ax=CD-1;}cDraw();},90);});
$("#cthrline").style.left=(100*CTHR)+"%";$("#czone").style.left=(100*Math.max(0,CTHR-0.12))+"%";$("#czone").style.width="12%";$("#cthr").textContent="flag at "+CTHR.toFixed(2);
setAnswer("c","ok","Analysing "+corder.length+" CT scans on this computer…","The AI scores every scan, then sorts the worklist so suspicious nodules come first.");
cRenderList();cAudit();setTimeout(function(){cScoreAll(0);},50);
