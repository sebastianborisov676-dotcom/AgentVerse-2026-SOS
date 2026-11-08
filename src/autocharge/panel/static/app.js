'use strict';
const $=id=>document.getElementById(id);
let rejectedEntries=new Set(),pendingAdd=-1;
function editorError(text){$('placementStatus').textContent=text;rejectedEntries=new Set([...text.matchAll(/obstacle (\d+)/gi)].map(m=>Number(m[1])-1));editor();const first=[...rejectedEntries].find(i=>manual[i]);if(first!==undefined){$('manualList').selectedIndex=first;$('manualList').onchange?.();}}
const sizes={box:.22,cone:.12,barrier:.35,stone:.06,slope:.5};
let state=null,manual=[],dirty=false,initialized=false,busy=false,playTimer=null,imageURL=null,lastAttempt=null,uiEpoch=0,placeArmed=false,placementPreview=null,lastImageAttempt=null,lastImageCamera=null,lastImageIndex=null;
for(let i=1;i<=11;i++)$('level').add(new Option(i===11?'Level 11 · Master Challenge':`Level ${String(i).padStart(2,'0')}`,i));
function stopPlayback(){if(playTimer){clearInterval(playTimer);playTimer=null;}$('playRecording').textContent='Play recording';}
function message(text,error=false){$('message').textContent=text;$('message').className=error?'error':'success';}
function activeLabel(){return state?.attempt?`Active scene: level ${state.config.level}, seed ${state.config.seed}, ${state.obstacles.length} obstacles (${state.attempt}).`:'No valid physics scene is active.';}
async function command(name,body={}){
  uiEpoch++;
  let response;
  try{response=await fetch('/api/'+name,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
  catch(e){throw new Error(`POST /api/${name}: connection failed (${e.message}). ${activeLabel()} Check that this page and server use the same local URL.`);}
  let result;
  try{result=await response.json();}catch(e){throw new Error(`POST /api/${name}: invalid response (HTTP ${response.status}). ${activeLabel()}`);}
  if(!response.ok)throw new Error(`POST /api/${name} · HTTP ${response.status}: ${result.error||'Command failed'}. Draft not applied. ${activeLabel()}`);
  if(result.status)paint(result);
  return result;
}
function lockControls(){document.querySelectorAll('button,select,input').forEach(el=>{if(busy)el.disabled=true;});}
async function run(action){if(busy){message('A simulator request is still pending; wait for it to finish.',true);return;}busy=true;lockControls();try{await action();}catch(e){message(e.message,true);editorError(e.message);}finally{busy=false;document.querySelectorAll('button,select,input').forEach(el=>el.disabled=false);editor();if(state?.pose)paint(state);}}
function config(){return {level:Number($('level').value),seed:Number($('seed').value),mode:$('mode').value,manual:$('mode').value==='random'?[]:structuredClone(manual),moving_car:$('movingCar').checked,random_start:$('randomStart').checked};}
function syncForm(c){$('level').value=c.level;$('seed').value=c.seed;$('mode').value=c.mode;$('movingCar').checked=c.moving_car;$('randomStart').checked=c.random_start;manual=structuredClone(c.manual||[]);dirty=false;pendingAdd=-1;rejectedEntries.clear();editor();}
async function build(){stopPlayback();message('Building and validating scene geometry, route and fixed cable…');const result=await command('configure',config());state=result;syncForm(result.config);clearPreview();$('placementStatus').textContent='Applied to physics. '+activeLabel();paint(result);message('Applied to physics. '+activeLabel()+' Previous attempts are preserved.');}
function editor(){const allowed=Number($('level').value)>=5;for(const id of ['kind','x','y','angle','add','move','remove','useLayout','placeGround'])$(id).disabled=!allowed||(state?.status==='running');
  if(!allowed){$('mode').value='random';manual=[];}
  $('mode').options[1].disabled=$('mode').options[2].disabled=!allowed;
  $('editorHint').textContent=allowed?'Edits rebuild a paused scene as a new attempt. The previous attempt is preserved for comparison.':'Manual and hybrid placement unlock at Level 5.';
  const selection=$('manualList').selectedIndex;$('manualList').replaceChildren(...manual.map((o,i)=>new Option(`${rejectedEntries.has(i)?"⚠ ":""}${i+1}  ${o.kind.padEnd(7)} ${o.x.toFixed(2)}, ${o.y.toFixed(2)}`,i)));
  $('manualList').selectedIndex=Math.min(selection,manual.length-1);
  const canPlace=allowed&&!busy&&state?.status!=='running'&&!state?.playback&&$('camera').value==='external'&&$('mode').value!=='random';$('placeGround').disabled=!canPlace;$('applyPlacement').disabled=!canPlace||!placementPreview?.valid;
  $('draftStatus').textContent=`Draft: ${manual.length} manual obstacles${dirty?' · NOT APPLIED':''} | active physics: ${state?.obstacles?.length??0}`;
  sliderLabels();
}
function sliderLabels(){ $('xValue').textContent=Number($('x').value).toFixed(2)+' m';$('yValue').textContent=Number($('y').value).toFixed(2)+' m';$('angleValue').textContent=$('angle').value+'°';if(['cone','stone'].includes($('kind').value))$('angle').disabled=true;}
function selected(){return $('manualList').selectedIndex;}
function draftObstacle(){return {kind:$('kind').value,x:Number($('x').value),y:Number($('y').value),size:manual[selected()]?.kind===$('kind').value?manual[selected()].size:sizes[$('kind').value],angle:Number($('angle').value)};}
async function edit(operation,value=draftObstacle()){if(Number($('level').value)<5)throw new Error('Manual placement starts at Level 5');if(state?.status==='running')throw new Error('Pause before editing geometry');if($('mode').value==='random'){manual=structuredClone(state.obstacles);$('mode').value='manual';}
  if(operation==='add'){if(pendingAdd>=0&&pendingAdd<manual.length)manual[pendingAdd]=value;else {pendingAdd=manual.length;manual.push(value);}}else {const i=selected();if(i<0)throw new Error('Select a manual obstacle first');if(operation==='move')manual[i]={...value,size:manual[i].kind===$('kind').value?manual[i].size:sizes[$('kind').value]};else {manual.splice(i,1);if(i===pendingAdd)pendingAdd=-1;else if(i<pendingAdd)pendingAdd--;}}
  dirty=true;editor();await build();
}
for(const id of ['x','y','angle'])$(id).oninput=sliderLabels;
$('kind').onchange=()=>{editor();};
$('manualList').onchange=()=>{const o=manual[selected()];if(!o)return;$('kind').value=o.kind;$('x').value=o.x;$('y').value=o.y;$('angle').value=o.angle;sliderLabels();};
for(const id of ['level','seed','mode','movingCar','randomStart'])$(id).onchange=()=>{dirty=true;editor();message('Scenario settings changed. Build scenario to validate and apply.');};
$('apply').onclick=()=>run(build);
$('newSeed').onclick=()=>run(async()=>{$('seed').value=Number($('seed').value)+1;dirty=true;await build();});
$('add').onclick=()=>run(()=>edit('add'));$('move').onclick=()=>run(()=>edit('move'));$('remove').onclick=()=>run(()=>edit('remove'));
$('useLayout').onclick=()=>{if(!state)return;manual=structuredClone(state.obstacles);pendingAdd=-1;$('mode').value='manual';dirty=true;editor();message('Current layout copied to manual editor. Select an obstacle to move or remove it.');};
$('saveLayout').onclick=()=>run(async()=>{const r=await command('save_layout',{name:'level_'+$('level').value});const blob=new Blob([JSON.stringify(r.layout,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='autocharge-layout.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);message('Saved ACTIVE physics layout ('+r.layout.obstacles.length+' obstacles), not pending draft edits: '+r.path);});
$('loadLayout').onchange=()=>run(async()=>{const file=$('loadLayout').files[0];if(!file)return;const layout=JSON.parse(await file.text());if(!Array.isArray(layout.obstacles))throw new Error('Expected an obstacles array');manual=layout.obstacles;pendingAdd=-1;if('moving_car' in layout)$('movingCar').checked=layout.moving_car;if('random_start' in layout)$('randomStart').checked=layout.random_start;if(Number.isInteger(layout.seed))$('seed').value=layout.seed;$('level').value=Math.max(5,layout.level||Number($('level').value));$('mode').value='manual';dirty=true;editor();await build();});
$('start').onclick=()=>run(async()=>{if(dirty)await build();await command('start');message('Simulation running. Navigation uses range-limited ray observations.');});
for(const name of ['pause','resume','live','vehicle_motion','save_run']){const ids={vehicle_motion:'vehicle',save_run:'saveRun'};$(ids[name]||name).onclick=()=>run(async()=>{if(name==='live'||name==='resume')stopPlayback();const r=await command(name);message(r.path?'Recording saved: '+r.path:name==='vehicle_motion'?'Vehicle request processed; check interlock and mission events.':name==='live'?'Returned to live state, paused.':name+' command applied.');});}
$('restart').onclick=()=>run(async()=>{stopPlayback();const r=await command('restart');syncForm(r.config);paint(r);message('Identical scenario rebuilt with the same seed. Press Start.');});
const cameraNames={external:'External perspective',robot_rgb:'Robot RGB camera',depth:'Robot depth · metric 0–5 m',connector_cam:'Connector camera',connector_side:'Connector side · operator'};
$('camera').onchange=()=>run(async()=>{clearPreview();await command('camera',{camera:$('camera').value});$('viewTitle').textContent=cameraNames[$('camera').value];$('viewCaption').textContent=$('camera').value==='depth'?'Metric depth: white = near, black = 5 m or farther':$('camera').value==='external'?'Drag to orbit · scroll to zoom · blue markers show the planned route':$('camera').value==='connector_side'?'Operator view of the physical plug tip, socket opening and gap — perception camera unchanged':'Rendered from the physical robot camera pose';});
let drag=null,orbit={azimuth:100,elevation:-25,distance:5.7},orbitSending=false;
function imageArea(){const r=$('orbitArea').getBoundingClientRect();const scale=Math.min(r.width/900,r.height/540);return {rect:r,width:900*scale,height:540*scale,left:(r.width-900*scale)/2,top:(r.height-540*scale)/2};}
function drawPreview(){const canvas=$('placementCanvas'),area=imageArea(),ratio=window.devicePixelRatio||1;canvas.width=area.rect.width*ratio;canvas.height=area.rect.height*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);if(!placementPreview)return;const points=placementPreview.footprint_uv;ctx.beginPath();points.forEach(([u,v],i)=>{const x=area.left+u*area.width,y=area.top+v*area.height;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.closePath();ctx.fillStyle=placementPreview.valid?'#378bd750':'#d3434350';ctx.strokeStyle=placementPreview.valid?'#83ccff':'#ff7979';ctx.lineWidth=2;ctx.fill();ctx.stroke();}
function clearPreview(){placementPreview=null;$('applyPlacement').disabled=true;drawPreview();}
async function previewGround(event){
 if(state.status==='running'||state.playback)throw new Error('Pause and return to live view before ground placement');
 const a=imageArea(),u=(event.clientX-a.rect.x-a.left)/a.width,v=(event.clientY-a.rect.y-a.top)/a.height;
 const result=await command('preview_placement',{u,v,mode:$('mode').value,obstacle:draftObstacle(),selected:-1,manual:structuredClone(manual)});
 placementPreview=result;$('x').value=result.obstacle.x;$('y').value=result.obstacle.y;sliderLabels();
 $('applyPlacement').disabled=!result.valid;
 $('placementStatus').textContent=(result.valid?'Preview only — not in physics. ':'Rejected preview: '+result.reason+'. ')+`Robot edge: ${result.clearance_m.robot.toFixed(3)} m; connector edge: ${result.clearance_m.connector.toFixed(3)} m; required ${result.required_clearance_m.toFixed(2)} m.`;
 if(!result.valid)editorError($('placementStatus').textContent);
 drawPreview();
}
$('placeGround').onclick=()=>{placeArmed=!placeArmed;$('placeGround').textContent='Click ground: '+(placeArmed?'on':'off');$('placeGround').setAttribute('aria-pressed',String(placeArmed));clearPreview();message(placeArmed?'Click the ground to preview the selected obstacle; drag still orbits. Apply preview rebuilds and validates the reset scene.':'Ground placement off.');};
$('applyPlacement').onclick=()=>run(async()=>{if(!placementPreview?.valid)return;await edit('add',placementPreview.obstacle);clearPreview();});
async function sendOrbit(){if(orbitSending)return;orbitSending=true;try{await command('camera',{camera:'external',...orbit});}catch(e){message(e.message,true);}finally{orbitSending=false;}}
$('orbitArea').onpointerdown=e=>{if($('camera').value!=='external')return;drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,moved:false};e.currentTarget.setPointerCapture(e.pointerId);};
$('orbitArea').onpointermove=e=>{if(!drag)return;if(Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)>4)drag.moved=true;if(!drag.moved)return;clearPreview();orbit.azimuth+=(e.clientX-drag.x)*.4;orbit.elevation=Math.max(-85,Math.min(-5,orbit.elevation+(e.clientY-drag.y)*.3));drag={...drag,x:e.clientX,y:e.clientY};sendOrbit();};
$('orbitArea').onpointerup=e=>{if(!drag)return;const click=!drag.moved;drag=null;if(click&&placeArmed)run(()=>previewGround(e));else sendOrbit();};
$('orbitArea').onpointercancel=()=>{drag=null;};
$('orbitArea').addEventListener('wheel',e=>{if($('camera').value!=='external')return;e.preventDefault();clearPreview();orbit.distance=Math.max(1,Math.min(12,orbit.distance+e.deltaY*.004));sendOrbit();},{passive:false});
window.addEventListener('resize',drawPreview);
function playbackID(){return $('attemptB').value||state?.attempt;}
async function seek(index){const r=await command('playback',{id:playbackID(),index:Number(index)});$('timeline').value=index;return r;}
$('timeline').oninput=()=>run(()=>seek($('timeline').value));
$('playRecording').onclick=()=>{if(playTimer){clearInterval(playTimer);playTimer=null;$('playRecording').textContent='Play recording';return;} $('playRecording').textContent='Stop playback';playTimer=setInterval(()=>{const next=Number($('timeline').value)+1;if(next>Number($('timeline').max)){$('playRecording').click();return;}run(()=>seek(next));},150);};
$('restore').onclick=()=>run(async()=>{stopPlayback();const r=await command('restore',{id:playbackID(),index:Number($('checkpointSelect').value||$('timeline').value)});syncForm(r.config);paint(r);message('Restored complete in-process checkpoint at '+r.time.toFixed(3)+' s. A new branch attempt preserves the original future.');});
$('attemptB').onchange=()=>run(()=>seek(0));
$('checkpointSelect').onchange=()=>{if($('checkpointSelect').value!=='')run(()=>seek($('checkpointSelect').value));};
$('compare').onclick=()=>run(async()=>{const r=await command('compare',{a:$('attemptA').value,b:$('attemptB').value});const summary=o=>({phase:o.phase,time:o.time,cable_peak_N:o.cable?.peak_tension_N,strain_relief_peak_N:o.cable?.peak_strain_relief_N,replans:o.replans,obstacle_contacts:o.contacts?.pair_episodes?.obstacle_robot,failure:o.failure});$('comparison').hidden=false;$('comparison').textContent=JSON.stringify({A:{config:r.config_a,result:summary(r.outcome_a)},B:{config:r.config_b,result:summary(r.outcome_b)}},null,2);});
function paint(s){state=s;if(s.status==='loading')return;if(!initialized&&s.config){syncForm(s.config);initialized=true;}if(!s.pose){if(s.error)message(s.error,true);return;}
 const shown=s.playback?.telemetry||s;
 if(s.view_camera?.view&&!drag&&!orbitSending)orbit={...s.view_camera.view};
 if(s.camera){$('camera').value=s.camera;$('viewTitle').textContent=cameraNames[s.camera];}
 $('phase').textContent=shown.phase.replaceAll('_',' ').toUpperCase();$('simTime').textContent='t = '+shown.time.toFixed(2)+' s';$('basePose').textContent=shown.pose[0].toFixed(2)+' / '+shown.pose[1].toFixed(2)+' m';$('heading').textContent='Heading '+(shown.pose[2]*180/Math.PI).toFixed(1)+'°';
 $('tension').textContent=shown.tension.toFixed(1);$('strain').textContent=shown.strain_relief.toFixed(1);$('tensionMeter').value=shown.tension;$('strainMeter').value=shown.strain_relief;
 $('detections').textContent=shown.detected_cells+' cells';$('replans').textContent=shown.replans+' route changes · '+shown.ray_hits+' ray returns';$('missionPhase').textContent=shown.phase;$('portVelocity').textContent=(shown.port_speed*1000).toFixed(1)+' mm/s';$('alignment').textContent=shown.alignment.lateral_mm.toFixed(2)+' mm / '+shown.alignment.angle_deg.toFixed(2)+'°';
 $('startVariation').textContent=(s.config.level===11?'Master start: distance +0–0.05 m / lateral ±0.05 m / heading ±2°. ':'Start variation ±0.5 m / ±0.3 m / ±5°. ')+'Infeasible starts are rejected; cable length is never increased.';$('nominal').textContent=s.nominal_distance.toFixed(2)+' m';$('startPose').textContent=s.start_pose.map(v=>v.toFixed(2)).join(' / ');$('cableLength').textContent=s.cable_length.toFixed(3)+' m';$('validation').textContent=s.error?'Build rejected; active scene retained':'Geometry / cable checked';$('obstacleCount').textContent='Active: '+s.obstacles.length;$('activeScene').textContent=activeLabel();$('layoutReadout').textContent=JSON.stringify(s.obstacles,null,2);
 const counts=shown.contacts.pair_episodes;const unsafe=counts.obstacle_robot+counts.robot_self+counts.unsafe_robot_environment;$('contactSummary').textContent=unsafe+' unsafe · '+counts.obstacle_cable+' cable/obstacle';
 const events=shown.events.slice().sort((a,b)=>a[0]-b[0]);$('eventLog').replaceChildren(...events.map(([t,text])=>{const row=document.createElement('div');const time=document.createElement('span');time.className='event-time';time.textContent=t.toFixed(3)+' s';row.append(time,document.createTextNode(text));return row;}));$('eventLog').scrollTop=$('eventLog').scrollHeight;
 $('attemptLabel').textContent=s.attempt;$('timeline').max=Math.max(0,(s.playback?.frames||s.frames)-1);if(s.playback)$('timeline').value=s.playback.index;else $('timeline').value=Math.max(0,s.frames-1);$('frameCount').textContent=(s.playback?.frames||s.frames)+' frames';$('timelineTime').textContent=shown.time.toFixed(2)+' s';$('playbackLabel').textContent=s.playback?'VISUAL PLAYBACK · NOT RE-SIMULATION':'LIVE SIMULATION · '+s.status.toUpperCase();
 const records=s.records||[];for(const id of ['attemptA','attemptB']){const old=$(id).value;const signature=records.map(r=>r.id+':'+r.frames).join();if($(id).dataset.signature!==signature){$(id).replaceChildren(...records.map((r,i)=>new Option(`${String.fromCharCode(65+i)} · ${r.id.replace('attempt_','')} · ${r.kind||'visual playback'}`,r.id)));$(id).dataset.signature=signature;$(id).value=records.some(r=>r.id===old)?old:(id==='attemptB'?s.attempt:records[0]?.id)||'';}}
 if(s.attempt!==lastAttempt){if(lastAttempt&&records.some(r=>r.id===lastAttempt))$('attemptA').value=lastAttempt;$('attemptB').value=s.playback?.id||s.attempt;lastAttempt=s.attempt;}
 $('start').disabled=s.status==='running'||s.status==='finished'||!!s.playback;$('pause').disabled=s.status!=='running';$('resume').disabled=s.status!=='paused'||!!s.playback;$('vehicle').disabled=s.config.level<9||s.status==='finished';const source=records.find(r=>r.id===playbackID());
 const checkpoints=source?.checkpoints||[];const selectedFrame=Number($('timeline').value);
 const checkpointSignature=source?.id+':'+checkpoints.map(c=>c.frame).join();
 if($('checkpointSelect').dataset.signature!==checkpointSignature){$('checkpointSelect').replaceChildren(new Option(checkpoints.length?'Select live checkpoint':'Visual playback only — no live checkpoint',''),...checkpoints.map(c=>new Option(c.time.toFixed(3)+' s · frame '+c.frame,c.frame)));$('checkpointSelect').dataset.signature=checkpointSignature;}
 const nearest=checkpoints.filter(c=>c.frame<=selectedFrame).at(-1);$('checkpointSelect').value=nearest?String(nearest.frame):'';
 $('restore').disabled=!nearest;$('restore').title=nearest?'Restore live state at '+nearest.time.toFixed(3)+' s':'Disk recordings are visual playback only';
 if(s.camera_error){$('viewPlaceholder').style.display='block';$('viewPlaceholder').textContent=s.camera_error;}if(s.failure)message(s.failure,true);editor();lockControls();
}
async function poll(){const epoch=uiEpoch;try{const response=await fetch('/api/state');if(!response.ok)throw new Error('Server unavailable');const s=await response.json();if(epoch===uiEpoch)paint(s);$('connection').textContent='Local simulator connected';$('led').className='connected';}catch(e){$('connection').textContent='Simulator disconnected';$('led').className='';}setTimeout(poll,250);}
async function image(){try{const response=await fetch('/api/frame');if(response.ok){const attempt=response.headers.get('X-Autocharge-Attempt'),camera=response.headers.get('X-Autocharge-Camera'),index=response.headers.get('X-Autocharge-Index');const expected=state?.playback?.id||state?.attempt;const expectedIndex=state?.playback?String(state.playback.index):'live';if(attempt!==expected||camera!==state?.camera||index!==expectedIndex){$('viewPlaceholder').style.display='block';$('viewPlaceholder').textContent='Updating viewport for the applied scene / selected frame…';$('viewport').style.opacity='.15';setTimeout(image,150);return;}const url=URL.createObjectURL(await response.blob());$('viewport').src=url;await $('viewport').decode();if(imageURL)URL.revokeObjectURL(imageURL);imageURL=url;lastImageAttempt=attempt;lastImageCamera=camera;lastImageIndex=index;$('viewport').style.opacity='1';$('viewPlaceholder').style.display='none';}}catch(e){}setTimeout(image,150);}
editor();poll();image();
