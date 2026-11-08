#!/usr/bin/env python3
"""Browser regression for the verified Level 11 demo on the isolated port 8772 panel."""
import sys,json
from pathlib import Path
import numpy as np
sys.path[:0]=['scripts']
from verify_panel_browser import BrowserCheck
from playwright.sync_api import sync_playwright
out=Path('outputs/master_demo'); evidence=out/'browser';evidence.mkdir(exist_ok=True)
selected=json.loads((out/'selected_demo.json').read_text());record=Path(selected['recording']);demo=record.stem
archive=np.load(record);metadata=json.loads(str(archive['metadata']));times=np.array([t['time'] for t in metadata['telemetry']]);checks=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1500,'height':1100})
 c=BrowserCheck(page,'http://127.0.0.1:8772',evidence)
 if c.state().get('playback'):
  assert page.locator('#attemptB').input_value()==c.state()['playback']['id']
 if c.state()['camera']!='external':
  with page.expect_response(lambda r:r.url.endswith('/api/camera')):page.select_option('#camera','external')
  page.wait_for_function('!busy')
 assert 'Master Challenge' in page.locator('#level').inner_text()
 page.select_option('#level','11');page.fill('#seed','42');page.select_option('#mode','random');page.check('#movingCar');page.check('#randomStart')
 result=c.click('#apply','configure');assert result['obstacles']==selected['obstacles'];assert list(result['start_pose'])==selected['start_pose']
 saved=c.click('#saveRun','save_run');initial=np.load(saved['path']);assert np.array_equal(initial['states'][0],archive['states'][0]);checks.append({'panel_vs_batch_initial_integration_state_equal':True})
 with page.expect_response(lambda r:r.url.endswith('/api/playback')) as pending:page.select_option('#attemptB',demo)
 assert pending.value.status==200;page.wait_for_function('!busy');assert page.locator('#restore').is_disabled()
 for name,time in [('avoidance',.6),('braking',33.3),('updated_target',37.8),('seated',1000)]:
  index=int(np.searchsorted(times,time));index=min(index,len(times)-1)
  with page.expect_response(lambda r:r.url.endswith('/api/playback')) as pending:
   page.locator('#timeline').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"))}',index)
  assert pending.value.status==200;page.wait_for_function('!busy')
  page.wait_for_function('(v)=>lastImageAttempt===v.id && lastImageIndex===String(v.index)',arg={'id':demo,'index':index},timeout=60000)
  page.screenshot(path=str(evidence/(name+'.png')),full_page=True);page.locator('#orbitArea').screenshot(path=str(evidence/(name+'_view.png')));s=c.state();checks.append({'name':name,'index':index,'time':s['playback']['telemetry']['time'],'phase':s['playback']['telemetry']['phase'],'events':s['playback']['telemetry']['events']})
 with page.expect_response(lambda r:r.url.endswith('/api/camera')):page.select_option('#camera','connector_side')
 page.wait_for_function('!busy');page.wait_for_function('lastImageCamera==="connector_side"',timeout=60000);page.screenshot(path=str(evidence/'connector_seated.png'))
 assert not c.errors,c.errors
 checks.append({'browser_errors':c.errors,'disk_restore_disabled':True});(evidence/'verification.json').write_text(json.dumps(checks,indent=2));browser.close()
print('Panel build, initial-state equality, recording playback, timeline, camera and restore guard passed')
