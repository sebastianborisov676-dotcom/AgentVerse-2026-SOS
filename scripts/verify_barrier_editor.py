#!/usr/bin/env python3
"""Real Chromium editor regression; start an isolated panel on port 8771 first."""
import json,sys
from pathlib import Path
sys.path.insert(0,'src');sys.path.insert(0,'scripts')
from verify_panel_browser import BrowserCheck
from autocharge.panel.placement import project
from playwright.sync_api import sync_playwright
out=Path('outputs/editor_barrier_fix');results=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1500,'height':1100},accept_downloads=True)
 c=BrowserCheck(page,'http://127.0.0.1:8771',out)
 page.on('response',lambda r: (out/'last_configure_trace.json').write_text(json.dumps({'request':r.request.post_data_json,'status':r.status,'response':r.json()},indent=2)) if r.url.endswith('/api/configure') else None)
 page.select_option('#level','6');page.fill('#seed','42');page.select_option('#mode','random');c.click('#apply','configure');page.select_option('#mode','hybrid');c.click('#apply','configure')
 page.select_option('#kind','barrier');page.locator('#angle').evaluate("e=>{e.value=-87;e.dispatchEvent(new Event('input'))}")
 page.click('#placeGround')
 view=c.state()['view_camera'];print('viewkeys',view.keys(),flush=True)
 uv=project(view, [[-1.46,3.6,0]])[0]
 a=page.evaluate('imageArea()');r=page.locator('#orbitArea').bounding_box()
 with page.expect_response(lambda x:x.url.endswith('/api/preview_placement')) as pending:
  page.mouse.click(r['x']+a['left']+uv[0]*a['width'],r['y']+a['top']+uv[1]*a['height'])
 preview=pending.value.json();results.append({'preview':preview});print('preview',preview['valid'],preview.get('reason'),flush=True)
 page.wait_for_function('!busy')
 before=len(c.state()['obstacles']);response=c.click('#applyPlacement','configure');assert len(response['obstacles'])==before+1
 barrier=[x for x in response['obstacles'] if x['kind']=='barrier'];assert len(barrier)==1;results.append({'accepted':barrier})
 with page.expect_download() as dl:c.click('#saveLayout','save_layout')
 path=out/'accepted_layout.json';dl.value.save_as(path)
 with page.expect_response(lambda x:x.url.endswith('/api/configure')) as pending:page.set_input_files('#loadLayout',str(path.resolve()))
 assert pending.value.status==200;page.wait_for_function('!busy');assert any(x['kind']=='barrier' for x in c.state()['obstacles'])
 page.screenshot(path=str(out/'accepted_barrier.png'))
 # Reproduce a protected-dock rejection using real Add controls; repeat cannot accumulate.
 page.select_option('#kind','barrier')
 for key,val in [('x',-0.3),('y',0.1)]:page.locator('#'+key).evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input"))}',val)
 response=c.click('#add','configure',400);n=page.evaluate('manual.length');response2=c.click('#add','configure',400);assert page.evaluate('manual.length')==n
 assert 'Obstacle' in page.locator('#placementStatus').inner_text();assert '⚠' in page.locator('#manualList').inner_text()
 results.append({'rejected':response,'repeat':response2,'draft_count':n,'inline':page.locator('#placementStatus').inner_text()})
 page.screenshot(path=str(out/'rejected_barrier.png'))
 # The rejected entry remains selectable and removable.
 page.select_option('#manualList',str(n-1));c.click('#remove','configure')
 assert page.evaluate('manual.length')==n-1
 results.append({'remove_rejected_entry':'passed'})
 (out/'browser_results.json').write_text(json.dumps(results,indent=2));browser.close()
