"""Conservative angle-interval screen, not a photometric simulation."""
import json, math
from pathlib import Path

def screen(window, walkway, margin=2):
    if not isinstance(window,(list,tuple)) or not isinstance(walkway,(list,tuple)) or len(window)!=2 or len(walkway)!=2:
        raise ValueError('each angle interval must contain exactly two endpoints')
    all_values=[*window,*walkway,margin]
    if not all(type(v) in (int,float) and math.isfinite(v) for v in all_values):
        raise ValueError('finite numeric angles required')
    if not 0<=window[0]<window[1]<=90 or not 0<=walkway[0]<walkway[1]<=90 or not 0<=margin<=15:
        raise ValueError('ordered downward angles 0..90 and margin 0..15 required')
    shield=(max(0,window[0]-margin),min(90,window[1]+margin))
    overlap=max(shield[0],walkway[0])<=min(shield[1],walkway[1])+1e-9
    return {'window_deg':window,'walkway_deg':walkway,'margin_deg':margin,
            'shield_interval_deg':shield,'angular_conflict':overlap,
            'result':'stop_and_redesign' if overlap else 'candidate_for_3d_and_field_validation',
            'installation_approved':False,'illuminance_simulated':False}

if __name__=='__main__':
    cases=[('separated',[10,28],[36,72],2,False),
           ('overlap',[25,48],[36,72],2,True),
           ('touching',[10,34],[36,72],2,True),
           ('margin_creates_conflict',[10,32],[36,72],5,True),
           ('shield_contains_walkway',[10,70],[36,60],0,True)]
    results=[]
    for name,w,p,m,expected in cases:
        result=screen(w,p,m)
        assert result['angular_conflict']==expected
        results.append({'case':name,**result,'passed':True})
    for name,w,p,m in [('reversed',[28,10],[36,72],2),('nan',[10,float('nan')],[36,72],2),('outside',[10,92],[36,72],2)]:
        try: screen(w,p,m)
        except ValueError: results.append({'case':name,'invalid_input_rejected':True,'passed':True})
        else: raise AssertionError(name)
    report={'cases':results,'passed':len(results),'all_inputs':'synthetic assumed angles; no real location or measurement',
            'claim_limit':'A non-overlapping interval is only a candidate. A 2D point-light model cannot approve field installation or establish lux/glare/energy savings.'}
    Path(__file__).with_name('geometry-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':len(results),'measured_field_data':False}))
