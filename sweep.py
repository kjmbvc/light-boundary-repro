"""One-variable design exploration, not an optimization or field validation."""
import copy,csv,json
from pathlib import Path
import numpy as np
from model import illuminance,surface_grid,metrics,validate_shield

BASE=Path(__file__).resolve().parent
config=json.loads((BASE/'inputs.json').read_text(encoding='utf-8'))
cache={}
for surface in config['surfaces']:
    p,n,_,_=surface_grid(surface)
    cache[surface['id']]=(p,n,illuminance(config['source'],p,n,config['source_grid']))
rows=[]
for bottom in [5.0,5.45,5.65,5.8,5.9]:
    shield=copy.deepcopy(config['shield']);shield['z_min']=bottom
    validate_shield(shield,config['source'])
    for surface in config['surfaces']:
        p,n,before=cache[surface['id']]
        after=illuminance(config['source'],p,n,config['source_grid'],shield)
        assert np.all(after<=before+1e-12)
        m=metrics(before,after)
        rows.append({'shield_bottom_z_m':bottom,'surface':surface['id'],
                     'before_mean_lx':m['before_mean_lx'],'after_mean_lx':m['after_mean_lx'],
                     'mean_reduction_pct':m['mean_reduction_pct'],
                     'mean_retention_pct':100-m['mean_reduction_pct'],
                     'minimum_pointwise_retention_pct':m['minimum_pointwise_retention_pct']})
with (BASE/'parameter_sweep.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(BASE/'results/parameter_sweep.json').write_text(json.dumps({'varied_parameter':'shield.z_min',
  'source_grid':config['source_grid'],'rows':rows,
  'interpretation':'Five assumed designs; no optimal design or legal threshold is claimed.'},indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows))
