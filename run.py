from pathlib import Path
import csv,copy,hashlib,json,platform,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from model import illuminance,metrics,surface_grid,validate_shield

BASE=Path(__file__).resolve().parent
OUT=BASE/'results'; FIG=BASE/'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
config=json.loads((BASE/'inputs.json').read_text(encoding='utf-8'))

def evaluate(surface,source_grid,counts=None):
    points,normals,u,v=surface_grid(surface,counts)
    before=illuminance(config['source'],points,normals,source_grid)
    after=illuminance(config['source'],points,normals,source_grid,config['shield'])
    assert np.all(after<=before+1e-12) and np.all(after>=0)
    return {'u':u,'v':v,'before':before.reshape(len(v),len(u)),
            'after':after.reshape(len(v),len(u)),'metrics':metrics(before,after)}

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def make_heatmaps(fields):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,'axes.labelsize':10,
                         'xtick.labelsize':10,'ytick.labelsize':10})
    fig,axs=plt.subplots(2,2,figsize=(7,3.6),layout='constrained')
    fig.suptitle('Assumed geometry | Direct light only | Not measured',fontsize=10)
    wl=fields['window_left']; wr=fields['window_right']; ground=fields['walkway']
    surfaces={s['id']:s for s in config['surfaces']}
    left,right,walk=[surfaces[k] for k in ['window_left','window_right','walkway']]
    extent=lambda s:[*s['u_range'],*s['v_range']]
    window_x=(min(left['u_range'][0],right['u_range'][0]),max(left['u_range'][1],right['u_range'][1]))
    window_z=(min(left['v_range'][0],right['v_range'][0]),max(left['v_range'][1],right['v_range'][1]))
    window_max=max(wl['before'].max(),wr['before'].max()); ground_max=ground['before'].max()
    for j,key in enumerate(['before','after']):
        ax=axs[0,j]; ax.set_facecolor('#e4e7ea')
        m=ax.imshow(wl[key],origin='lower',extent=extent(left),vmin=0,vmax=window_max,cmap='inferno',aspect='auto')
        ax.imshow(wr[key],origin='lower',extent=extent(right),vmin=0,vmax=window_max,cmap='inferno',aspect='auto')
        ax.set_xlim(*window_x); ax.set_ylim(*window_z)
        ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_title('Windows: '+('before' if j==0 else 'after')); ax.set_xlabel('x (m)'); ax.set_ylabel('z (m)')
        if left['u_range'][1]<right['u_range'][0]:
            ax.text((left['u_range'][1]+right['u_range'][0])/2,sum(window_z)/2,'wall',ha='center',va='center',rotation=90,color='#47505b',fontsize=10)
        bx=axs[1,j]
        n=bx.imshow(ground[key],origin='lower',extent=extent(walk),vmin=0,vmax=ground_max,cmap='viridis',aspect='auto')
        bx.xaxis.set_major_locator(MaxNLocator(3));bx.yaxis.set_major_locator(MaxNLocator(3))
        bx.set_xlabel('x (m)'); bx.set_ylabel('y (m)')
        bx.set_title('Walkway: '+('before' if j==0 else 'after'))
    fig.colorbar(m,ax=list(axs[0,:]),label='lux',fraction=.03,pad=.02)
    fig.colorbar(n,ax=list(axs[1,:]),label='lux',fraction=.03,pad=.02)
    fig.get_layout_engine().set(h_pad=.025,w_pad=.04,hspace=.02,wspace=.02)
    fig.savefig(FIG/'illuminance_before_after.png',dpi=240)
    fig.savefig(FIG/'illuminance_before_after.svg')
    plt.close(fig)

def make_scene():
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7,2.35),layout='constrained')
    fig.suptitle('Assumed geometry | Zero-thickness opaque shield',fontsize=10)
    s=config['source'];sh=config['shield'];sx,sy,sz=s['center']
    windows=[v for v in config['surfaces'] if v['plane']=='y'];walk=next(v for v in config['surfaces'] if v['id']=='walkway')
    wy=walk['v_range'];wx=walk['u_range'];wz=walk['fixed']
    ax.plot(wy,[wz,wz],color='#347e51',lw=4)
    for window in windows:
        ax.plot([window['fixed']]*2,window['v_range'],color='#985bc4',lw=4)
    ax.plot([sy-s['depth']/2,sy+s['depth']/2],[sz,sz],color='#d59a00',lw=5)
    ax.plot([sh['y_plane']]*2,[sh['z_min'],sh['z_max']],color='#1265ad',lw=3)
    first=windows[0]
    for pz in first['v_range']:ax.plot([sy,first['fixed']],[sz,pz],'--',lw=.9,color='#ac9c79')
    ax.plot([sy,wy[1]],[sz,wz],'--',lw=.9,color='#ac9c79')
    ymin=min(wy[0],sy,*[w['fixed'] for w in windows]);ymax=max(wy[1],sy,*[w['fixed'] for w in windows]);span=ymax-ymin
    zmax=max(sz,sh['z_max'],*[w['v_range'][1] for w in windows]);zmin=min(wz,*[w['v_range'][0] for w in windows]);zspan=zmax-zmin
    ax.annotate('source',xy=(sy,sz),xytext=(ymin+.03*span,sz-.05*zspan),arrowprops={'arrowstyle':'-','lw':.8},fontsize=10)
    ax.annotate('shield',xy=(sh['y_plane'],(sh['z_min']+sh['z_max'])/2),xytext=(sy+.18*span,zmax+.01*zspan),arrowprops={'arrowstyle':'-','lw':.8},fontsize=10)
    ax.text(first['fixed']-.03*span,first['v_range'][1]+.06*zspan,'windows',ha='right',fontsize=10,color='#623980')
    ax.set(xlim=(ymin-.05*span,ymax+.09*span),ylim=(zmin-.06*zspan,zmax+.09*zspan),xlabel='y (m)',ylabel='z (m)',title='Side projection')
    ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3));ax.grid(alpha=.15)
    bx.add_patch(Rectangle((wx[0],wy[0]),wx[1]-wx[0],wy[1]-wy[0],color='#347e51',alpha=.2))
    for i,w in enumerate(windows):
        bx.plot(w['u_range'],[w['fixed']]*2,color='#985bc4',lw=5)
        bx.text(sum(w['u_range'])/2,w['fixed']+.06*span,'L' if i==0 else 'R',ha='center',fontsize=10)
    bx.add_patch(Rectangle((sx-s['width']/2,sy-s['depth']/2),s['width'],s['depth'],color='#d59a00'))
    bx.plot([sh['x_min'],sh['x_max']],[sh['y_plane']]*2,color='#1265ad',lw=3)
    xspan=wx[1]-wx[0]
    bx.annotate('shield',xy=(sh['x_max'],sh['y_plane']),xytext=(sh['x_max']+.12*xspan,sh['y_plane']+.05*span),arrowprops={'arrowstyle':'-','lw':.8},fontsize=10)
    bx.text(sum(wx)/2,sum(wy)/2,'walkway',ha='center',color='#25603e',fontsize=10)
    bx.set(xlim=(wx[0]-.05*xspan,wx[1]+.05*xspan),ylim=(ymin-.06*span,ymax+.15*span),xlabel='x (m)',ylabel='y (m)',title='Plan view')
    bx.xaxis.set_major_locator(MaxNLocator(3));bx.yaxis.set_major_locator(MaxNLocator(3));bx.grid(alpha=.15)
    fig.savefig(FIG/'scene_layout.png',dpi=240); fig.savefig(FIG/'scene_layout.svg');plt.close(fig)

def main():
    start=time.perf_counter();validate_shield(config['shield'],config['source'])
    fields={}; metrics_list=[]
    for surface in config['surfaces']:
        data=evaluate(surface,config['source_grid']);fields[surface['id']]=data
        metrics_list.append({'surface':surface['id'],**data['metrics']})
    source_convergence=[]
    for n in config['source_convergence_grids']:
        for surface in config['surfaces']:
            data=evaluate(surface,n);ref=fields[surface['id']]
            source_convergence.append({'source_grid':n,'patches':n*n,'surface':surface['id'],
             'receiver_grid':surface['grid'],'before_mean_lx':float(data['before'].mean()),
             'after_mean_lx':float(data['after'].mean()),
             'reference_source_grid':config['source_grid'],
             'max_abs_before_difference_to_reference_lx':float(np.abs(data['before']-ref['before']).max()),
             'max_abs_after_difference_to_reference_lx':float(np.abs(data['after']-ref['after']).max())})
    receiver_convergence=[]
    for n in config['receiver_convergence_grids']:
        for surface in config['surfaces']:
            data=evaluate(surface,config['source_grid'],[n,n])
            receiver_convergence.append({'surface':surface['id'],'receiver_grid':[n,n],
             'source_grid':config['source_grid'],**data['metrics']})
    for surface in config['surfaces']:
        rows=[r for r in receiver_convergence if r['surface']==surface['id']]
        ref=rows[-1]
        for row in rows:
            row['reference_receiver_grid']=ref['receiver_grid']
            row['abs_mean_before_difference_to_reference_lx']=abs(row['before_mean_lx']-ref['before_mean_lx'])
            row['abs_mean_after_difference_to_reference_lx']=abs(row['after_mean_lx']-ref['after_mean_lx'])
    # This fixed symmetric control is independent of the editable design inputs.
    control_source={'center':[0,0,6],'width':.24,'depth':.16,'flux_lm':3000,'normal':[0,0,-1]}
    control_shield={'y_plane':.25,'x_min':0,'x_max':.4,'z_min':5.45,'z_max':6.1}
    control_point=[0,4,2]
    partial=[]
    for n in [13,21,31,32,64]:
        before=illuminance(control_source,[control_point],[0,-1,0],n)[0]
        after=illuminance(control_source,[control_point],[0,-1,0],n,control_shield)[0]
        partial.append({'source_grid':n,'before_lx':float(before),'after_lx':float(after),'retained_fraction':float(after/before),
                        'absolute_error_from_symmetric_half_fraction':float(abs(after/before-.5))})
    make_heatmaps(fields);make_scene()
    result={'model':'finite one-sided Lambertian area source; direct illuminance only',
      'scenario_status':'assumed geometry, not measured', 'source_grid':config['source_grid'],
      'inputs_sha256':hashlib.sha256((BASE/'inputs.json').read_bytes()).hexdigest(),
      'source_shield_overlap':False,'all_receivers_after_le_before':True,
      'metrics':metrics_list,'source_grid_convergence':source_convergence,
      'receiver_grid_convergence':receiver_convergence,'partial_occlusion_control':partial,
      'partial_control_inputs':{'source':control_source,'shield':control_shield,'receiver_point':control_point,'receiver_normal':[0,-1,0]},
      'partial_control_note':'Fixed symmetric control independent of inputs.json. Half source covered at x>=0. A center column lies exactly on the absorbing edge for odd grids; convergence is toward the symmetric half-flux limit.',
      'window_quantity':'incident vertical illuminance on the external window plane; not indoor illuminance',
      'retention_note':'minimum of after/before at points whose before value exceeds 1e-12 lux; not ratio of minima',
      'versions':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
      'elapsed_seconds':round(time.perf_counter()-start,3)}
    write_json(BASE/'results.json',result)
    write_json(OUT/'convergence.json',{'source':source_convergence,'receiver':receiver_convergence,'partial_occlusion':partial})
    with (BASE/'comparison.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics_list[0]));writer.writeheader();writer.writerows(metrics_list)
    arrays={}
    for ident,data in fields.items():
        for key in ['u','v','before','after']:arrays[ident+'_'+key]=data[key]
    np.savez_compressed(OUT/'illuminance_fields.npz',**arrays)
    print(json.dumps({'metrics':metrics_list,'seconds':result['elapsed_seconds']},ensure_ascii=False))

if __name__=='__main__': main()
