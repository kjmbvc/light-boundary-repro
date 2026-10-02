"""Direct illuminance from a finite one-sided Lambertian rectangular source."""
from __future__ import annotations
import numpy as np

EPS = 1e-12

def midpoint_axis(low, high, count):
    if not np.isfinite([low, high]).all() or high <= low or type(count) is not int or count < 1:
        raise ValueError('finite increasing limits and a positive integer count required')
    return low + (np.arange(count, dtype=float) + 0.5) * (high - low) / count

def source_patches(source, grid):
    center = np.asarray(source['center'], dtype=float)
    width, depth, flux = source['width'], source['depth'], source['flux_lm']
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite([width,depth,flux]).all():
        raise ValueError('finite source values required')
    if width <= 0 or depth <= 0 or flux <= 0:
        raise ValueError('positive source dimensions and flux required')
    normal = np.asarray(source['normal'], dtype=float)
    if normal.shape != (3,) or not np.allclose(normal, [0,0,-1], atol=0, rtol=0):
        raise ValueError('this model requires a horizontal downward source')
    x = midpoint_axis(center[0]-width/2, center[0]+width/2, grid)
    y = midpoint_axis(center[1]-depth/2, center[1]+depth/2, grid)
    xx, yy = np.meshgrid(x,y,indexing='xy')
    patches = np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,center[2])))
    return patches, flux/patches.shape[0]

def validate_shield(shield, source=None):
    if shield is None:
        return
    vals = [shield[k] for k in ('y_plane','x_min','x_max','z_min','z_max')]
    if not np.isfinite(vals).all() or shield['x_max'] <= shield['x_min'] or shield['z_max'] <= shield['z_min']:
        raise ValueError('finite nondegenerate rectangular shield required')
    if shield.get('thickness_m',0) != 0 or shield.get('reflectance',0) != 0 or not shield.get('opaque',True):
        raise ValueError('only zero-thickness opaque absorbing shields are supported')
    if source:
        cx,cy,cz=source['center']; w=source['width']; d=source['depth']
        overlap=(cy-d/2-EPS <= shield['y_plane'] <= cy+d/2+EPS
                 and shield['z_min']-EPS <= cz <= shield['z_max']+EPS
                 and max(cx-w/2,shield['x_min']) <= min(cx+w/2,shield['x_max'])+EPS)
        if overlap:
            raise ValueError('shield intersects the emitter surface')

def segment_occluded(origins, targets, shield):
    """M target rows x N source columns; intersection must be inside the segment.

    Rectangle edges are conservatively opaque. Endpoints are excluded. Parallel
    (including coplanar) segments have no isolated plane crossing in this model.
    """
    validate_shield(shield)
    o=np.asarray(origins,dtype=float); p=np.asarray(targets,dtype=float)
    if o.ndim!=2 or p.ndim!=2 or o.shape[1]!=3 or p.shape[1]!=3 or not np.isfinite(o).all() or not np.isfinite(p).all():
        raise ValueError('finite Nx3 source and Mx3 target arrays required')
    delta=p[:,None,:]-o[None,:,:]
    denominator=delta[:,:,1]
    t=np.divide(shield['y_plane']-o[None,:,1],denominator,
                out=np.full(denominator.shape,np.inf),where=np.abs(denominator)>EPS)
    valid=(t>EPS)&(t<1-EPS)&np.isfinite(t)
    safe_t=np.where(valid,t,0.0)
    x=o[None,:,0]+safe_t*delta[:,:,0]
    z=o[None,:,2]+safe_t*delta[:,:,2]
    return (valid & (x>=shield['x_min']-EPS)&(x<=shield['x_max']+EPS)
            &(z>=shield['z_min']-EPS)&(z<=shield['z_max']+EPS))

def illuminance(source, points, normals, source_grid=31, shield=None, chunk_size=192):
    """Return sampled direct illuminance in lux, without reflected light.

    Equal patch flux dPhi = Phi/N; dE = (dPhi/pi)*cos_s*cos_r/r^2.
    Patch area is already represented by dPhi and is not multiplied again.
    """
    patches,patch_flux=source_patches(source,source_grid)
    validate_shield(shield,source)
    pts=np.asarray(points,dtype=float); norms=np.asarray(normals,dtype=float)
    if pts.ndim!=2 or pts.shape[1]!=3 or not np.isfinite(pts).all():
        raise ValueError('finite Mx3 receiver points required')
    if norms.shape==(3,): norms=np.broadcast_to(norms,pts.shape)
    if norms.shape!=pts.shape or not np.isfinite(norms).all() or not np.allclose(np.linalg.norm(norms,axis=1),1,atol=1e-12):
        raise ValueError('unit receiver normals required')
    if type(chunk_size) is not int or chunk_size<1: raise ValueError('positive chunk size required')
    result=np.empty(len(pts),dtype=float)
    sn=np.asarray(source['normal'],dtype=float)
    for start in range(0,len(pts),chunk_size):
        stop=min(start+chunk_size,len(pts)); block=pts[start:stop]
        delta=block[:,None,:]-patches[None,:,:]
        r2=np.sum(delta*delta,axis=2)
        if np.any(r2<=EPS): raise ValueError('receiver coincides with a source patch')
        unit=delta/np.sqrt(r2[:,:,None])
        cos_s=np.maximum(0,np.einsum('mni,i->mn',unit,sn))
        cos_r=np.maximum(0,-np.einsum('mni,mi->mn',unit,norms[start:stop]))
        values=(patch_flux/np.pi)*cos_s*cos_r/r2
        if shield is not None: values=np.where(segment_occluded(patches,block,shield),0.0,values)
        result[start:stop]=np.sum(values,axis=1)
    return result

def surface_grid(surface, counts=None):
    nu,nv=counts or surface['grid']
    u=midpoint_axis(*surface['u_range'],nu); v=midpoint_axis(*surface['v_range'],nv)
    uu,vv=np.meshgrid(u,v,indexing='xy'); fixed=np.full(uu.size,surface['fixed'])
    if surface['plane']=='y': points=np.column_stack((uu.ravel(),fixed,vv.ravel()))
    elif surface['plane']=='z': points=np.column_stack((uu.ravel(),vv.ravel(),fixed))
    else: raise ValueError('supported receiver planes are y and z')
    return points,np.asarray(surface['normal'],dtype=float),u,v

def metrics(before, after):
    before=np.asarray(before,dtype=float); after=np.asarray(after,dtype=float)
    if before.shape!=after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('matching finite arrays required')
    positive=before>EPS
    retention=after[positive]/before[positive] if positive.any() else np.array([])
    return {'sample_count':int(before.size),'before_min_lx':float(before.min()),'before_mean_lx':float(before.mean()),
            'before_max_lx':float(before.max()),'after_min_lx':float(after.min()),'after_mean_lx':float(after.mean()),
            'after_max_lx':float(after.max()),'mean_reduction_pct':float((1-after.mean()/before.mean())*100) if before.mean()>EPS else None,
            'minimum_pointwise_retention_pct':float(retention.min()*100) if retention.size else None,
            'maximum_increase_lx':float(np.max(after-before)),
            'mean_method':'arithmetic mean at uniform receiver-cell centers; midpoint estimate of area mean'}
