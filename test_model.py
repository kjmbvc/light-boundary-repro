import copy,json,unittest
from pathlib import Path
import numpy as np
from model import illuminance,segment_occluded,source_patches,validate_shield

BASE=Path(__file__).resolve().parent
CONFIG=json.loads((BASE/'inputs.json').read_text(encoding='utf-8'))

class PhysicsTests(unittest.TestCase):
    def point(self,height=2,flux=100*np.pi):
        return {'center':[0,0,height],'width':1e-5,'depth':1e-5,'flux_lm':flux,'normal':[0,0,-1]}
    def test_analytic_point_limit(self):
        actual=illuminance(self.point(),[[0,0,0]],[0,0,1],21)[0]
        self.assertAlmostEqual(actual,25.0,places=8)
    def test_inverse_square_point_limit(self):
        e2=illuminance(self.point(2),[[0,0,0]],[0,0,1],21)[0]
        e4=illuminance(self.point(4),[[0,0,0]],[0,0,1],21)[0]
        self.assertAlmostEqual(e2/e4,4.0,places=9)
    def test_source_and_receiver_cosines(self):
        # 3-4-5 triangle: source cos=4/5, horizontal receiver cos=4/5.
        s=self.point(4)
        e=illuminance(s,[[3,0,0]],[0,0,1],21)[0]
        expected=100*(4/5)*(4/5)/25
        self.assertAlmostEqual(e,expected,places=8)
    def test_linear_flux_scaling(self):
        s=copy.deepcopy(CONFIG['source']); p=[[1,0,0],[-1,4,2]]; n=[[0,0,1],[0,-1,0]]
        first=illuminance(s,p,n,13,CONFIG['shield']); s['flux_lm']*=2.7
        second=illuminance(s,p,n,13,CONFIG['shield'])
        np.testing.assert_allclose(second,2.7*first,rtol=1e-12,atol=1e-12)
    def test_one_sided_source_and_receiver(self):
        s=self.point()
        self.assertEqual(illuminance(s,[[0,0,3]],[0,0,-1],13)[0],0)
        self.assertEqual(illuminance(s,[[0,0,0]],[0,0,-1],13)[0],0)
    def test_patch_flux_conservation(self):
        for n in [13,21,31]:
            p,f=source_patches(CONFIG['source'],n)
            self.assertAlmostEqual(len(p)*f,CONFIG['source']['flux_lm'],places=9)
    def test_segment_intersection_and_rectangle_bounds(self):
        sh={'y_plane':1,'x_min':-1,'x_max':1,'z_min':-1,'z_max':1}
        source=np.array([[0,0,0]])
        targets=np.array([[0,2,0],[2,2,0],[2.01,2,0],[0,.5,0],[0,-2,0],[2,0,0]])
        np.testing.assert_array_equal(segment_occluded(source,targets,sh).ravel(),[True,True,False,False,False,False])
    def test_intersection_beyond_target_is_not_occlusion(self):
        sh={'y_plane':3,'x_min':-5,'x_max':5,'z_min':-5,'z_max':5}
        self.assertFalse(segment_occluded([[0,0,0]],[[0,2,0]],sh)[0,0])
    def test_source_shield_overlap_rejected(self):
        sh=copy.deepcopy(CONFIG['shield']); sh['y_plane']=0
        with self.assertRaises(ValueError): validate_shield(sh,CONFIG['source'])
    def test_nominal_geometry_does_not_overlap_source(self):
        validate_shield(CONFIG['shield'],CONFIG['source'])
    def test_full_and_no_occlusion(self):
        s=CONFIG['source']; sh=CONFIG['shield']
        point=[[0,4,2]]; norm=[0,-1,0]
        self.assertGreater(illuminance(s,point,norm,21)[0],0)
        self.assertEqual(illuminance(s,point,norm,21,sh)[0],0)
        before=illuminance(s,[[0,0,0]],[0,0,1],21)[0]
        after=illuminance(s,[[0,0,0]],[0,0,1],21,sh)[0]
        self.assertEqual(before,after)
    def test_partial_occlusion_approaches_half(self):
        # At x=0, source contributions are symmetric in x. Half-source limit=0.5.
        sh=copy.deepcopy(CONFIG['shield']); sh['x_min']=0
        errors=[]
        for n in [13,21,31]:
            before=illuminance(CONFIG['source'],[[0,4,2]],[0,-1,0],n)[0]
            after=illuminance(CONFIG['source'],[[0,4,2]],[0,-1,0],n,sh)[0]
            fraction=after/before
            self.assertGreater(fraction,.45); self.assertLess(fraction,.5)
            errors.append(abs(.5-fraction))
        self.assertTrue(errors[2]<errors[1]<errors[0])
        for n in [32,64]:
            before=illuminance(CONFIG['source'],[[0,4,2]],[0,-1,0],n)[0]
            after=illuminance(CONFIG['source'],[[0,4,2]],[0,-1,0],n,sh)[0]
            self.assertAlmostEqual(after/before,.5,places=12)
    def test_absorbing_shield_never_increases_direct_light(self):
        rng=np.random.default_rng(7321)
        points=np.column_stack((rng.uniform(-5,5,80),rng.uniform(-4,4,80),np.zeros(80)))
        before=illuminance(CONFIG['source'],points,[0,0,1],13)
        after=illuminance(CONFIG['source'],points,[0,0,1],13,CONFIG['shield'])
        self.assertTrue(np.all(after<=before+1e-12)); self.assertTrue(np.all(after>=0))
    def test_invalid_inputs(self):
        s=copy.deepcopy(CONFIG['source']); s['flux_lm']=-1
        with self.assertRaises(ValueError): illuminance(s,[[0,0,0]],[0,0,1])
        with self.assertRaises(ValueError): illuminance(CONFIG['source'],[[0,0,0]],[0,0,2])
        with self.assertRaises(ValueError): illuminance(CONFIG['source'],[[np.nan,0,0]],[0,0,1])

if __name__=='__main__': unittest.main(verbosity=2)
