import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,TARGETS,RAW,load_data,partition,group_folds,pad_curves,physical_metrics,anchors_from_curve
from joint_math import consistent_anchors,reconstruct,pwl_metrics,AreaPower
from joint_model import Representation,ResidualNetwork,tensor_pack,specifications as joint_specs
from classical import specifications as classic_specs,ClassicalModel
from metrics import safe_r2,choose
from features import feature_matrix,preprocessing

CONFIG=yaml.safe_load((ROOT/'config/split811_20261102.yml').read_text())


class CoreTests(unittest.TestCase):
    def setUp(self):self.frame,self.curves=partition(20261102)
    def train(self,n=28):return self.frame[self.frame.fit_eligible].iloc[:n].copy()
    def test_population(self):
        self.assertEqual(len(self.frame),199);self.assertEqual(self.frame.fit_eligible.sum(),158)
        self.assertEqual(self.frame.role.eq('quarantine').sum(),1)
        self.assertEqual(self.frame.curve_id.str.startswith('PRESENT_').sum(),10)
    def test_all_partitions(self):
        for seed in CONFIG['split_seeds']:
            f,_=partition(seed)
            self.assertEqual(f.split.value_counts().to_dict(),dict(train=158,val=20,test=20,quarantine=1))
            self.assertEqual(f.groupby('specimen_group').split.nunique().max(),1)
            self.assertEqual(f.nested_A.sum(),131)
            self.assertTrue(f.loc[f.fit_eligible & f.curve_id.str.startswith('PRESENT_'),'nested_A'].all())
            for role,quotas in CONFIG['role_quotas'].items():
                self.assertEqual([int(((f.role==role)&(f.split==s)).sum()) for s in ('train','val','test')],quotas)
    def test_group_folds(self):
        f=self.train(100)
        for a,b in group_folds(f,CONFIG['seeds'][0]):self.assertFalse(set(f.iloc[a].specimen_group)&set(f.iloc[b].specimen_group))
    def test_targets(self):
        for r in self.frame.itertuples():
            np.testing.assert_allclose(anchors_from_curve(self.curves[r.curve_id]),[getattr(r,t) for t in TARGETS],rtol=2e-11)
    def test_units_present(self):
        own=self.frame[self.frame.curve_id.str.startswith('PRESENT_')]
        self.assertEqual(sorted(own.m.unique()),[1200,1300])
        self.assertEqual(own.iloc[1].v,4.21);self.assertTrue(own.d.isna().all());self.assertTrue(own.fyt.isna().all())
    def test_candidates(self):
        self.assertEqual(len(classic_specs(CONFIG)),56);self.assertEqual(len(joint_specs(CONFIG)),96)
    def test_no_response_features(self):
        f=self.train();before=feature_matrix(f,'physics');f[TARGETS]=999999
        f['source_group_id']='changed';np.testing.assert_equal(before,feature_matrix(f,'physics'))
        self.assertEqual(len(RAW),17)
    def test_identity(self):
        for structure in ('impulse','fullness'):
            base=torch.tensor([[100.,20.,1500.,.7,.2],[10.,2.,100.,.1,.9]],dtype=torch.float64)
            a,d=consistent_anchors(base,structure)
            np.testing.assert_allclose(a[:,2],a[:,0]*a[:,1]*a[:,3],rtol=2e-14)
            self.assertTrue(bool(((a[:,3]>0)&(a[:,3]<1)).all()))
    def test_calibration_bounds(self):
        b=torch.tensor([[100.,20.,1000.,.5,.3]],dtype=torch.float64)
        a,_=consistent_anchors(b,'fullness',torch.ones((1,4),dtype=torch.float64)*100)
        np.testing.assert_allclose(a[0,:2]/b[0,:2],1.1,rtol=1e-12)
        self.assertAlmostEqual(float(torch.logit(a[0,3])-torch.logit(b[0,3])),.2)
    def test_reconstruction(self):
        raw=torch.tensor(np.tile(np.sin(np.linspace(0,np.pi,201)),(2,1)),dtype=torch.float64)
        a=torch.tensor([[100.,20.,800.,.4,.173],[25.,80.,500.,.25,.777]],dtype=torch.float64)
        t,f,info=reconstruct(raw,a)
        self.assertTrue(bool((f>=0).all()));np.testing.assert_allclose(info['area_error'],0,atol=2e-12)
        for i in range(2):
            ti,ix=np.unique(t[i].numpy(),return_index=True)
            actual=anchors_from_curve((ti,f[i].numpy()[ix]));np.testing.assert_allclose(actual,a[i],rtol=1e-10)
    def test_pwl_nonzero_ends(self):
        tx=np.array([0.,1.,2.]);fy=np.array([.05,1.,.05]);px=np.array([0.,1.,3.]);py=np.array([0.,.8,0.])
        expected=physical_metrics(tx,fy,px,py)
        tensors=[torch.tensor(v[None,:],dtype=torch.float64) for v in (tx,fy,px,py)]
        ratio,nrmse,_=pwl_metrics(*tensors)
        self.assertAlmostEqual(float(1-ratio[0]),expected['curve_R2'],places=12)
        self.assertAlmostEqual(float(nrmse[0]),expected['NRMSE'],places=12)
    def test_padded_pwl(self):
        ids=self.frame.curve_id.iloc[[0,189]].tolist();t,f=pad_curves(ids,self.curves)
        ratio,_,_=pwl_metrics(torch.tensor(t),torch.tensor(f),torch.tensor(t),torch.tensor(f))
        np.testing.assert_allclose(ratio,0,atol=1e-12)
    def test_area_gradient(self):
        base=torch.tensor([[.2,.5,.9]],dtype=torch.float64,requires_grad=True)
        w=torch.tensor([[.3,.4,.3]],dtype=torch.float64,requires_grad=True)
        area=torch.tensor([.4],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(AreaPower.apply,(base,w,area),eps=1e-6,atol=2e-5))
    def test_physical_loss_gradient(self):
        raw=torch.tensor((.1+.8*np.sin(np.linspace(0,np.pi,21)))[None,:],dtype=torch.float64)
        true_t=torch.tensor([[0.,3.2,9.1,15.3]],dtype=torch.float64)
        true_f=torch.tensor([[5.,100.,40.,5.]],dtype=torch.float64)
        base=torch.tensor([[95.,20.,850.,.4,.173]],dtype=torch.float64,requires_grad=True)
        def loss(a):
            aa,_=consistent_anchors(a,'impulse');t,f,_=reconstruct(raw,aa)
            return pwl_metrics(true_t,true_f,t,f)[0]
        self.assertTrue(torch.autograd.gradcheck(loss,(base,),eps=1e-6,atol=2e-4,rtol=2e-3))
    def test_neural_repeat(self):
        from joint_model import train_network
        f=self.train(12);cfg=CONFIG|dict(device='cpu',cuda_graph=False,maximum_epochs=2,early_stopping_check_every=1)
        rep=Representation(joint_specs(cfg)[0],cfg).fit(f,f[TARGETS].to_numpy(),self.curves)
        arrays=rep.arrays(f,f[TARGETS].to_numpy(),self.curves,True)
        a,_=train_network(rep,arrays,cfg,20260519,epochs=2)
        b,_=train_network(rep,arrays,cfg,20260519,epochs=2)
        for name,value in a.state_dict().items():np.testing.assert_array_equal(value,b.state_dict()[name])
    def test_zero_residual(self):
        f=self.train();spec=joint_specs(CONFIG)[0]
        rep=Representation(spec,CONFIG).fit(f,f[TARGETS].to_numpy(),self.curves)
        network=ResidualNetwork(rep).double();arr=rep.arrays(f,f[TARGETS].to_numpy(),leave_self_out=True)
        pack=tensor_pack(arr,'cpu',torch.float64);raw,c,coef=network(pack['x'],pack['template'])
        expected=torch.sigmoid(pack['template'])*network.mask+network.peak
        np.testing.assert_allclose(raw.detach(),expected,atol=1e-12);np.testing.assert_allclose(c.detach(),0,atol=1e-12)
        np.testing.assert_allclose(coef.detach(),0,atol=1e-12)
    def test_classic_batch_and_labels(self):
        f=self.train();held=self.frame[self.frame.split=='train'].iloc[30:33].copy()
        model=ClassicalModel(classic_specs(CONFIG)[0],CONFIG).fit(f,f[TARGETS].to_numpy(),self.curves)
        base=held[TARGETS].to_numpy();aa,p,_=model.predict(held,base)
        held[TARGETS]=100000
        bb,q,_=model.predict(held,base);np.testing.assert_allclose(aa,bb)
        for j in range(len(held)):
            _,single,_=model.predict(held.iloc[[j]],base[[j]])
            np.testing.assert_allclose(p[j][1],single[0][1],atol=1e-12)
    def test_legacy_batch(self):
        from controls import legacy
        _,analog=legacy();x=pd.DataFrame(np.arange(30.).reshape(10,3));y=np.array([np.sin(np.linspace(0,np.pi,200))*(.2+i/12) for i in range(10)])
        for cls in (analog.LocalKNNCurveRegressor,analog.PeakWarpLocalKNNRegressor):
            m=cls(preprocessing(),k=3) if cls is analog.LocalKNNCurveRegressor else cls(preprocessing(),np.linspace(0,1,200),k=3)
            m.fit(x,y);q=x.iloc[[1,4,8]]+.125
            np.testing.assert_allclose(m.predict(q),np.vstack([m.predict(q.iloc[[j]]) for j in range(3)]),atol=1e-12)
    def test_negative_not_clipped(self):
        m=physical_metrics(np.array([0.,1.,2.]),np.array([0.,1.,0.]),np.array([0.,1.,2.]),np.array([0.,100.,0.]))
        self.assertLess(m['curve_R2'],0)
    def test_zero_prediction_response(self):
        a=anchors_from_curve((np.array([0.,1.,2.]),np.zeros(3)))
        self.assertEqual(a[0],0);self.assertTrue(np.isnan(a[3]))
    def test_degenerate_r2(self):self.assertTrue(np.isnan(safe_r2([1,1.00000001],[10,12])))
    def test_guard_no_relaxation(self):
        self.assertIsNone(choose([dict(anchor_protection=False,median_R2=.99)],CONFIG))


if __name__=='__main__':unittest.main()
