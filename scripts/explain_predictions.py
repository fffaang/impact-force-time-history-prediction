"""Permutation SHAP of the frozen primary five-output function, including derived Astar."""
from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app/runtime'))
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from common import RAW,TARGETS
from anchors import predict_anchors
from classical import consistent_anchors


def explain(output,check):
    frame=pd.read_csv(ROOT/'data/archive_199.csv')
    split=pd.read_csv(ROOT/'data/splits/20261102.csv').set_index('curve_id')
    frame['split']=split.loc[frame.curve_id,'split'].to_numpy()
    train=frame[(frame.split=='train')&(frame.role!='quarantine')].reset_index(drop=True)
    test=frame[frame.split=='test'].reset_index(drop=True)
    model=joblib.load(ROOT/'models/20261102/C_final.joblib')
    rng=np.random.default_rng(20261103)
    indices=np.sort(rng.choice(len(train),24,replace=False))
    background=train.iloc[indices][RAW].to_numpy(float)
    assert not set(train.iloc[indices].curve_id)&set(test.curve_id)
    def function(x):
        independent=predict_anchors(model.anchors,pd.DataFrame(x,columns=RAW))
        aa,_=consistent_anchors(independent,'impulse')
        return np.column_stack((np.log(aa[:,:3]),logit(np.clip(aa[:,3:],1e-12,1-1e-12))))
    expectation=function(background).mean(0)
    orders=[]
    for _ in range(32):
        order=rng.permutation(len(RAW));orders.extend([order,order[::-1]])
    orders=np.asarray(orders);rows=[];maximum=0.
    for case in range(len(test)):
        point=test.iloc[case][RAW].to_numpy(float);contributions=[]
        for block in range(0,len(orders),8):
            tensors=[]
            for order in orders[block:block+8]:
                hybrid=background.copy();path=[hybrid.copy()]
                for j in order:
                    hybrid[:,j]=point[j];path.append(hybrid.copy())
                tensors.append(np.stack(path))
            tensor=np.stack(tensors)
            values=function(tensor.reshape(-1,len(RAW))).reshape(len(tensors),len(RAW)+1,len(background),5).mean(2)
            for k,order in enumerate(orders[block:block+8]):
                phi=np.empty((len(RAW),5));phi[order]=np.diff(values[k],axis=0);contributions.append(phi)
        contributions=np.asarray(contributions);phi=contributions.mean(0);prediction=function(point[None,:])[0]
        error=float(np.max(abs(expectation+phi.sum(0)-prediction)));maximum=max(maximum,error)
        assert error<1e-8
        for j,feature in enumerate(RAW):
            for k,target in enumerate(TARGETS):
                rows.append(dict(curve_id=test.iloc[case].curve_id,target=target,feature=feature,value=point[j],
                    shap=float(phi[j,k]),base_value=float(expectation[k]),output=float(prediction[k]),
                    first_half=float(contributions[:32,j,k].mean()),second_half=float(contributions[32:,j,k].mean()),
                    scale='log' if k<3 else 'logit'))
        print('SHAP',case+1,'/',len(test),'additivity',error,flush=True)
    df=pd.DataFrame(rows)
    if check:
        keys=['curve_id','target','feature'];cols=['shap','base_value','output','first_half','second_half']
        actual=df.set_index(keys).sort_index();expected=pd.read_csv(ROOT/'results/production/shap_values.csv').set_index(keys).sort_index()
        np.testing.assert_allclose(actual[cols],expected[cols],rtol=1e-9,atol=1e-8)
    output.mkdir(parents=True,exist_ok=True)
    df.to_csv(output/'shap_values.csv',index=False)
    importance=df.groupby(['target','feature'],sort=False).shap.apply(lambda x:abs(x).mean()).rename('mean_abs_shap').reset_index()
    importance['rank']=importance.groupby('target').mean_abs_shap.rank(method='first',ascending=False).astype(int)
    importance.sort_values(['target','rank']).to_csv(output/'shap_importance.csv',index=False)
    qa=dict(passed=True,maximum_additivity_error=maximum,stored_values_checked=check,records=20,
            background_ids=train.iloc[indices].curve_id.tolist(),permutations=64,seed=20261103,
            actual_derived_output=True,raw_inputs_perturbed=True,causal_interpretation=False)
    (output/'verification.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'output/shap');p.add_argument('--check',action='store_true')
    a=p.parse_args();explain(a.output,a.check)
