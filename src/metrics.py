"""No clipping of negative R2; exact physical-time union support metrics."""
import numpy as np
import pandas as pd
from common import TARGETS,physical_metrics,anchors_from_curve


def safe_r2(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    ss=float(np.sum((y-y.mean())**2));scale=max(float(np.mean(y*y)),1e-20)
    if len(y)<2 or ss/len(y)<1e-6*scale:return np.nan
    return float(1-np.sum((p-y)**2)/ss)


def anchor_summary(true,pred):
    result={}
    for j,t in enumerate(TARGETS):
        y,p=true[:,j],pred[:,j];e=p-y
        result[t]=dict(R2=safe_r2(y,p),MAE=float(abs(e).mean()),RMSE=float(np.sqrt(np.mean(e*e))),
            MdRE=float(np.median(abs(e)/np.maximum(abs(y),1e-12))*100),
            phase_absolute_percentage_points=float(abs(e).mean()*100) if t=='tp_ratio' else np.nan)
    return result


def metric_rows(frame,curves,anchors,predictions,extras=None,fold=None):
    out=[];extras=extras or {}
    for j,row in enumerate(frame.itertuples()):
        if not np.isfinite(anchors[j]).all() or not all(np.isfinite(x).all() for x in predictions[j]):
            raise FloatingPointError('Nonfinite prediction; case must not be dropped: '+row.curve_id)
        if not np.all(np.diff(predictions[j][0])>0):raise ValueError('Prediction times must increase: '+row.curve_id)
        actual=anchors_from_curve(predictions[j])
        record=dict(curve_id=row.curve_id,split=row.split,role=row.role,source_group_id=row.source_group_id,
            lineage_group_id=row.lineage_group_id,specimen_group=row.specimen_group,fold=fold,
            present_study=row.curve_id.startswith('PRESENT_'),force_nonnegative=bool(np.min(predictions[j][1])>=0),
            zero_force_history=bool(actual[0]<=0),
            **physical_metrics(*curves[row.curve_id],*predictions[j]))
        for k,t in enumerate(TARGETS):record.update({t+'_true':float(getattr(row,t)),t+'_pred':float(anchors[j,k]),t+'_curve':float(actual[k])})
        for name,value in extras.items():
            if np.asarray(value).ndim==1:record[name]=float(value[j])
        out.append(record)
    return out


def rank_candidate(number,kind,spec,rows,baseline,config):
    df=pd.DataFrame(rows);am=anchor_summary(df[[t+'_true' for t in TARGETS]].to_numpy(),df[[t+'_pred' for t in TARGETS]].to_numpy())
    protect=all(am[t]['R2']>=baseline[t]['R2']-config['anchor_protection_R2_drop'] and
        am[t]['MdRE']<=baseline[t]['MdRE']*config['anchor_protection_MdRE_factor']+1e-12 for t in TARGETS[:3])
    complexity=(spec.get('width',1)*spec['k'] if kind=='joint' else spec['k'])
    return dict(number=number,kind=kind,spec=spec,median_R2=float(df.curve_R2.median()),
        negative_count=int((df.curve_R2<0).sum()),NRMSE_p90=float(df.NRMSE.quantile(.9)),
        complexity=complexity,anchor_protection=bool(protect),anchor_metrics=am)


def choose(rankings,config):
    eligible=[r for r in rankings if r['anchor_protection']]
    if not eligible:return None
    best=max(r['median_R2'] for r in eligible)
    return min([r for r in eligible if r['median_R2']>=best-config['near_best_median_R2']],
        key=lambda r:(r['negative_count'],r['NRMSE_p90'],r['complexity'],r['number']))


def summaries(df):
    aa=[];cc=[]
    for (seed,arm,split),part in df.groupby(['split_seed','arm','split']):
        populations={'all':part,'ordinary_RC':part[part.role=='primary'],'numerical':part[part.role=='numerical_aux'],
            'special':part[part.role=='shape_aux'],'present_study':part[part.present_study]}
        for population,group in populations.items():
            if group.empty:continue
            base=dict(split_seed=seed,arm=arm,split=split,population=population,n=len(group))
            for target,m in anchor_summary(group[[t+'_true' for t in TARGETS]].to_numpy(),group[[t+'_pred' for t in TARGETS]].to_numpy()).items():
                aa.append(base|dict(target=target,**m))
            cc.append(base|dict(median_curve_R2=float(group.curve_R2.median()),MAE_kN=float(group.MAE_kN.median()),
                RMSE_kN=float(group.RMSE_kN.median()),median_NRMSE=float(group.NRMSE.median()),
                NRMSE_p90=float(group.NRMSE.quantile(.9)),negative_R2_count=int((group.curve_R2<0).sum()),
                minimum_curve_R2=float(group.curve_R2.min()),force_nonnegative=bool(group.force_nonnegative.all())))
    return pd.DataFrame(aa),pd.DataFrame(cc)


def paired_bootstrap(first,second,replicates,seed):
    left=first.set_index('curve_id').sort_index();right=second.set_index('curve_id').loc[left.index]
    assert left.index.equals(right.index) and (left.lineage_group_id==right.lineage_group_id).all()
    groups=sorted(left.lineage_group_id.unique());blocks={g:np.flatnonzero(left.lineage_group_id.to_numpy()==g) for g in groups}
    rng=np.random.default_rng(seed);values={};valid=0
    for _ in range(replicates):
        draw=rng.choice(groups,len(groups),replace=True)
        if len(set(draw))<2:continue
        ix=np.concatenate([blocks[g] for g in draw]);a,b=left.iloc[ix],right.iloc[ix];valid+=1
        differences=dict(median_curve_R2=float(b.curve_R2.median()-a.curve_R2.median()),
            median_NRMSE=float(b.NRMSE.median()-a.NRMSE.median()))
        for t in TARGETS:
            ra=safe_r2(a[t+'_true'],a[t+'_pred']);rb=safe_r2(b[t+'_true'],b[t+'_pred'])
            differences[t+'_R2']=rb-ra
        for key,value in differences.items():
            if np.isfinite(value):values.setdefault(key,[]).append(value)
    rows=[]
    for metric,v in values.items():
        enough=len(v)>=max(100,int(.8*replicates))
        if metric=='median_curve_R2':point=float(right.curve_R2.median()-left.curve_R2.median())
        elif metric=='median_NRMSE':point=float(right.NRMSE.median()-left.NRMSE.median())
        else:
            target=metric[:-3];point=safe_r2(right[target+'_true'],right[target+'_pred'])-safe_r2(left[target+'_true'],left[target+'_pred'])
        rows.append(dict(metric=metric,observed_difference=point,lower95=float(np.quantile(v,.025)) if enough else np.nan,
            upper95=float(np.quantile(v,.975)) if enough else np.nan,valid_resamples=len(v),requested=replicates,
            status='OK' if enough else 'INSUFFICIENT_NONDEGENERATE_RESAMPLES',source_groups=len(groups)))
    return rows
