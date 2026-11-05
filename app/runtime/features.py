import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import RAW


def feature_matrix(frame, scheme):
    x = frame.loc[:, RAW].astype(float).copy()
    if scheme == 'physics':
        area = x.b*x.h
        x['energy_J'] = .5*x.m*x.v**2
        x['momentum_Ns'] = x.m*x.v
        x['section_aspect'] = x.b/x.h.replace(0,np.nan)
        x['length_depth_ratio'] = x.H0/x.h.replace(0,np.nan)
        x['length_effective_depth_ratio'] = x.H0/x.d.replace(0,np.nan)
        x['cover_depth_ratio'] = x.c/x.h.replace(0,np.nan)
        x['axial_ratio_reported_strength'] = x.N*1000/(x.fc_reported*area)
        x['longitudinal_ratio'] = x.Asl/area
        no_stirrups = (x['dt']==0)&(x.Asv==0)
        x['transverse_ratio'] = np.where(no_stirrups,0,x.Asv/(x.b*x.s.replace(0,np.nan)))
        x['steel_force_scale_kN'] = x.Asl*x.fyl/1000
        x['stirrup_force_per_length'] = np.where(no_stirrups,0,x.Asv*x.fyt/x.s.replace(0,np.nan))
        # Geometry only: no unjustified elastic modulus conversion across strength types.
        x['section_J_mm4'] = x.b*x.h**3/12
    elif scheme != 'raw':
        raise ValueError(scheme)
    return x.replace([np.inf,-np.inf],np.nan).to_numpy(float)


def preprocessing():
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                         VarianceThreshold(1e-14),StandardScaler())


def shape_inputs(frame, anchors, scheme):
    anchors = np.asarray(anchors,float)
    pos = np.log(np.maximum(anchors[:,:3],1e-10))
    b = np.clip(anchors[:,3:],1e-6,1-1e-6)
    return np.column_stack([feature_matrix(frame,scheme),pos,np.log(b/(1-b))])
