"""Synthetic no-external-data tests of V8 feature logic."""
import numpy as np,pandas as pd
from v8_features import shape_features,premium_features,trend_features

def main():
    ref=pd.DataFrame({'MONTH_INDEX':[1,2,3,3],'TOWN':['A','A','A','B'],
                      'FLAT_TYPE':['4-room','4-room','4-room','3-room'],
                      'MONTHLY_RENT':[1000.,2000.,3000.,1400.]})
    q=pd.DataFrame({'MONTH_INDEX':[2,3,4,5,4],'TOWN':['A','A','A','A','Z'],
                    'FLAT_TYPE':['4-room']*5})
    a=trend_features(ref,q,cutoff_month_index=4)
    assert a.loc[2].equals(a.loc[3])
    q2=q.copy();q2['MONTHLY_RENT']=999999
    pd.testing.assert_frame_equal(a,trend_features(ref,q2,cutoff_month_index=4))
    changed=ref.copy();changed.loc[changed.MONTH_INDEX==3,'MONTHLY_RENT']=900000
    b=trend_features(changed,q,cutoff_month_index=4)
    pd.testing.assert_series_equal(a.loc[1],b.loc[1])
    back=trend_features(ref,q.iloc[::-1],cutoff_month_index=4)
    pd.testing.assert_frame_equal(back,a.iloc[::-1].reset_index(drop=True))
    x=pd.DataFrame({'TOWN':['A','B'],'FLAT_TYPE':['4-room','executive'],
                    'FLAT_MODEL':['x','y'],'FLOOR_AREA_SQM':[90.,130.],
                    'LEASE_AGE':[25.,18.],'MRT_NEAREST_KM':[0.3,1.0],
                    'HIST_TOWN_TYPE_MEAN':[3000.,3900.],
                    'COMP_TYPE_AREA_MEAN':[3100.,3800.],
                    'COMP_STREET_TYPE_MEAN':[3200.,4000.],
                    'COMP_STREET_TYPE_AREA_MEAN':[3250.,4100.],
                    'COMP_BLOCK_TYPE_AREA_MEAN':[3300.,4150.]})
    assert shape_features(x)['V8_SHAPE_ROOM_LEVEL'].tolist()==[4.,6.]
    assert np.isfinite(premium_features(x).to_numpy()).all()
    print('PASS: V8 feature geometry, fold freeze, label exclusion, and row-order')
if __name__=='__main__':main()
