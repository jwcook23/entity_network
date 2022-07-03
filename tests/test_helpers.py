import pandas as pd
import numpy as np
from entity_network import _helpers

def test_column_group():

    df = pd.DataFrame([
        [1,5],
        [1,8],
        [3,4],
        [2,8],
        [9,pd.NA],
        [0,pd.NA]
    ], dtype='Int64')

    df = _helpers._row_connected(df)
    assert df['network_id'].equals(pd.Series([0,0,2,0,4,5]))


    df = pd.DataFrame([
        [1,2,3],
        [0,4,2],
        [4,2,5],
        [6,1,1],
        [1,3,5],
        [3,0,1],
        [0,4,np.nan],
    ])

    df = _helpers._row_connected(df)
    assert df['network_id'].equals(pd.Series([0, 1, 0, 3, 0, 3, 1]))