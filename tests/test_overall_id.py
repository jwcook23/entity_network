import pandas as pd
import numpy as np
from entity_network import _helpers

def test_two_cols():
    df = pd.DataFrame([
        [0, 1, 5, 0],
        [1, 1, 8, 0],
        [2, 3, 4, 2],
        [3, 2, 8, 0],
    ], columns=['node','0_id','1_id','network_id'])
    df[['node','0_id','1_id']] = df[['node','0_id','1_id']].astype('Int64')
    df['network_id'] = df['network_id'].astype('int64')

    network_id, network_map = _helpers.overall_id(df.drop(columns='network_id').copy())
    assert network_id.equals(df[['node','network_id']])
    assert network_map.equals(df)


def test_two_cols_nans():

    df = pd.DataFrame([
        [0, 1, 5, 0],
        [1, 1, 8, 0],
        [2, 3, 4, 2],
        [3, 2, 8, 0],
        [4, 9, pd.NA, 4],
        [5, 0, pd.NA, 5]
    ], columns=['node','0_id','1_id','network_id'])
    df[['node','0_id','1_id']] = df[['node','0_id','1_id']].astype('Int64')
    df['network_id'] = df['network_id'].astype('int64')

    network_id, network_map = _helpers.overall_id(df.drop(columns='network_id').copy())
    assert network_id.equals(df[['node','network_id']])
    assert network_map.equals(df)


def test_three_cols():

    df = pd.DataFrame([
        [0,1,2,3,0],
        [1,0,4,2,1],
        [2,4,2,5,0],
        [3,6,1,1,3],
        [4,1,3,5,0],
        [5,3,0,1,3],
        [6,0,4,np.nan,1],
    ], columns=['node','0_id','1_id','2_id','network_id'])
    df[['node','0_id','1_id','2_id']] = df[['node','0_id','1_id','2_id']].astype('Int64')
    df['network_id'] = df['network_id'].astype('int64')

    network_id, network_map = _helpers.overall_id(df.drop(columns='network_id').copy())
    assert network_id.equals(df[['node','network_id']])
    assert network_map.equals(df)


def test_large_matrix():

    df = [
        [1, 5, 0],
        [1, 8, 0],
        [3, 4, 2],
        [2, 8, 0],
        [9, pd.NA, 4],
        [0, pd.NA, 5]
    ]*10000
    df = pd.DataFrame(df, columns=['0_id','1_id','network_id'])
    df['node'] = range(0,len(df))

    df[['node','0_id','1_id']] = df[['node','0_id','1_id']].astype('Int64')
    df['network_id'] = df['network_id'].astype('int64')

    network_id, network_map = _helpers.overall_id(df.drop(columns='network_id').copy())
    assert network_id.equals(df[['node','network_id']])
    assert network_map.equals(df[network_map.columns])