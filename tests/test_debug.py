import pandas as pd
import pytest

from entity_network.entity_resolver import entity_resolver

@pytest.fixture()
def df_samples():

    df1 = pd.DataFrame({
        'Address0': [
            '3148 amy falls mission reedmouth nv 56583',
            '1111 e amy falls mission reedmouth nv 56583',
            '3148 w amy falls mission reedmouth nv 56583',
            '4611 59th way lauderhill al 23790',
            # TODO: this should be included in the network for two dataframes
            '4611 59th way lauderhill al 23790',
        ]
    })

    df2 = pd.DataFrame({
        'Address1': [
            '3148 w amy falls mission reedmouth nv 56583',
            '4611 59th ter ft lauderdale al 23790'
        ]
    })

    return df1, df2

def test_address_one_df(df_samples):

    threshold = 0.7
    er = entity_resolver(df_samples[0])
    er.compare('address', columns='Address0', threshold=threshold)

    similar, in_cluster, out_cluster = er.debug_similar('address')

    expected = pd.DataFrame({
        'threshold': pd.Series([True, True, False, False, False, False], dtype='bool'),
        'column': pd.Series(['Address0']*6),
        'df_index': pd.Series([2,0,0,2,1,1], dtype='Int64'),
        'df_index_similar': pd.Series([0,2,1,1,0,2], dtype='Int64')
    })
    expected.index = pd.MultiIndex.from_arrays([[2,0,0,2,1,1],[0,2,1,1,0,2]], names=('node','node_similar'))

    assert similar.columns.equals(pd.Index([
        'score', 'threshold', 'column', 'id_similar',
        'df_index', 'df_index_similar',
        'address_df_value', 'address_df_similar_value'
    ]))
    assert (similar['score']>threshold).equals(expected['threshold'])
    assert similar['threshold'].equals(expected['threshold'])
    assert similar['column'].equals(expected['column'])
    assert similar['df_index'].equals(expected['df_index'])
    assert similar['df_index_similar'].equals(expected['df_index_similar'])
    assert len(in_cluster)==2
    assert sum(in_cluster.columns.isin(['address_df_diff','address_df_similar_diff']))==2
    assert len(out_cluster)==4
    assert sum(in_cluster.columns.isin(['address_df_diff','address_df_similar_diff']))==2


def test_address_two_df(df_samples):

    threshold = 0.7
    er = entity_resolver(df_samples[0], df_samples[1])
    er.compare('address', columns={'df': 'Address0', 'df2': 'Address1'}, threshold=threshold)

    similar, in_cluster, out_cluster = er.debug_similar('address')

    assert similar.columns.equals(pd.Index([
        'score', 'threshold', 'column', 'id_similar', 'df_index', 'df2_index',
        'address_df_value', 'column_df2', 'address_df2_similar_value',
    ]))
    assert (similar['score']>threshold).equals(pd.Series([True, True, False, False, False, False, False, False]))
    assert similar['threshold'].equals(pd.Series([True, True, False, False, False, False, False, False]))
    assert similar['column'].equals(pd.Series(['Address0']*7+['Address1']))
    assert similar['df_index'].equals(pd.Series([0,2,1,1,0,2,pd.NA,3], dtype='Int64'))
    assert similar['df2_index'].equals(pd.Series([pd.NA,pd.NA,pd.NA,pd.NA,pd.NA,pd.NA,1,pd.NA], dtype='Int64'))
    assert similar['df_index_similar'].equals(pd.Series([2,0,0,2,1,1,3,pd.NA], dtype='Int64'))
    assert similar['df2_index_similar'].equals(pd.Series([pd.NA,pd.NA,pd.NA,pd.NA,pd.NA,pd.NA,pd.NA,1], dtype='Int64'))
    assert len(in_cluster)==2
    assert 'address_difference' in in_cluster
    assert len(out_cluster)==6
    assert 'address_difference' in out_cluster