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

    assert similar.columns.equals(pd.Index([
        'score', 'threshold', 'column', 'id_similar',
        'df_index', 'df_index_similar',
        'address_df_value', 'address_df_similar_value'
    ]))
    assert (similar['score']>threshold).equals(pd.Series([True, True, False, False, False, False], dtype='boolean'))
    assert similar['threshold'].equals(pd.Series([True, True, False, False, False, False], dtype='boolean'))
    assert similar['column'].equals(pd.Series(['Address0']*6))
    assert similar['df_index'].equals(pd.Series([2,0,0,2,1,1], dtype='Int64'))
    assert similar['df_index_similar'].equals(pd.Series([0,2,1,1,0,2], dtype='int64'))

    assert len(in_cluster)==2
    assert 'difference' in in_cluster.columns
    assert len(out_cluster)==4
    assert 'difference' in out_cluster.columns


def test_address_two_df(df_samples):

    threshold = 0.7
    er = entity_resolver(df_samples[0], df_samples[1])
    er.compare('address', columns={'df': 'Address0', 'df2': 'Address1'}, threshold=threshold)

    similar, in_cluster, out_cluster = er.debug_similar('address')

    assert similar.columns.equals(pd.Index([
        'score', 'threshold', 'column', 'id_similar', 'df_index', 'df2_index',
        'address_df_value', 'column_df2', 'address_df2_similar_value',
    ]))
    assert (similar['score']>threshold).equals(pd.Series([True, True, False, True, True], dtype='boolean'))
    assert similar['threshold'].equals(pd.Series([True, True, False, True, True], dtype='boolean'))
    assert similar['column'].equals(pd.Series(['Address0']*5))
    assert similar['column_df2'].equals(pd.Series(['Address1']*5))
    assert similar['df_index'].equals(pd.Series([2,0,1,3,4], dtype='Int64'))
    assert similar['df2_index'].equals(pd.Series([0,0,0,1,1], dtype='Int64'))

    assert len(in_cluster)==4
    assert 'difference' in in_cluster.columns
    assert len(out_cluster)==1
    assert 'difference' in out_cluster.columns