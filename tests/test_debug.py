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
        'df_address', 'df_address_similar', 'address_difference'
    ]))
    assert (similar['score']>threshold).equals(pd.Series([True, True, False, False, False, False]))
    assert similar['threshold'].equals(pd.Series([True, True, False, False, False, False]))
    assert similar['column'].equals(pd.Series(['Address0']*6))
    assert similar['df_index'].equals(pd.Series([0,2,1,1,0,2], dtype='Int64'))
    assert similar['df_index_similar'].equals(pd.Series([2,0,0,2,1,1], dtype='Int64'))
    assert similar['df_address'].equals(pd.Series([
        '3148 amy falls mission reedmouth nv 56583',
        '3148 w amy falls mission reedmouth nv 56583',
        '1111 e amy falls mission reedmouth nv 56583',
        '1111 e amy falls mission reedmouth nv 56583',
        '3148 amy falls mission reedmouth nv 56583',
        '3148 w amy falls mission reedmouth nv 56583'
    ], dtype='string'))
    assert similar['df_address_similar'].equals(pd.Series([
        '3148 w amy falls mission reedmouth nv 56583',
        '3148 amy falls mission reedmouth nv 56583',
        '3148 amy falls mission reedmouth nv 56583',
        '3148 w amy falls mission reedmouth nv 56583',
        '1111 e amy falls mission reedmouth nv 56583',
        '1111 e amy falls mission reedmouth nv 56583'
    ], dtype='string'))

    assert len(in_cluster)==2
    assert len(out_cluster)==4


def test_address_two_df(df_samples):

    threshold = 0.7
    er = entity_resolver(df_samples[0], df_samples[1])
    er.compare('address', columns=['Address0','Address1'], threshold=threshold)

    similar, in_cluster, out_cluster = er.debug_similar('address')

    assert similar.columns.equals(pd.Index([
        'score', 'threshold', 'column', 'id_similar', 
        'df_index', 'df2_index','df_index_similar', 'df2_index_similar',
        'df_address', 'df_address_similar', 'df2_address', 'df2_address_similar',
        'address_difference'
    ]))
    assert (similar['score']>threshold).equals(pd.Series([True, True, False, False, False, False, False, False]))
    assert similar['threshold'].equals(pd.Series([True, True, False, False, False, False, False, False]))
    assert similar['column'].equals(pd.Series(['Address0']*7+['Address1']))
    assert similar['df_index'].equals(pd.Series([0,2,1,1,0,2,pd.NA,3], dtype='Int64'))
    assert similar['df2_index'].equals(pd.Series([pd.NA,pd.NA,pd.NA,pd.NA,pd.NA,pd.NA,1,pd.NA], dtype='Int64'))
    assert similar['df_index_similar'].equals(pd.Series([2,0,0,2,1,1,3,pd.NA], dtype='Int64'))
    assert similar['df2_index_similar'].equals(pd.Series([pd.NA,pd.NA,pd.NA,pd.NA,pd.NA,pd.NA,pd.NA,1], dtype='Int64'))

    assert len(in_cluster)==2
    assert len(out_cluster)==6