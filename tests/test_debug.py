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


def test_one_df(df_samples):

    er = entity_resolver(df_samples[0])
    er.compare('address', columns='Address0', threshold=0.7)

    comparison = er.index_comparison('address', index_df=[0, 3])

    actual = comparison[['df_index','df_index_similar']]
    expected = pd.DataFrame({
        'df_index': [0,0,3],
        'df_index_similar': [2, 1, 4]
    }, dtype='Int64')

    assert actual.equals(expected)


def test_two_dfs_index_df(df_samples):

    er = entity_resolver(df_samples[0], df_samples[1])
    er.compare('address', columns=['Address0', 'Address1'], threshold=0.7)

    comparison = er.index_comparison('address', index_df=[0,3])

    actual = comparison[['df_index','df_index_similar','df2_index','df2_index_similar']]
    expected = pd.DataFrame({
        'df_index': [0,0,0,3,3],
        'df_index_similar': [2, 1, pd.NA, pd.NA, 4],
        'df2_index': [pd.NA, pd.NA, 0, pd.NA, pd.NA],
        'df2_index_similar': [pd.NA, pd.NA, pd.NA, 1, pd.NA]
    }, dtype='Int64')

    assert actual.equals(expected)


def test_two_dfs_index_df2(df_samples):

    er = entity_resolver(df_samples[0], df_samples[1])
    er.compare('address', columns=['Address0', 'Address1'], threshold=0.7)

    comparison = er.index_comparison('address', index_df2=[0,1])

    actual = comparison[['df_index','df_index_similar','df2_index','df2_index_similar']]
    expected = pd.DataFrame({
        'df_index': [2, 0, 1, 3, pd.NA],
        'df_index_similar': [pd.NA, pd.NA, pd.NA, pd.NA, 4],
        'df2_index': [0,0,0,pd.NA, pd.NA],
        'df2_index_similar': [pd.NA, pd.NA, pd.NA, 1, 1]
    }, dtype='Int64')

    assert actual.equals(expected)