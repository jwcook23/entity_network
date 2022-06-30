from itertools import chain

import pandas as pd

from entity_network.entity_resolver import entity_resolver

import sample

def test_single_category_exact():

    n_unique = 1000
    n_duplicates = 30

    # generate sample data
    sample_df = sample.unique_records(n_unique)
    columns = {'phone': ['HomePhone','WorkPhone','CellPhone']}
    sample_df, sample_id, sample_feature = sample.duplicate_records(sample_df, n_duplicates, columns)

    # compare and derive network
    er = entity_resolver(sample_df)
    er.compare('phone', columns=columns['phone'])
    network_id, network_feature = er.network()

    # check expected network relationships exist
    relation = network_id.merge(sample_id, on='df_index')
    relation = relation.groupby(['network_id','sample_id']).size()
    assert len(relation)==n_duplicates
    assert (relation==2).all()

    # check relationship matching features
    feature = network_feature.merge(sample_feature, on=['df_index', 'column'])
    feature = feature.groupby(['feature_id','sample_id']).size()
    assert len(feature)==n_duplicates*len(columns['phone'])
    assert (feature==2).all()


def test_all_category_exact():

    n_unique = 1000
    n_duplicates = 30

    # generate sample data
    sample_df = sample.unique_records(n_unique)
    columns_compare = {
        'phone': ['HomePhone','WorkPhone','CellPhone'],
        'email': ['Email'],
        'address': ['Address']
    }
    sample_df, sample_id, sample_feature = sample.duplicate_records(sample_df, n_duplicates, columns_compare)

    # compare and derive network
    er = entity_resolver(sample_df)
    for category, cols in columns_compare.items():
        er.compare(category, columns=cols, text_cleaner=None)
    network_id, network_feature = er.network()

     # check expected network relationships exist
    relation = network_id.merge(sample_id, on='df_index')
    relation = relation.groupby(['network_id','sample_id']).size()
    assert len(relation)==n_duplicates
    assert (relation==2).all()   

    # check relationship matching features
    feature = network_feature.merge(sample_feature, on=['df_index', 'column'])
    feature = feature.groupby(['feature_id','sample_id']).size()
    assert len(feature)==n_duplicates*len(list(chain(*columns_compare.values())))
    assert (feature==2).all()


def test_two_dfs_exact():

    n_unique = 1000
    n_duplicates = 30

    # generate sample data
    df1 = sample.unique_records(n_unique)
    df2, sample_id, sample_feature, columns_compare, columns_match = sample.similar_df(df1, n_duplicates)

    # compare and derive network
    er = entity_resolver(df1, df2)
    for category, cols in columns_compare.items():
        er.compare(category, columns=cols)
    network_id, network_feature = er.network()

    # check expected network pairs exist
    for index, match in columns_match.items():
        relation = network_id.merge(sample_id[[index,'sample_id']].dropna(), on=index)
        relation = relation.groupby(['network_id','sample_id']).size()
        assert len(relation)==n_duplicates
        assert (relation==match['size']).all()

    # check relationship matching features
    for index, match in columns_match.items():
        feature = network_feature.merge(sample_feature[[index,'sample_id','column']].dropna(), on=[index, 'column'])
        feature = feature.groupby(['feature_id','sample_id']).size()
        assert len(feature)==n_duplicates*len(match['columns'])
        assert (feature==match['size']).all()


def test_record_self_exact():

    sample_df = pd.DataFrame({
        'Name': ['NameA', 'NameB','NameC'],
        'PhoneA': ['123456789', '123456789', '1112223333'],
        'PhoneB': ['123456789', '123456789', '1112223333']
    })

    # compare and derive network
    er = entity_resolver(sample_df)
    er.compare('phone', columns=['PhoneA', 'PhoneB'])
    network_id, network_feature = er.network()

    # NameC should not self match
    assert 2 not in network_id['df_index']
    assert 2 not in network_feature['df_index']