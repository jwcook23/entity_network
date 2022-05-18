from itertools import chain

import sample

from entity_network.entity_resolver import entity_resolver

def test_single_category():

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

def test_all_category():

    n_unique = 1000
    n_duplicates = 30

    # generate sample data
    sample_df = sample.unique_records(n_unique)
    columns = {
        'phone': ['HomePhone','WorkPhone','CellPhone'],
        'email': ['Email'],
        'address': ['Address']
    }
    sample_df, sample_id, sample_feature = sample.duplicate_records(sample_df, n_duplicates, columns)

    # compare and derive network
    er = entity_resolver(sample_df)
    for category, cols in columns.items():
        er.compare(category, columns=cols)
    network_id, network_feature = er.network()

     # check expected network relationships exist
    relation = network_id.merge(sample_id, on='df_index')
    relation = relation.groupby(['network_id','sample_id']).size()
    assert len(relation)==n_duplicates
    assert (relation==2).all()   

    # check relationship matching features
    feature = network_feature.merge(sample_feature, on=['df_index', 'column'])
    feature = feature.groupby(['feature_id','sample_id']).size()
    assert len(feature)==n_duplicates*len(list(chain(*columns.values())))
    assert (feature==2).all()


# def test_two_dfs():

#     n_unique = 1000
#     n_duplicates = 30

#     # generate sample data
#     df1 = sample.unique_records(n_unique)
#     df2, sample_id = sample.similar_df(df1, n_duplicates)

#     # compare and derive network
#     er = entity_resolver(df1, df2)
#     criteria = {
#         'phone': ['HomePhone','WorkPhone','CellPhone', 'Phone'],
#         'email': ['Email','EmailAddress'],
#         'address': ['Address','StreetAddress']
#     }
#     for category, columns in criteria.items():
#         er.compare(category, columns=columns)
#     network_id, network_feature, network_relation = er.network()

#     # check expected network pairs exist
#     result_df = sample_id['df'].merge(network_id['df'], on='index')
#     result_df2 = sample_id['df2'].merge(network_id['df2'], on='index')
#     result = result_df.merge(result_df2, on='sample_id', suffixes=('_df','_df2'))
#     assert (result['network_id_df']==result['network_id_df2']).all()
#     assert len(result)==n_duplicates

#     assert 1==1