from itertools import chain

import pandas as pd

from entity_network.entity_resolver import entity_resolver

import sample

def check_network(network_id, network_map, columns, sample_id, sample_map, n_duplicates):

    # extract expected matching indices and count of mactching records in each dataframe from sample
    matching_records = {
        'df_index': sample_map['df_index'].str.len().iloc[0]
    }
    if 'df2_index' in network_id.columns:
        matching_records.update({
            'df2_index': sample_map['df2_index'].str.len().iloc[0]
        })

    # check network_id matches for indcies according to sample_id column
    for df, records in matching_records.items():
        # extract indices from current dataframe
        network_check = network_id[['network_id', df]].dropna()
        sample_check = sample_id[['sample_id', df]].dropna()
        # merge on index for current dataframe between the sample and actual result
        check = network_check.merge(sample_check, on=df)
        # compare the matching of indices sets
        check = check.groupby(['network_id','sample_id']).size()
        assert len(check)==n_duplicates
        assert (check==records).all()   

    # check relationship matching features are correct
    for df, records in matching_records.items():
        # extract the current dataframe
        sample_check = sample_map.explode(df)
        sample_check = sample_check.set_index(df)
        # group the current dataframe
        network_check = network_map.groupby(df)
        network_check = network_check.agg({f'{col}_id': 'unique' for col in columns.keys()})
        # merge on index for current dataframe between the sample and actual result
        merged_check = network_check.merge(sample_check, left_index=True, right_index=True, suffixes=('_actual','_sample'))
        # check each feature
        for category in columns.keys():
            # extract the current feature being compared
            id_names = [f'{category}_id_actual', f'{category}_id_sample']
            check = merged_check[id_names]
            check = check.apply(pd.Series.explode)
            # compare the matching of feature id sets
            check = check.groupby(id_names)
            check = check.size()
            assert len(check)==n_duplicates*sample_check[f'{category}_id'].str.len().max()
            assert (check==records).all()


def test_single_category_exact():

    n_unique = 1000
    n_duplicates = 30

    # generate sample data
    sample_df = sample.unique_records(n_unique)
    columns = {'phone': ['HomePhone','WorkPhone','CellPhone']}
    sample_df, sample_id, sample_map = sample.duplicate_records(sample_df, n_duplicates, columns)

    # compare and derive network
    er = entity_resolver(sample_df)
    er.compare('phone', columns=columns['phone'])
    network_id, network_map, _ = er.network()

    # assert results
    check_network(network_id, network_map, columns, sample_id, sample_map, n_duplicates)


def test_all_category_exact():

    n_unique = 1000
    n_duplicates = 30

    # generate sample data
    sample_df = sample.unique_records(n_unique)
    columns = {
        'phone': ['HomePhone','WorkPhone','CellPhone'],
        'email': ['Email'],
        'address': ['Address']
    }
    sample_df, sample_id, sample_map = sample.duplicate_records(sample_df, n_duplicates, columns)

    # compare and derive network
    er = entity_resolver(sample_df)
    for category, cols in columns.items():
        er.compare(category, columns=cols, text_cleaner=None)
    network_id, network_map, _ = er.network()

    # assert results
    check_network(network_id, network_map, columns, sample_id, sample_map, n_duplicates)


def test_two_dfs_exact():

    n_unique = 1000
    n_duplicates = 30

    # generate sample data
    df1 = sample.unique_records(n_unique)
    columns = {
        'phone': ['HomePhone','WorkPhone','CellPhone', 'Phone'],
        'email': ['Email','EmailAddress'],
        'address': ['Address','StreetAddress']
    }
    df2, sample_id, sample_map = sample.duplicate_df(df1, n_duplicates, columns)

    # compare and derive network
    er = entity_resolver(df1, df2)
    for category, cols in columns.items():
        er.compare(category, columns=cols)
    network_id, network_map, _ = er.network()

    # assert results
    check_network(network_id, network_map, columns, sample_id, sample_map, n_duplicates)    


def test_record_self_exact():

    sample_df = pd.DataFrame({
        'Name': ['NameA', 'NameB','NameC'],
        'PhoneA': ['123456789', '123456789', '1112223333'],
        'PhoneB': ['123456789', '123456789', '1112223333']
    })

    # compare and derive network
    er = entity_resolver(sample_df)
    er.compare('phone', columns=['PhoneA', 'PhoneB'])
    network_id, network_map, network_feature = er.network()

    # NameC should not self match
    assert 2 not in network_id['df_index']
    assert 2 not in network_map['df_index']
    assert 2 not in network_feature['phone']['df_index']