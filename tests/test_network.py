from itertools import chain

import pandas as pd

from entity_network.entity_resolver import entity_resolver

import sample

def check_network(network_id, network_map, columns, sample_id, sample_map, n_duplicates):

    record_count = {
        'df_index': sample_map['df_index'].str.len().max()
    }
    if 'df2_index' in network_id.columns:
        record_count.update({
            'df2_index': sample_map['df2_index'].str.len().max()
        })

     # check expected network relationships exist
    for index, records in record_count.items():
        network_check = network_id[['network_id', index]].dropna()
        sample_check = sample_id[['sample_id', index]].dropna()
        check = network_check.merge(sample_check, on=index)
        check = check.groupby(['network_id','sample_id']).size()
        assert len(check)==n_duplicates
        assert (check==records).all()   

    # check relationship matching features are correct
    for index, records in record_count.items():
        sample_check = sample_map.explode(index)
        sample_check = sample_check.set_index(index)
        network_check = network_map.groupby(index)
        network_check = network_check.agg({f'{col}_id': 'unique' for col in columns.keys()})
        merged_check = network_check.merge(sample_check, left_index=True, right_index=True, suffixes=('_actual','_sample'))
        for category in columns.keys():
            id_names = [f'{category}_id_actual', f'{category}_id_sample']
            check = merged_check[id_names]
            check = check.apply(pd.Series.explode)
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
    network_id, network_feature = er.network()

    # NameC should not self match
    assert 2 not in network_id['df_index']
    assert 2 not in network_feature['df_index']