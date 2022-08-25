import os

import pandas as pd

from entity_network.entity_resolver import entity_resolver

from .. import sample

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


def test_split_column():

    n_unique = 10

    df1 = sample.address_components(n_unique)

    df2 = pd.DataFrame({
        'Address': [df1.at[0,'Street']+' '+df1.at[0,'City']+' '+df1.at[0,'State']+' '+df1.at[0,'Zip']]
    })

    er = entity_resolver(df1, df2)
    columns = {'df': [['Street', 'City', 'State', 'Zip']], 'df2': 'Address'}
    er.compare('address', columns=columns)
    
    assert all(er.network_feature['address'].index == [0, 10])
    assert all(er.network_feature['address']['column'] == ['Street,City,State,Zip', 'Address'])
    assert all(er.network_feature['address']['id_exact'] == [0,0])
    assert all(er.network_feature['address']['id_similar'].isna())
    assert all(er.network_feature['address']['address_id'] == [0,0])


def test_similar_address():

    file_path = os.path.join('tests','similar_address.csv')

    df = pd.read_csv(file_path)
    df = df[df['threshold=0.7']==1]
    df1 = df[['Address0']]
    df2 = df[['Address1']]

    er = entity_resolver(df1, df2)
    er.compare('address', columns={'df': 'Address0', 'df2': 'Address1'}, threshold=0.7)
    er.network()

    assert len(er.network_summary)==len(df)
    assert (er.network_summary['df_index'].explode().sort_values()==df1.index).all()
    assert (er.network_summary['df2_index'].explode().sort_values()==df2.index).all()
    assert (er.network_summary['df_index']==er.network_summary['df2_index']).all()
    assert (er.network_summary['feature_match']=='address').all()
    assert er.network_summary.columns.equals(
        pd.Index(['df_index', 'df2_index', 'feature_match', 'df_column_address', 'df2_column_address', 'address_normalized', 'address_difference'])
    )

def test_similar_phone():

    df1 = pd.DataFrame({'MainPhone': ['555-456-7890']})
    df2 = pd.DataFrame({'WorkPhone': ['555-456-7890 extension 123']})

    er = entity_resolver(df1, df2)
    er.compare('phone', columns={'df': 'MainPhone', 'df2': 'WorkPhone'}, threshold=0.5)
    er.network()

    assert len(er.network_summary)==len(df1)
    assert (er.network_summary['df_index'].explode().sort_values()==df1.index).all()
    assert (er.network_summary['df2_index'].explode().sort_values()==df2.index).all()
    assert (er.network_summary['df_index']==er.network_summary['df2_index']).all()
    assert (er.network_summary['feature_match']=='phone').all()
    assert er.network_summary.columns.equals(
        pd.Index(['df_index', 'df2_index', 'feature_match', 'df_column_phone', 'df2_column_phone', 'phone_normalized', 'phone_difference'])
    )

def test_nothing_similar():

    df1 = pd.DataFrame({
        'AddressA': [
            '1234 S NameA Street Town, ST 56789',
            '1234 South NameA Street Town, ST 56789-0147'
        ],
    })
    df2 = pd.DataFrame({
        'AddressA': [
            '5678 N NameB Road Place, NA 01234',
        ],
    })

    er = entity_resolver(df1, df2)
    er.compare('address', columns={'df': 'AddressA', 'df2': 'AddressA'}, threshold=0.8)

    assert len(er.network_feature['address'])==0


def test_combine_similar_exact():


    df1 = pd.DataFrame({
        'AddressA': [
            '1234 SW NameA Street Town, ST 56789',
            '1234 South NameA Street Town, ST 56789-0147'
        ],
    })
    df2 = pd.DataFrame({
        'AddressA': [
            '1234 SE NameB Street Town, ST 56789',
            '1234 South NameA Street Town, ST 56789-0147'
        ],
    })


    er = entity_resolver(df1, df2)
    er.compare('address', columns={'df': 'AddressA', 'df2': 'AddressA'}, threshold=0.8)

    expected = pd.DataFrame({
        'column': ['AddressA']*4,
        'id_exact': [0,0, pd.NA, pd.NA],
        'id_similar': [0, 0, 0, 0],
        'address_id': [0]*4,
        'df_index': [1, pd.NA, 0, pd.NA],
        'df2_index': [pd.NA, 1, pd.NA, 0]
    }, index=pd.Index([1,3,0,2], name='node'))
    expected['column'] = expected['column'].astype('string')
    cols = ['id_exact','id_similar','df_index','df2_index']
    expected[cols] = expected[cols].astype('Int64')

    assert er.network_feature['address'].equals(expected)


def test_two_dfs_exact():

    n_unique = 1000
    n_duplicates = 30

    # generate sample data
    df1 = sample.unique_records(n_unique)
    columns = {
        'phone': {'df': ['HomePhone','WorkPhone','CellPhone'], 'df2':['Phone']},
        'email': {'df': 'Email', 'df2': 'EmailAddress'},
        'address': {'df': 'Address', 'df2':'StreetAddress'}
    }
    df2, sample_id, sample_map = sample.duplicate_df(df1, n_duplicates, columns)

    # compare and derive network
    er = entity_resolver(df1, df2)
    for category, cols in columns.items():
        er.compare(category, columns=cols)
    er.network()

    # assert results
    check_network(er.network_id, er.network_map, columns, sample_id, sample_map, n_duplicates)    

