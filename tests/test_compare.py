
import os
import pandas as pd

import sample

from entity_network.entity_resolver import entity_resolver

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
    assert df1.index.isin(er.network_summary['df_index']).all()
    assert df2.index.isin(er.network_summary['df2_index']).all()
    assert (er.network_summary['df_index']==er.network_summary['df2_index']).all()
    assert (er.network_summary['df_feature']=='address').all()
    assert (er.network_summary['df2_feature']=='address').all()


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
