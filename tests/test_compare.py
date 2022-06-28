
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
    er.compare('address', columns=[['Street', 'City', 'State', 'Zip'], 'Address'])
    
    assert all(er.related['address'].index == [0, 10])
    assert all(er.related['address']['column'] == ['Street,City,State,Zip', 'Address'])
    assert all(er.related['address']['address_id_exact'] == [0,0])
    assert all(er.related['address']['address_id_similar'].isna())
    assert all(er.related['address']['address_id'] == [0,0])

def test_similar_address():

    file_path = os.path.join('tests','similar_address.csv')

    df = pd.read_csv(file_path)
    df = df[df['threshold=0.7']==1]
    df1 = df[['Address0']]
    df2 = df[['Address1']]

    er = entity_resolver(df1, df2)
    er.compare('address', columns=['Address0', 'Address1'], threshold=0.7)

    network_id, _ = er.network()

    actual = network_id.groupby('network_id')
    list_notna = lambda l: [x for x in l if pd.notna(x)]
    actual = actual.agg({
        'df_index': list_notna,
        'df2_index': list_notna
    })


    actual = actual.apply(pd.Series.explode)
    missing_df = df1.index[~df1.index.isin(actual['df_index'])]
    missing_df2 = df2.index[~df2.index.isin(actual['df2_index'])]

    # comparison = er.index_comparison('address', index_df=missing_df, index_df2=missing_df2)
    assert (actual['df_index']==actual['df2_index']).all()
    assert len(missing_df)==0
    assert len(missing_df2)==0


def test_combine_similar_exact():


    df1 = pd.DataFrame({
        'AddressA': [
            '1234 S NameA Street Town, ST 56789',
            '1234 South NameA Street Town, ST 56789-0147'
        ],
    })
    df2 = pd.DataFrame({
        'AddressA': [
            '1234 S NameB Street Town, ST 56789',
            '1234 South NameA Street Town, ST 56789-0147'
        ],
    })


    er = entity_resolver(df1, df2)
    er.compare('address', columns=['AddressA', 'AddressA'], threshold=0.8)

    expected = pd.DataFrame({
        'column': ['AddressA']*5,
        'address_id_exact': [0,0,0,pd.NA, pd.NA],
        'address_id_similar': [pd.NA, pd.NA, pd.NA, 0, 0],
        'address_id': [0]*5,
        'df_index': [0, 1, pd.NA, 0, pd.NA],
        'df2_index': [pd.NA, pd.NA, 1, pd.NA, 0]
    }, index=pd.Index([0,1,3,0,2], name='index'))
    expected['column'] = expected['column'].astype('string')
    cols = ['address_id_exact','address_id_similar','df_index','df2_index']
    expected[cols] = expected[cols].astype('Int64')

    assert er.related['address'].equals(expected)
