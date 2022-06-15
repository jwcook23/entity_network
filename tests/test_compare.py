
import pandas as pd

import sample

from entity_network.entity_resolver import entity_resolver

def test_split_column():

    n_unique = 10
    n_duplicate = 1

    df1 = sample.address_components(n_unique)

    df2 = pd.DataFrame({
        'Address': [df1.at[0,'Street']+' '+df1.at[0,'City']+' '+df1.at[0,'State']+' '+df1.at[0,'Zip']]
    })

    er = entity_resolver(df1, df2)
    er.compare('address', columns=[['Street', 'City', 'State', 'Zip'], 'Address'])

    pd.DataFrame({
        'column': ['Street,City,State,Zip', 'Address']
    })
    
    assert all(er.relationship['address'].index == [0, 11])
    assert all(er.relationship['address']['column'] == ['Street,City,State,Zip', 'Address'])
    assert all(er.relationship['address']['address_id_exact'] == [0,0])
    assert all(er.relationship['address']['address_id_similar'].isna())
    assert all(er.relationship['address']['address_id'] == [0,0])