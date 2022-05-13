import sample

from entity_network.entity_resolver import entity_resolver

def test_single_category():

    n_unique = 1000
    n_duplicates = 30

    # generate sample data
    sample_df = sample.unique_records(n_unique)
    sample_df, sample_id = sample.duplicate_records(sample_df, n_duplicates)

    # compare and derive network
    er = entity_resolver(sample_df)
    columns = ['HomePhone','WorkPhone','CellPhone']
    er.compare('phone', columns=columns)
    network_id, network_feature, network_relation = er.network()

    # assert duplicate pairs exist
    result = network_id.merge(sample_id, on='index')
    result = result.groupby(['network_id','sample_id']).size()
    assert len(result)==n_duplicates
    assert (result==2).all()

    # assert how duplicate pairs are related
    assert (sample_id['index'].sort_values()==network_feature.index).all()
    assert (network_feature['category']=='phone').all()

    # assert details on how duplicate pairs are related
    result = network_relation['phone'].merge(sample_id, on='index')
    result = result.groupby(['phone_id','sample_id'])
    result = result.agg({'column': list, 'index': 'count'})
    assert len(result)==n_duplicates*3
    assert (result['index']==2).all()
    allowed = [[x,x] for x in columns]
    assert result['column'].apply(lambda x: x in allowed).all()


def test_all_category():

    n_unique = 1000
    n_duplicates = 30

    # generate sample data
    sample_df = sample.unique_records(n_unique)
    sample_df, sample_id = sample.duplicate_records(sample_df, n_duplicates)

    # compare and derive network
    er = entity_resolver(sample_df)
    criteria = {
        'phone': ['HomePhone','WorkPhone','CellPhone'],
        'email': ['Email'],
        'address': ['Address']
    }
    for category, columns in criteria.items():
        er.compare(category, columns=columns)
    network_id, network_feature, network_relation = er.network()

    # assert duplicate pairs exist
    result = network_id.merge(sample_id, on='index')
    result = result.groupby(['network_id','sample_id']).size()
    assert len(result)==n_duplicates
    assert (result==2).all()

    for category, columns in criteria.items():

        # assert how duplicate pairs are related
        assert (sample_id['index'].sort_values()==network_feature[network_feature['category']==category].index).all()

        # assert details on how duplicate pairs are related
        result = network_relation[category].merge(sample_id, on='index')
        result = result.groupby([f'{category}_id','sample_id'])
        result = result.agg({'column': list, 'index': 'count'})
        assert len(result)==n_duplicates*len(columns)
        assert (result['index']==2).all()
        allowed = [[x,x] for x in columns]
        assert result['column'].apply(lambda x: x in allowed).all()
