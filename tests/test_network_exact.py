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

    # check expected network pairs exist
    result = network_id.merge(sample_id, on='index')
    result = result.groupby(['network_id','sample_id']).size()
    assert len(result)==n_duplicates
    assert (result==2).all()

    # network pairs all match on phone
    assert (sample_id['index'].sort_values()==network_feature.index).all()
    assert (network_feature['category']=='phone').all()

    # network pairs all match directly on phone the same phone column
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

    # check expected network pairs exist
    result = network_id.merge(sample_id, on='index')
    result = result.groupby(['network_id','sample_id']).size()
    assert len(result)==n_duplicates
    assert (result==2).all()

    for category, columns in criteria.items():

        # network pairs match on each category
        assert (sample_id['index'].sort_values()==network_feature[network_feature['category']==category].index).all()

        # network paris match directy on each category
        result = network_relation[category].merge(sample_id, on='index')
        result = result.groupby([f'{category}_id','sample_id'])
        result = result.agg({'column': list, 'index': 'count'})
        assert len(result)==n_duplicates*len(columns)
        assert (result['index']==2).all()
        allowed = [[x,x] for x in columns]
        assert result['column'].apply(lambda x: x in allowed).all()


def test_two_dfs():

    n_unique = 1000
    n_duplicates = 30

    # generate sample data
    df1 = sample.unique_records(n_unique)
    df2, sample_id = sample.similar_df(df1, n_duplicates)

    # compare and derive network
    er = entity_resolver(df1, df2)
    criteria = {
        'phone': ['HomePhone','WorkPhone','CellPhone', 'Phone'],
        'email': ['Email','EmailAddress'],
        'address': ['Address','StreetAddress']
    }
    for category, columns in criteria.items():
        er.compare(category, columns=columns)
    network_id, network_feature, network_relation = er.network()

    # check expected network pairs exist
    result_df = sample_id['df'].merge(network_id['df'], on='index')
    result_df2 = sample_id['df2'].merge(network_id['df2'], on='index')
    result = result_df.merge(result_df2, on='sample_id', suffixes=('_df','_df2'))
    assert (result['network_id_df']==result['network_id_df2']).all()
    assert len(result)==n_duplicates

    assert 1==1