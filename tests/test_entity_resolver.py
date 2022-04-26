import pandas as pd
import pytest

from entity_network import entity_resolver, _exceptions

@pytest.fixture(scope='module')
def df():

    df = pd.DataFrame(columns=['CompanyName','PersonName','WorkPhone','HomePhone','CellPhone','Email','Address'])
    df.loc[0] = ["Foo Bar's of   City", None, "1234567890", None, None, "name@companyname.org", "123 North RoadName Road, FL 12345-6789"]
    df.loc[1] = ["Foo   Bar", None, "(1) 123-456-7890", "123-456-7890", "123-456-7890", "name@companyname.org", "123 North RoadName Road Apartment 3C, FL 12345-6789"]
    df.loc[2] = [None, "Foo Bar", None, "123-456-7890", "123-456-7890", "blahblah@companyname.org", None]
    df.loc[3] = [None, "John Smith", None, None, None, "john.smith@companyname.org", None]
    df.loc[4] = [None, "First Last", None, None, None, "blah_blah@companyname.org", None]
    df.loc[5] = ["NameA NameB", "NameA NameB", "123-444-4444", None, None, None, None]
    df.loc[6] = [None, "NameC", None, "(1) 123-456-7890 ext 11", "123 444 4444", None, None]
    df.loc[7] = ["Foo Bar", None, None, None, None, "blahblah@companyname.org", None]

    yield df

def test_argument_exceptions():

    with pytest.raises(_exceptions.ReservedColumn):
        df = pd.DataFrame({'ColumnA': ['a','b'], 'entity_id': [0, 1], 'network_id': [0, 1]})
        er = entity_resolver.entity_resolver(df)

    with pytest.raises(_exceptions.DuplicatedIndex):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,1])
        er = entity_resolver.entity_resolver(df)

    with pytest.raises(_exceptions.MissingColumn):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver.entity_resolver(df)
        er.compare('phone', 'ColumnB', kneighbors=10, threshold=1)

    with pytest.raises(_exceptions.InvalidCategory):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver.entity_resolver(df)
        er.compare('foo', 'ColumnB', kneighbors=10, threshold=1)

    with pytest.raises(_exceptions.ThresholdRange):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver.entity_resolver(df)
        er.compare('email', 'ColumnA', kneighbors=10, threshold=100)

    with pytest.raises(_exceptions.KneighborsRange):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver.entity_resolver(df)
        er.compare('email', 'ColumnA', kneighbors=-1.2, threshold=0.8)


def test_network_simple(df):

    df_input = df.copy()

    # compare values
    er = entity_resolver.entity_resolver(df)
    er.compare('email', columns=['Email'], kneighbors=10, threshold=1)
    er.compare('phone', columns=['HomePhone','WorkPhone','CellPhone'], kneighbors=10, threshold=1)

    # compute entity
    df, entity_id, entity_feature = er.entity(columns=['CompanyName','PersonName'], kneighbors=10, threshold=1)
    assert df['entity_id'].equals(pd.Series([1,0,0,2,None,4,5,0], dtype='Int64'))
    assert entity_id['entity_id'].equals(pd.Series([1,0,0,2,4,5,0], index=[0,1,2,3,5,6,7]))
    assert entity_id['entity_name'].equals(
        pd.Series(['foo bar s city','foo bar','foo bar','john smith','namea nameb', 'namec','foo bar'], index=[0,1,2,3,5,6,7], dtype='string')
    )
    assert entity_feature['category'].equals(pd.Series(['phone','email','phone','email'], index=[1,2,2,7]))
    assert entity_feature['entity_id'].equals(pd.Series([0,0,0,0], index=[1,2,2,7]))

    # compute network
    df, network_id, network_feature = er.network()
    assert df['network_id'].equals(pd.Series([0,0,0,2,0,1,1,0], dtype='Int64'))
    assert network_id['network_id'].equals(pd.Series([0,0,0,2,0,1,1,0], index=[0,1,2,3,4,5,6,7]))
    assert network_id['network_name'].equals(
        pd.Series(['foo bar','foo bar','foo bar','john smith','foo bar','namea nameb', 'namea nameb', 'foo bar'], index=[0,1,2,3,4,5,6,7], dtype='string')
    )
    assert network_feature['category'].equals(pd.Series(['email','phone','email','phone','email','phone','email','phone','phone','email'], index=[0,0,1,1,2,2,4,5,6,7]))
    assert network_feature['network_id'].equals(pd.Series([0,0,0,0,0,0,0,1,1,0], index=[0,0,1,1,2,2,4,5,6,7]))

    # insure an original dataframe is returned
    assert df[df_input.columns].equals(df_input)


def test_network_similar(df):
    
    df_input = df.copy()

    # compare values
    er = entity_resolver.entity_resolver(df)
    er.compare('email', columns=['Email'], kneighbors=10, threshold=0.8)
    er.compare('phone', columns=['HomePhone','WorkPhone','CellPhone'], kneighbors=10, threshold=1)
    er.compare('address', columns='Address', kneighbors=10, threshold=0.7)

    # compute entity
    df, entity_id, entity_feature = er.entity(columns=['CompanyName','PersonName'], kneighbors=10, threshold=0.6, analyzer='word')
    assert df.at[0,'entity_id']==0
    assert entity_id.at[0,'entity_id']==0
    assert entity_feature.loc[0].equals(pd.DataFrame({'category': ['email','phone','address'], 'entity_id': [0,0,0]}, index=[0,0,0]))
    # no match on similarity since non-exact matches become exact after preprocessing
    assert len(er.similar['email'])==0
    # similar name match
    assert er.similar['name'][['index','index_similar']].equals(pd.DataFrame({'index': [0,1], 'index_similar': [1,0]}))
    # similar address match
    assert er.similar['address'][['index','index_similar']].equals(pd.DataFrame({'index': [0,1], 'index_similar': [1,0]}))

    # insure an original dataframe is returned
    assert df[df_input.columns].equals(df_input)