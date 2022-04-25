import pandas as pd
import pytest

from entity_network import entity_resolution

@pytest.fixture(scope='module')
def df():

    df = pd.DataFrame(columns=['CompanyName','PersonName','WorkPhone','HomePhone','CellPhone','Email','Address'])
    df.loc[0] = ["Foo Bar's of   City", None, "1234567890", None, None, "name@companyname.org", "123 North Name Road, FL 12345-6789"]
    df.loc[1] = ["Foo   Bar", None, "(1) 123-456-7890", "123-456-7890", "123-456-7890", "name@companyname.org", "123 North Name Road Apartment 3C, FL 12345-6789"]
    df.loc[2] = [None, "Foo Bar", None, "123-456-7890", "123-456-7890", "blahblah@companyname.org", None]
    df.loc[3] = [None, "John Smith", None, None, None, "john.smith@companyname.org", None]
    df.loc[4] = [None, "First Last", None, None, None, "blah_blah@companyname.org", None]
    df.loc[5] = [ "NameA NameB", "NameA NameB", "123-444-4444", None, None, None, None]
    df.loc[6] = [None, "NameC", None, None, "123 444 4444", None, None]
    df.loc[7] = [ "Foo Bar", None, None, None, None, "blahblah@companyname.org", None]

    yield df

def test_argument_exceptions():

    with pytest.raises(entity_resolution.DuplicatedIndex):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,1])
        er = entity_resolution.entity_resolution(df)

    with pytest.raises(entity_resolution.MissingColumn):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolution.entity_resolution(df)
        er.compare_records('phone', 'ColumnB', kneighbors=10, threshold=1)

    with pytest.raises(entity_resolution.InvalidCategory):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolution.entity_resolution(df)
        er.compare_records('foo', 'ColumnB', kneighbors=10, threshold=1)

    with pytest.raises(ValueError):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolution.entity_resolution(df)
        er.compare_records('email', 'ColumnA', kneighbors=10, threshold=100)

    with pytest.raises(ValueError):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolution.entity_resolution(df)
        er.compare_records('email', 'ColumnA', kneighbors=-1.2, threshold=0.8)


def test_network_simpliest(df):

    er = entity_resolution.entity_resolution(df)
    er.compare_records('email', columns=['Email'], kneighbors=10, threshold=1)
    er.compare_records('phone', columns=['HomePhone','WorkPhone','CellPhone'], kneighbors=10, threshold=1)
    er.resolve_entity(columns=['CompanyName','PersonName'], kneighbors=10, threshold=1)
    er.resolve_network()


def test_network_exact(df):

    er = entity_resolution.entity_resolution(df)
    er.compare_records('phone', columns=['HomePhone','WorkPhone','CellPhone'], kneighbors=10, threshold=1)
    # er.compare_records('email', columns=['Email'], kneighbors=10, threshold=1)
    # er.compare_records('address', columns=['Address'], threshold=0.8)

def test_network_similar(df):
    
    er = entity_resolution.entity_resolution(df)
    # er.compare_records('phone', columns=['HomePhone','WorkPhone','CellPhone'], kneighbors=10, threshold=0.9)
    er.compare_records('email', columns=['Email'], kneighbors=10, threshold=0.95)
    # er.compare_records('address', columns=['Address'], threshold=0.8)
    assert isinstance(df, pd.DataFrame)


def test_network_kneighbors_threshold():

    df = pd.DataFrame({
        'Phone': [
            '0 1234567890',
            '1 1234567890',
            '2 4569999999',
            '3 4569999999',
            '4 4569999999'
        ]
    })
    er = entity_resolution.entity_resolution(df)
    er.compare_records('phone', columns='Phone', kneighbors=3, threshold=0.9)


# read data
# df = pd.read_csv(
#     'C:/Users/jcook/Library/Temp/SCTASK0118191_Fire Loss Study for Michael Carver/data/claim_center/RelatedContacts.csv',
#     usecols=[
#         'CompanyName','PersonFirstName','PersonLastName',
#         'HomePhone','WorkPhone','CellPhone',
#         'EmailAddress1',
#         'AddressLine1','AddressLine2','AddressLine3','AddressCity','AddressState','AddressZip'
#     ],
#     dtype='string'
# )
# # derive columns
# df['PersonName'] = df['PersonFirstName']+' '+df['PersonLastName']
# columns = ['AddressLine1','AddressLine2','AddressLine3','AddressCity','AddressState','AddressZip']
# df['Address'] = df[columns].apply(lambda x: x.str.cat(sep=' '), axis=1)

# # resolve entities
# er = entity_resolution(df)
# er.compare_records('phone', columns=['Phone','WorkPhone','CellPhone'], kneighbors=10, threshold=0.9)