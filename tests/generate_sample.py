from multiprocessing.reduction import duplicate
import pandas as pd
from faker import Faker


fake = Faker(locale='en_US')


# fake.profile(fields=['name','company','mail','username','address','building_number','asdfds'])

def unique_records(n_samples=1000):
    '''Generate unique records by using the .unique attribue.'''

    df = pd.DataFrame({
        'PersonName': [fake.unique.name() for _ in range(n_samples)],
        'CompanyName': [fake.unique.company() for _ in range(n_samples)],
        'Email': [fake.unique.free_email() for _ in range(n_samples)],
        'HomePhone': [fake.unique.phone_number() for _ in range(n_samples)],
        'WorkPhone': [fake.unique.phone_number() for _ in range(n_samples)],
        'CellPhone': [fake.unique.phone_number() for _ in range(n_samples)],
        'Address': [fake.unique.address() for _ in range(n_samples)]
    })
    df['_EntityID'] = range(0,len(df))

    return df

def duplicate_entities(df, n_samples=100):
    '''Produce duplicate entities using the first n_samples.
    
    A duplicate entity has a similar name and another feature that represent potential typos or format differences.'''
    
    entity = df.head(n_samples).copy()

    # add 2 random characters to name
    entity['PersonName'] = entity['PersonName']+pd.Series([fake.pystr(max_chars=2) for _ in range(n_samples)], index=entity.index)

    # add 5 random characeters to address
    entity['Address'] = entity['Address']+pd.Series([fake.pystr(max_chars=5) for _ in range(n_samples)], index=entity.index)

    # replace home phone with cell phone and add a random digit
    entity['HomePhone'] = entity['CellPhone']+pd.Series([fake.pystr_format('#') for _ in range(n_samples)], index=entity.index)

    # remove exact match entity columns
    entity = entity.drop(columns=['CellPhone','WorkPhone','Email','CompanyName'])

    # incorporate duplicated entities into original dataframe
    df = pd.concat([df, entity], axis='index', ignore_index=True)

    return df, entity


def create_network(df, n_samples=10):
    '''Create networks of records using different features of the first n_samples.
    
    A network isn't similar on name but matches exactly on another feature after cleaning.'''
    
    # define the network id using the first n_samples
    source = df.head(n_samples).copy()
    source['_NetworkID'] = range(0, len(source))
    df = df.merge(source[['_NetworkID']], left_index=True, right_index=True, how='left')
    df['_NetworkID'] = df['_NetworkID'].astype('Int64')
    
    # intialize a network dataframe to hold features the network matches on
    network = pd.DataFrame()

    # network links on email
    email = source[['_NetworkID','Email']].copy()
    email['PersonName'] = [fake.unique.name() for _ in range(n_samples)]
    network = pd.concat([network, email], axis='index')

    # network links between home phone and work phone
    phone = source[['_NetworkID','HomePhone']].copy()
    phone = phone.rename(columns={'HomePhone': 'WorkPhone'})
    phone['PersonName'] = [fake.unique.name() for _ in range(n_samples)]
    network = pd.concat([network, phone], axis='index')

    # network links on address
    address = source[['_NetworkID','Address']].copy()
    address['PersonName'] = [fake.unique.name() for _ in range(n_samples)]
    network = pd.concat([network, address], axis='index')

    # incorporate network into original dataframe
    df = pd.concat([df, entity], axis='index', ignore_index=True)

    return df, network

df = unique_records()
df, entity = duplicate_entities(df)
df, network = create_network(df)

from entity_network import entity_resolver

# compare values
er = entity_resolver.entity_resolver(df)
er.compare('email', columns=['Email'], kneighbors=10, threshold=0.8)
er.compare('phone', columns=['HomePhone','WorkPhone','CellPhone'], kneighbors=10, threshold=1)
er.compare('address', columns='Address', kneighbors=10, threshold=0.7)

df, entity_id, entity_feature = er.entity(columns=['CompanyName','PersonName'], kneighbors=10, threshold=0.6, analyzer='word')