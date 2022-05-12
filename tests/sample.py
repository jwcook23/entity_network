'''Generate fake records using Faker https://faker.readthedocs.io/en/master/'''

import pandas as pd
from faker import Faker


fake = Faker(locale='en_US')


def unique_records(n_samples):

    df = pd.DataFrame({
        'PersonName': [fake.unique.name() for _ in range(n_samples)],
        'Email': [fake.unique.free_email() for _ in range(n_samples)],
        'HomePhone': [fake.unique.phone_number() for _ in range(n_samples)],
        'WorkPhone': [fake.unique.phone_number() for _ in range(n_samples)],
        'CellPhone': [fake.unique.phone_number() for _ in range(n_samples)],
        'Address': [fake.unique.address() for _ in range(n_samples)]
    })

    return df

def duplicate_records(df1, n_samples):

    df2 = df1.sample(n_samples).copy()
    seed = df1.index.max()+1
    df2.index = range(seed, seed+n_samples)

    df = pd.concat([df1, df2])

    # TODO: return network for test assertion
    return df


def similar_df(df1):
    '''Generate records for two dataframes where df2 shares values with df1.'''

    # df2 using values from df1 but rearranging and renaming columns
    n_samples = 30
    df2 = df1.sample(n_samples).copy()
    df2 = df2.reset_index(drop=True)

    # change name of the person to make a new entity
    df2['PersonName'] = [fake.unique.name() for _ in range(n_samples)]

    # rename columns for parameter specification when comparing
    df2 = df2.rename(columns={'Email': 'EmailAddress', 'Address': 'StreetAddress'})

    # develop single Phone column using values from HomePhone, WorkPhone, CellPhone
    df2['Phone'] = pd.NA
    df2.iloc[0:10, df2.columns=='Phone'] = df2.iloc[0:10, df2.columns=='HomePhone']
    df2.iloc[10:20, df2.columns=='Phone'] = df2.iloc[10:20, df2.columns=='WorkPhone']
    df2.iloc[20:30, df2.columns=='Phone'] = df2.iloc[20:30, df2.columns=='CellPhone']
    df2 = df2.drop(columns=['HomePhone','WorkPhone','CellPhone'])

    # TODO: return network for test assertion
    return df1, df2