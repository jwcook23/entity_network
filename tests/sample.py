'''Generate fake records using Faker https://faker.readthedocs.io/en/master/'''

import pandas as pd
from faker import Faker


fake = Faker(locale='en_US')


def unique_records(n_samples):

    sample_df = pd.DataFrame({
        'PersonName': [fake.unique.name() for _ in range(n_samples)],
        'Email': [fake.unique.free_email() for _ in range(n_samples)],
        'HomePhone': [fake.unique.phone_number() for _ in range(n_samples)],
        'WorkPhone': [fake.unique.phone_number() for _ in range(n_samples)],
        'CellPhone': [fake.unique.phone_number() for _ in range(n_samples)],
        'Address': [fake.unique.address() for _ in range(n_samples)]
    })
    for col in sample_df.columns:
        if sample_df[col].duplicated().any():
            raise RuntimeError(f'Column {col} contains duplicate values.')

    return sample_df

def duplicate_records(df1, n_samples):

    # generate new df and keep track of old and new indices for assertions
    df2 = df1.sample(n_samples).copy()
    new_index = df1.index.max()+1
    new_index = range(new_index, new_index+n_samples)
    sample_id = pd.DataFrame({'index': list(zip(df2.index, new_index))})
    sample_id['sample_id'] = range(0, len(sample_id))
    sample_id = sample_id.explode('index')
    df2.index = new_index

    sample_df = pd.concat([df1, df2])

    # TODO: return network for test assertion
    return sample_df, sample_id


def similar_df(df1, n_duplicates):
    '''Generate records for two dataframes where df2 shares values with df1 but is rearranged.'''

    # df2 using values from df1
    df2 = df1.sample(n_duplicates).copy()
    new_index = range(0, len(df2))
    sample_id = pd.DataFrame({'index': list(zip(df2.index, new_index))})
    sample_id['sample_id'] = range(0, len(sample_id))
    sample_id = sample_id.explode('index')
    df2.index = new_index

    # change name of the person to make a new entity
    df2['PersonName'] = [fake.unique.name() for _ in range(n_duplicates)]

    # rename columns to rearrange
    df2 = df2.rename(columns={'Email': 'EmailAddress', 'Address': 'StreetAddress'})

    # develop single Phone column using values from HomePhone
    df2['Phone'] = df2['HomePhone']
    df2 = df2.drop(columns=['HomePhone','WorkPhone','CellPhone'])

    return df2, sample_id