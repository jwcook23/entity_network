'''Generate fake records using Faker https://faker.readthedocs.io/en/master/'''
import random
from itertools import chain
import pandas as pd
from faker import Faker


fake = Faker(locale='en_US')


def unique_records(n_unique):

    sample_df = pd.DataFrame({
        'PersonName': [fake.unique.name() for _ in range(n_unique)],
        'Email': [fake.unique.free_email() for _ in range(n_unique)],
        'HomePhone': [fake.unique.phone_number() for _ in range(n_unique)],
        'WorkPhone': [fake.unique.phone_number() for _ in range(n_unique)],
        'CellPhone': [fake.unique.phone_number() for _ in range(n_unique)],
        'Address': [fake.unique.address() for _ in range(n_unique)]
    })
    for col in sample_df.columns:
        if sample_df[col].duplicated().any():
            raise RuntimeError(f'Column {col} contains duplicate values.')

    return sample_df


def duplicate_records(df1, n_duplicates, columns):

    # generate sample indicies
    index_df1 = random.sample(df1.index.tolist(), k=n_duplicates)
    index_df2 = range(df1.index.max()+1, df1.index.max()+1+n_duplicates)

    # generate new df
    cols = list(chain(*columns.values()))
    df2 = df1.loc[index_df1, cols].copy()
    df2.index = index_df2
    df2['PersonName'] = [fake.unique.name() for _ in range(n_duplicates)]

    # assertion for matching index
    sample_id = pd.DataFrame({'df_index': list(zip(index_df1, index_df2))})
    sample_id['sample_id'] = range(0, len(sample_id))
    sample_id = sample_id.explode('df_index')

    # assertion for matching feature columns
    sample_feature = pd.DataFrame({'df_index': list(zip(index_df1, index_df2))})
    sample_feature['sample_id'] = range(0, len(sample_feature))
    sample_feature = sample_feature.explode('df_index')
    for category, values in columns.items():
        sample_feature[f'{category}_feature'] = [values]*len(sample_feature)
    for category in columns.keys():
        sample_feature = sample_feature.explode(f'{category}_feature')

    # generate the sample frame
    sample_df = pd.concat([df1, df2])

    return sample_df, sample_id, sample_feature


def address_components(n_unique):

    sample_df = pd.DataFrame({
        'Street': [fake.unique.street_address() for _ in range(n_unique)],
        'City': [fake.unique.city() for _ in range(n_unique)],
        'State': [fake.unique.state() for _ in range(n_unique)],
        'Zip': [fake.unique.postcode() for _ in range(n_unique)],
    })
    for col in sample_df.columns:
        if sample_df[col].duplicated().any():
            raise RuntimeError(f'Column {col} contains duplicate values.')

    return sample_df


def similar_df(df1, n_duplicates):
    '''Generate records for two dataframes where df2 shares values with df1 but is rearranged. df2 contains
    two rows for each duplicated record from df1.'''

    columns_compare = {
        'phone': ['HomePhone','WorkPhone','CellPhone', 'Phone'],
        'email': ['Email','EmailAddress'],
        'address': ['Address','StreetAddress']
    }
    columns_match = {
        'df_index': {
            'columns': ['HomePhone', 'Email', 'Address'], 
            'size': 1
        },
        'df2_index': {
            'columns': ['Phone', 'EmailAddress', 'StreetAddress'],
            'size': 2
        }
    }

    # df2 using values from df1
    sample_index = random.sample(df1.index.tolist(), k=n_duplicates)
    df2 = df1.loc[sample_index].copy()
    df2['sample_id'] = range(0, len(df2))

    # produce duplicate rows in df2
    df2 = pd.concat([df2, df2])

    # assertion for matching index
    new_index = range(0, len(df2))
    sample_id = pd.concat([
        pd.DataFrame({'df_index': sample_index, 'sample_id': range(0, len(sample_index))}),
        pd.DataFrame({'df2_index': new_index, 'sample_id': df2['sample_id']})
    ], ignore_index=True)
    sample_id[['df_index','df2_index']] = sample_id[['df_index','df2_index']].astype('Int64')
    df2.index = new_index
    df2 = df2.drop(columns='sample_id')

    # change name of the person to make a new entity
    df2['PersonName'] = [fake.unique.name() for _ in range(len(df2))]

    # rename columns to rearrange
    df2 = df2.rename(columns={'Email': 'EmailAddress', 'Address': 'StreetAddress'})

    # develop single Phone column using values from HomePhone
    df2['Phone'] = df2['HomePhone']
    df2 = df2.drop(columns=['HomePhone','WorkPhone','CellPhone'])

    # assertion for matching feature columns
    sample_feature = sample_id.copy()
    sample_feature['column'] = None
    feature_df = sample_feature.index[sample_feature['df_index'].notna()]
    sample_feature.loc[feature_df, 'column'] = pd.Series([['Email','Address','HomePhone']]*len(feature_df), index=feature_df)
    feature_df2 = sample_feature.index[sample_feature['df2_index'].notna()]
    sample_feature.loc[feature_df2, 'column'] = pd.Series([['EmailAddress','StreetAddress','Phone']]*len(feature_df2), index=feature_df2)
    sample_feature = sample_feature.explode('column')

    return df2, sample_id, sample_feature, columns_compare, columns_match