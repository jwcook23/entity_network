import pandas as pd

from entity_network import clean_text, _exceptions

default_text_cleaner = {
    'name': clean_text.name,
    'phone': clean_text.phone,
    'email': clean_text.email,
    'email_domain': clean_text.email_domain,
    'address': clean_text.address
}

def flatten(df, columns):

    # prepare column argument
    if isinstance(columns, str):
        columns = [columns]
    split_cols = [idx for idx,cols in enumerate(columns) if isinstance(cols, list)]
    if len(split_cols)>0:
        for idx in split_cols:
            # join dataframe values
            combined = ','.join(columns[idx])
            df[combined] = df[columns[idx]].fillna('').agg(' '.join, axis=1)
            # adjust columns argument
            columns[idx] = combined

    # handle columns from two dataframes with the same column name
    columns = list(set(columns))

    # check presence of columns
    columns = pd.Series(columns)
    missing = columns[~columns.isin(df)]
    if any(missing):
        raise _exceptions.MissingColumn(f'Argument columns not in DataFrame: {missing.tolist()}')

    # prepare multiple columns by pivoting into a single column
    values = df[columns].stack()

    return values


def clean(values, category, text_cleaner):

    # check allowed category argument for default cleaner
    if category not in default_text_cleaner:
        raise _exceptions.InvalidCategory(f"Argument catgeory must be one of {list(default_text_cleaner.keys())} when text_cleaner=='default'.")

    # preprocess values by category type
    if text_cleaner=='default':
        values = default_text_cleaner[category](values)
    else:
        values = text_cleaner(values)

    return values
