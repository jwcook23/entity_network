from itertools import chain

from entity_network import clean_text, _exceptions

default_text_cleaner = {
    'name': clean_text.name,
    'phone': clean_text.phone,
    'email': clean_text.email,
    'email_domain': clean_text.email_domain,
    'address': clean_text.address
}

def flatten(df, columns):

    values = {'df': None, 'df2': None}
    compared = {'df': [None], 'df2': [None]}

    if df['df2'] is None:
        if not isinstance(columns, dict):
            columns = {'df': columns}
    else:
        if not isinstance(columns, dict):
            raise RuntimeError('Columns parameter must be a dict if two dataframes are provided.')
        if len({'df','df2'}-set(columns.keys()))>0:
            raise RuntimeError('Columns parameter must contain keys for df and df2 if two dataframes are provided.')

    for frame, cols in columns.items():

        # skip processing df2 if not provided
        if df[frame] is None:
            continue

        # change string to a list for potentially stacking of multiple columns
        if isinstance(cols, str):
            cols = [cols]

        # combine multiple columns into single if nested list
        split_cols = [idx for idx,nested in enumerate(cols) if isinstance(nested, list)]
        if len(split_cols)>0:
            for idx in split_cols:
                # form a single column name
                combined = ','.join(cols[idx])
                # join non-na columns using a space
                df[frame][combined] = df[frame][cols[idx]].fillna('').agg(' '.join, axis=1)
                # adjust columns argument
                cols[idx] = combined

        # # check presence of columns
        missing = [x for x in cols if x not in df[frame]]
        if len(missing)>0:
            raise _exceptions.MissingColumn(f'Argument columns not in DataFrame: {missing}')

        # track columns that were compared in a standard format
        compared[frame] = cols

        # prepare multiple columns by pivoting into a single column
        values[frame] = df[frame][cols].stack()

    # return a flat list of compared values
    compared = list(chain.from_iterable(compared.values()))
    compared = [col for col in compared if col is not None]

    return values, compared


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
