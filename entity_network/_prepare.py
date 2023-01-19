from itertools import chain

from entity_network import _exceptions

def flatten(df, columns, category):

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

        # create a copy in case columns where originally set from one variable
        cols = cols.copy()

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
        values[frame].index.names = ('node', 'column')
        values[frame].name = category

    # return a flat list of compared values
    compared = list(chain.from_iterable(compared.values()))
    compared = [col for col in compared if col is not None]

    return values, compared


def clean(dfs, category, text_cleaner):

    for frame, values in dfs.items():
        if values is not None:

            # preprocess values by category type
            values = text_cleaner(values)

            # for merging dataframes, preserve series name regardless of text_cleaner function
            values.name = category

            dfs[frame] = values

    return dfs
