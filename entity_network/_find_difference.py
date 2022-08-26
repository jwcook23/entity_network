from collections import Counter
from itertools import chain

from entity_network.clean_text import settings
from entity_network import parse_components

def main(values, category):

    # apply category specific or general component parser
    if category=='address':
        parsed = parse_components.address(values[category])
    elif category=='phone':
        parsed = parse_components.phone(values[category])
    else:
        delimiter = settings[category]['comparer']
        parsed = parse_components.common(values[category], delimiter=delimiter)

    # alias names for later merges that include multiple categories
    alias = {'parsed': f'{category}_normalized', 'components': f'{category}_difference'}

    # include source column descriptions if provided
    summary = {'parsed': list, 'components': list}
    list_notna = lambda x: list(x.dropna())
    if 'df2_column' in values:
        parsed['df2_column'] = values['df2_column']
        summary = {**{'df2_column': list_notna}, **summary}
        alias = {**{'df2_column': f'df2_column_{category}'}, **alias}
    if 'df_column' in values:
        parsed['df_column'] = values['df_column']
        summary = {**{'df_column': list_notna}, **summary}
        alias = {**{'df_column': f'df_column_{category}'}, **alias}

    # group components by id
    values = parsed.groupby(values.index.name)
    values = values.agg(summary)

    # calculate the difference in components
    values['components'] = values['components'].apply(_term_diff)

    # rename columns to reflect category for later merging and after difference is found
    values = values.rename(columns=alias)

    return values

def _term_diff(values):

    frequency = Counter(chain(*values))
    difference = ['='.join(key) for key,val in frequency.items() if val==1 and key[1] is not None]

    return difference