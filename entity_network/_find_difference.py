from collections import Counter
from itertools import chain

from entity_network.clean_text import comparison_rules
from entity_network import parse_components

def main(values, category):

    # apply category specific or general component parser
    if category=='address':
        parsed = parse_components.address(values[category])
    elif category=='phone':
        parsed = parse_components.phone(values[category])
    else:
        delimiter = comparison_rules[category]['comparer']
        parsed = parse_components.common(values[category], delimiter=delimiter)

    # alias names for later merges that include multiple categories
    alias = {'parsed': f'{category}_normalized', 'components': f'{category}_difference'}

    # include source column descriptions if provided
    plan = {'parsed': list, 'components': list}
    list_unique_notna = lambda x: list(x.drop_duplicates().dropna())
    if 'df2_column' in values:
        parsed['df2_column'] = values['df2_column']
        plan = {**{'df2_column': list_unique_notna}, **plan}
        alias = {**{'df2_column': f'df2_column_{category}'}, **alias}
    if 'df_column' in values:
        parsed['df_column'] = values['df_column']
        plan = {**{'df_column': list_unique_notna}, **plan}
        alias = {**{'df_column': f'df_column_{category}'}, **alias}

    # group components by id
    summary = parsed.groupby(values.index.name)
    summary = summary.agg(plan)

    # calculate the difference in components
    summary['components'] = summary['components'].apply(_term_diff)

    # rename columns to reflect category for later merging and after difference is found
    summary = summary.rename(columns=alias)

    return summary

def _term_diff(values):

    if values[0] is None or values[1] is None:
        return None

    frequency = Counter(chain(*values))
    difference = ['='.join(key) for key,val in frequency.items() if val==1 and key[1] is not None]

    return difference