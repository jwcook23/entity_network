from collections import Counter
from itertools import chain

from entity_network.clean_text import settings
from entity_network import parse_components

def main(values, category):

    # apply category specific or general component parser
    if category=='address':
        parsed = parse_components.address(values)
    elif category=='phone':
        parsed = parse_components.phone(values)
    else:
        parsed = parse_components.common(values, delimiter=settings[category]['comparer'])

    # group components to compare
    values = parsed.groupby(values.index.name)
    values = values.agg({'parsed': list,'components': list})

    # calculate the difference in components
    values['components'] = values['components'].apply(_term_diff)

    # rename columns to reflect category for later merging and after difference is found
    values = values.rename(columns={'parsed': category, 'components': f'{category}_difference'})

    return values

def _term_diff(values):

    frequency = Counter(chain(*values))
    difference = [key for key,val in frequency.items() if val==1]
    if len(difference)==0:
        difference = None

    return difference