from itertools import chain
from collections import Counter

from entity_network.clean_text import settings
from entity_network import parse_components

def main(values, category):

    # if multiple columns from a df, combine into a list
    if not isinstance(values, list):
        values = list(chain(*values.values))

    # apply category specific or general difference finder
    if category=='address':
        difference = address(values)
    else:
        difference = common(values, category)

    if len(difference)==0:
        difference = None

    return difference

def _term_diff(values):

    frequency = Counter(values)
    difference = [key for key,val in frequency.items() if val==1]

    return difference

def common(values, category):

    # split by characeters or words
    if settings[category]['comparer']=='word':
        values = [val.split(' ') for val in values]
    else:
        values = [list(val) for val in values]

    # form a flatten list
    values = list(chain(*values))

    # find terms only appearing once
    difference = _term_diff(values)

    return difference

def address(values):

    parsed = parse_components.address(values)

    # find terms only appearing once
    difference = _term_diff(parsed)    

    return difference