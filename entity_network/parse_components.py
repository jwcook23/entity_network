import re
from itertools import chain

import pandas as pd
import phonenumbers
import usaddress

def _to_frame(values):

    values = pd.DataFrame(values.tolist(), columns=['components','parsed'], index=values.index)

    values['parsed'] = values['parsed'].astype('string')

    return values

def common(values, delimiter):

    # split by characeters or words
    if delimiter=='word':
        components = values.str.split(' ')
        components = components.apply(lambda x: set(x))
    else:
        components = values.apply(lambda x: set(x))

    components.name = 'components'
    components = pd.DataFrame(components)
    components['parsed'] = values

    return components

def phone(values):

    # wrapper to allow for handling errors
    def parse(value):
        if len(value)==0:
            return {None: None}, None
        try:
            components = phonenumbers.parse(value, 'US')
            components = {
                'Country': components.country_code,
                'Number': components.national_number,
                'Extension': components.extension
            }
            parsed = str(components['Country'])+' '+str(components['Number'])
            if components['Extension'] is not None:
                # TODO: remove 'ext' string so phonenumbers is able to reparse
                # TODO: should these values be reparsed during network summary term difference
                parsed += ' ext '+str(components['Extension'])
            components = set(components.items())
        except phonenumbers.phonenumberutil.NumberParseException:
            components = set(['Error'])
            parsed = re.sub(r'[^0-9\s]+', '', value)
        return components, parsed
    
    # apply wrapper around open source library
    values = values.apply(parse)
    values = _to_frame(values)

    return values

def address(values):

    # wrapper to allow for handling errors
    def parse(value):
        if len(value)==0:
            return {None: None}, None
        try:
            components,_ = usaddress.tag(value)
            parsed = ' '.join([str(val) for val in components.values() if val is not None])
            components = set(components.items())
        except:
            components = set(['Error'])
            parsed = value
        return components, parsed

    # apply wrapper around open source library
    values = values.apply(parse)
    values = _to_frame(values)
    
    return values