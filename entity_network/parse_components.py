import phonenumbers
import usaddress

def phone(value):

    try:
        parsed = phonenumbers.parse(value, 'US')
        parsed = str(parsed.national_number)
    except phonenumbers.phonenumberutil.NumberParseException:
        parsed, components = None
    
    return parsed, components

def address(values):

    # parse address components into a flattened list
    parsed = []
    for val in values:
        try:
            components,_ = usaddress.tag(val)
            components = set(components.items())
        except usaddress.RepeatedLabelError:
            components = {'Error': None}
        parsed += components

    return parsed