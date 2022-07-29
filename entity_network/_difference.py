import usaddress

def address(values):

    values = values.dropna()

    # parse address components
    try:
        components0,_ = usaddress.tag(values[0])
        components1,_ = usaddress.tag(values[1])
    except usaddress.RepeatedLabelError:
        return None, None

    # determine difference in address components
    components0 = set(components0.items())
    components1 = set(components1.items())
    diff0 = components0-components1
    if len(diff0)==0:
        diff0 = None
    diff1 = components1-components0
    if len(diff1)==0:
        diff1 = None

    return diff0, diff1

term_difference = {
    'address': address
}