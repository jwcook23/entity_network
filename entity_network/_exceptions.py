class DuplicatedIndex(Exception):
    '''Exception for dataframe not having a unique index.'''
    pass

class MissingColumn(Exception):
    '''Exception for column not present in input dataframe.'''
    pass

class InvalidCategory(Exception):
    '''Exception for invalid category supplied.'''
    pass

class KneighborsThreshold(Exception):
    '''Exception for combination of kneighbors and threshold excluding similar matches.'''
    pass