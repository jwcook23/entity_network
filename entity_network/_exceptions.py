class ReservedColumn(Exception):
    '''Exception for dataframe containing a reserved column name.'''
    pass

class DuplicatedIndex(Exception):
    '''Exception for dataframe not having a unique index.'''
    pass

class MissingColumn(Exception):
    '''Exception for column not present in input dataframe.'''
    pass

class InvalidCategory(Exception):
    '''Exception for invalid category supplied.'''
    pass

class ThresholdRange(Exception):
    '''Exception for threshold out of allowed range.'''
    pass

class KneighborsRange(Exception):
    '''Exception for kneighbors out of allowed range.'''
    pass

class KneighborsThreshold(Exception):
    '''Exception for combination of kneighbors and threshold excluding similar matches.'''
    pass