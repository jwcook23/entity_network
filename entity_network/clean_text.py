'''Text cleaning functions for different categories of data.'''
import pandas as pd
import flashtext
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

from entity_network import parse_components

# TODO: investigate cleantext https://pypi.org/project/clean-text/


def _alphanumeric_only(prepared):

    return prepared.str.replace(r'[\W_]+', ' ', regex=True)


def _common_presteps(prepared):

    # insure pandas.String is returned
    prepared = prepared.astype('string')

    # lowercase
    prepared = prepared.str.lower()

    return prepared


def _common_poststeps(prepared):

    # remove extra whitespace
    prepared = prepared.str.replace(r'\s{2,}', ' ', regex=True)
    prepared = prepared.str.strip()

    # set values that only contain text as empty
    prepared[prepared.str.len()==0] = pd.NA

    return prepared


def _remove_stopwords(prepared, stopwords, category): 

    if stopwords is None:
        return prepared
    elif stopwords=='default':
        stopwords = comparison_rules[category]['stopwords']
    
    pattern = r'\b(?:{})\b'.format('|'.join(stopwords))
    prepared = prepared.str.replace(pattern, '', regex=True)

    # prepared[prepared==''] = pd.NA

    return prepared


def possible_stopwords(values) -> pd.Series: 
    '''Count words in decending order.
    
    Parameters
    ----------
    values (pd.Series) : sentenances of words

    Returns
    -------
    possible_stopwords (pd.Series) : word and count of word apperance
    '''

    possible_stopwords = values.str.split(r'\s+')
    possible_stopwords = possible_stopwords.explode()
    possible_stopwords = possible_stopwords.value_counts()

    return possible_stopwords


def alphanumeric(values: pd.Series) -> pd.Series:
    '''Prepocess generic string containing characters and numbers only.

    Parameters
    ----------
    values (pd.Series) : values before processing

    Returns
    -------
    prepared (pd.Series) : values after processing
    '''  

    prepared = _common_presteps(values)
    prepared = _alphanumeric_only(prepared)

    return _common_poststeps(prepared)
    

def name(values: pd.Series, stopwords='default') -> pd.Series:
    '''Prepocess comapany and person names.

    Parameters
    ----------
    values (pd.Series) : values before processing

    Returns
    -------
    prepared (pd.Series) : values after processing
    '''  

    prepared = _common_presteps(values)
    prepared = _alphanumeric_only(prepared)

    # manually remove stopwords, as TfidfVectorizer stopwords only applys if analyzer='word'
    prepared = _remove_stopwords(prepared, stopwords, 'name')

    return _common_poststeps(prepared)


def phone(values: pd.Series, stopwords='default') -> pd.Series:
    '''Prepocess phone numbers.

    Parameters
    ----------
    values (pd.Series) : values before preprocessing

    Returns
    -------
    prepared (pd.Series) : values after processing
    '''

    prepared = _common_presteps(values)

    # manually remove stopwords, as TfidfVectorizer stopwords only applys if analyzer='word'
    prepared = _remove_stopwords(prepared, stopwords, 'phone')

    # parse using external library
    prepared = parse_components.phone(prepared)
    prepared = prepared['parsed']
    
    return _common_poststeps(prepared)


def email(values: pd.Series, stopwords='default') -> pd.Series:
    '''Prepocess email addresses.

    Parameters
    ----------
    prepared (pd.Series) : values before processing

    Returns
    -------
    prepared (DataFrame) : contains prepared email addresses and email domain
    prepared['email_address'] (pd.Series) : whole email address after processing
    prepared['email_domain'] (pd.Series) : email domain after processing
    '''

    prepared = _common_email(values)

    # manually remove stopwords, as TfidfVectorizer stopwords only applys if analyzer='word'
    prepared = _remove_stopwords(prepared, stopwords, 'email')

    # keep only letters and numbers
    prepared = prepared.str.replace(r'[\W_]+','', regex=True)

    return _common_poststeps(prepared)


def email_domain(values: pd.Series, stopwords='default') ->pd.Series:

    prepared = _common_email(values)

    # parse email domain
    prepared = prepared.str.extract('@(.*)')
    if prepared.shape[1]>1:
        raise NotImplementedError('Multiple domains detected.')
    else:
        prepared = prepared[0]

    # manually remove stopwords, as TfidfVectorizer stopwords only applys if analyzer='word'
    prepared = _remove_stopwords(prepared, stopwords, 'email_domain')

    return _common_poststeps(prepared)


def _common_email(values:pd.Series):

    email = _common_presteps(values)

    # remove all spaces
    email = email.replace(r'\s+', '', regex=True)

    return email


def address(values: pd.Series, stopwords='default') -> pd.Series:
    '''Prepocess street addresses.

    Parameters
    ----------
    values (pd.Series) : values before processing

    Returns
    -------
    prepared (pd.Series) : values after processing
    '''

    prepared = _common_presteps(values)

    # manually remove stopwords, as TfidfVectorizer stopwords only applys if analyzer='word'
    prepared = _remove_stopwords(prepared, stopwords, 'address')

    # remove ZIP+4 since commonly isn't given
    # TODO: allow option to include or ignore zip+4
    prepared = prepared.replace(to_replace=r'-\d+$', value='', regex=True)

    # keep only letters and numbers
    prepared = _alphanumeric_only(prepared)

    # introduce space between letters and digits
    prepared = prepared.replace(to_replace=r'(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)', value=' ', regex=True)

    # use common address abbreviations instead of full word
    pattern = {
        'avenue':'ave', 'terrace':'ter', 'terr': 'ter', 'court':'ct', 'street':'st',
        'place':'pl', 'lane':'ln', 'suite':'ste', 'building':'bldg', 'saint':'st',
        'apartment':'apt', 'fort':'ft', 'highway':'hwy', 'parkway':'pkwy', 'road':'rd',
        'drive': 'dr', 'boulevard': 'blvd',
        'beach': 'bch',
        'north':'n', 'south':'s', 'east':'e', 'west':'w',
        'northeast':'ne', 'northwest':'nw', 'southeast':'se', 'southwest':'sw'
    }
    replacer = flashtext.KeywordProcessor()
    for k,v in pattern.items():
        replacer.add_keyword(k,v)
    prepared = pd.Series(
        list(map(replacer.replace_keywords, prepared)),
        dtype='string', index=prepared.index
    )

    # remove space between single characters
    pattern = r'(?<=\b[^\W\d_])\s(?=[^\W\d_]\b)'
    prepared = prepared.str.replace(pattern, '', regex=True)

    # remove space between numbers and ordinal component
    pattern = r'(?<=1)\s+(?=st\b)|(?<=2)\s+(?=nd\b)|(?<=3)\s+(?=rd\b)|(?<=\d)\s+(?=th\b)'
    prepared = prepared.replace(to_replace=pattern, value='', regex=True)

    # remove unit like identifiers
    prepared = prepared.replace(to_replace=r'\b(lot)\b|\bbldg\b|\bapt\b|\bunit\b|\bste\b', value=' ', regex=True)

    # remove first part of address if starts with a text
    prepared = prepared.replace(to_replace=r'^\D+', value='', regex=True)

    return _common_poststeps(prepared)


comparison_rules = {
    "alphanumeric": {
        "comparer": "word",
        "cleaner": alphanumeric,
        "stopwords": None
    },
    "name": {
        "comparer": "char", 
        "cleaner": name,
        "stopwords": list(ENGLISH_STOP_WORDS)
    },
    "phone": {
        "comparer": "word", 
        "cleaner": phone,
        "stopwords": [
            r'^(\d)\1+$'
        ]
    },
    "email": {
        "comparer": "char", 
        "cleaner": email,
        "stopwords": [
            'noreply@noreply.com'
        ]
    },
    "email_domain": {
        "comparer": "char", 
        "cleaner": email_domain,
        "stopwords": [
            '.com','.net','.org',
            'gmail', 'yahoo', 'hotmail','aol','msn','noreply','comcast','outlook', 'att','verizon','icloud',
        ]
    },
    "address": {
        "comparer": "word",
        "cleaner": address,
        "stopwords": list(ENGLISH_STOP_WORDS)
    }
}