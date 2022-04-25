'''Text cleaning functions for different categories of data.'''

from pandas import DataFrame, Series, NA
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

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
    prepared[prepared.str.len()==0] = NA

    return prepared


def _remove_stopwords(prepared, stopwords): 

    pattern = r'\b(?:{})\b'.format('|'.join(stopwords))
    prepared = prepared.str.replace(pattern, '', regex=True)

    return prepared


def possible_stopwords(values) -> Series: 
    '''Count words in decending order.
    
    Parameters
    ----------
    values (Series) : sentenances of words

    Returns
    -------
    possible_stopwords (Series) : word and count of word apperance
    '''

    possible_stopwords = values.str.split(r'\s+')
    possible_stopwords = possible_stopwords.explode()
    possible_stopwords = possible_stopwords.value_counts()

    return possible_stopwords


def name(values: Series) -> Series:
    '''Prepocess comapany and person names.

    Parameters
    ----------
    values (Series) : values before processing

    Returns
    -------
    prepared (Series) : values after processing
    '''  

    prepared = _common_presteps(values)
    prepared = _alphanumeric_only(prepared)

    # manually remove stopwords, as TfidfVectorizer stopwords only applys if analyzer='word'
    stopwords = list(ENGLISH_STOP_WORDS)
    stopwords = stopwords+[
        # generic
        'company','associates','group','corporation',
        'llc','inc','law','corp','group','na',
        # industry specific
        'insurance','claim','claims',
        'mortgage','loan','loans','bank','isaoa','atima','financial',
        'adjusters','adjusting','agency','pa',
        'remodeling','service','services','servicing',
        'consulting','consultants','restoration','reporting',
        
    ]
    prepared = _remove_stopwords(prepared, stopwords)

    return _common_poststeps(prepared)


def phone(values: Series) -> Series:
    '''Prepocess phone numbers.

    Parameters
    ----------
    values (Series) : values before preprocessing

    Returns
    -------
    prepared (Series) : values after processing
    '''

    prepared = _common_presteps(values)

    # remove trailing zeros with decimal from mixed types
    prepared = prepared.replace('\.0$', '', regex=True)

    # keep numbers only
    prepared = prepared.replace(r'[^0-9]+', '', regex=True)

    # manually remove stopwords, as TfidfVectorizer stopwords only applys if analyzer='word'
    stopwords = [
        # words containing only one repeating digit
        r'^(\d)\1+$'
    ]
    prepared = _remove_stopwords(prepared, stopwords)

    return _common_poststeps(prepared)


def email(values: Series) -> Series:
    '''Prepocess email addresses.

    Parameters
    ----------
    prepared (Series) : values before processing

    Returns
    -------
    prepared (DataFrame) : contains prepared email addresses and email domain
    prepared['email_address'] (Series) : whole email address after processing
    prepared['email_domain'] (Series) : email domain after processing
    '''

    email = _common_email(values)

    # keep only letters and numbers
    email = email.str.replace(r'[\W_]+','', regex=True)

    return _common_poststeps(email)


def email_domain(values: Series) ->Series:

    email = _common_email(values)

    # parse email domain
    domain = email.str.extract('@(.*)')
    stopwords = [
        # common domains
        '.com','.net','.org',
        # common domain names
        'gmail', 'yahoo', 'hotmail','aol','msn','noreply','comcast','outlook',
        'att','verizon','icloud',
        # common insurance domain names
        'statefarm','allstate'
    ]
    if domain.shape[1]>1:
        raise NotImplementedError('Multiple domains detected.')
    else:
        domain = domain[0]
    domain = _remove_stopwords(domain, stopwords)

    return _common_poststeps(domain)


def _common_email(values:Series):

    email = _common_presteps(values)

    # remove all spaces
    email = email.replace(r'\s+', '', regex=True)

    # manually remove stopwords, as TfidfVectorizer stopwords only applys if analyzer='word'
    stopwords = [
        # empty filler values
        'noreply@noreply.com',
    ]
    email = _remove_stopwords(email, stopwords)

    return email


def address(values: Series) -> Series:
    '''Prepocess street addresses.

    Parameters
    ----------
    values (Series) : values before processing

    Returns
    -------
    prepared (Series) : values after processing
    '''

    prepared = _common_presteps(values)

    # remove ZIP+4 since commonly isn't given
    prepared = prepared.replace(to_replace=r'-\d+$', value='', regex=True)

    # keep only letters and numbers
    prepared = _alphanumeric_only(prepared)

    # introduce space between letters and digits
    prepared = prepared.replace(to_replace=r'(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)', value=' ', regex=True)

    # use common address abbreviations instead of full word
    pattern = {
        'avenue':'ave', 'terrace':'ter', 'court':'ct', 'street':'st',
        'place':'pl', 'lane':'ln', 'suite':'ste', 'building':'bldg', 'saint':'st',
        'apartment':'apt', 'fort':'ft', 'highway':'hwy', 'parkway':'pkwy', 'road':'rd',
        'north':'n', 'south':'s', 'east':'e', 'west':'w',
        'northeast':'ne', 'northwest':'nw', 'southeast':'se', 'southwest':'sw'
    }
    pattern = {r'\b'+k+r'\b':v for k,v in pattern.items()}
    prepared = prepared.fillna('')
    prepared = prepared.replace(pattern, regex=True)

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
