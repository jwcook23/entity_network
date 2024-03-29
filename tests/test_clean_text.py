from time import time

import pandas as pd
from faker import Faker

from entity_network import clean_text

fake = Faker(locale='en_US')

def test_possible_stopwords():

    values = pd.Series([
        'three   three one',
        'three two',
        'two',
    ])
    
    possible_stopwords = clean_text.possible_stopwords(values)
    
    assert possible_stopwords.equals(
        pd.Series([
            3, 2, 1
        ], index=['three','two','one'])
    )


def test_name():

    values = pd.Series([
        "Foo Bar's of   City",
        "Baz's Foo.",
        ""
    ])

    prepared = clean_text.name(values)

    assert prepared.equals(
        pd.Series([
            "foo bar s city",
            "baz s foo",
            pd.NA
        ], dtype='string')
    )


def test_phone():

    values = pd.Series([
        "123456789",
        "111111111",
        "",
        "Some Name",
        '(1) 555-456-7890 extension 12'
    ])

    prepared = clean_text.phone(values)

    expected = pd.Series([
        "1 123456789",
        pd.NA,
        pd.NA,
        pd.NA,
        "1 5554567890 ext 12"
    ], dtype='string')
    assert prepared.equals(expected)


def test_email():

    values = pd.Series([
        "some_email@gmail.com",
        "name@companyname.org",
        "",
    ])

    prepared = clean_text.email(values)

    assert prepared.equals(
        pd.Series([
            "someemailgmailcom",
            "namecompanynameorg",
            pd.NA,
        ], dtype='string')
    )


def test_email_domain():

    values = pd.Series([
        "some_email@gmail.com",
        "name@companyname.org",
        "",
    ])

    prepared = clean_text.email_domain(values)

    assert prepared.equals(
        pd.Series([
            pd.NA,
            "companyname",
            pd.NA,
        ], dtype='string')
    )


def test_address():

    values = pd.Series([
        "123 North RoadName Road, FL 12345-6789",
        "name@companyname.org",
        "",
        "comapany 123 N E RoadName Road, FL. 12345-6789",
        "123 apartment #4/5 North RoadName Road, FL 12345",
        "123 North roadName Road Apartment 3C, FL 12345-6789",
    ])

    prepared = clean_text.address(values)

    assert prepared.equals(
        pd.Series([
            "123 n roadname rd fl 12345",
            pd.NA,
            pd.NA,
            "123 ne roadname rd fl 12345",
            "123 4 5 n roadname rd fl 12345",
            "123 n roadname rd 3 c fl 12345",
        ], dtype='string')
    )


def test_address_speed():

    values = [fake.address() for _ in range(1000)]*100
    values = pd.Series(values)

    tstart = time()
    _ = clean_text.address(values)
    duration = time()-tstart

    assert duration<10