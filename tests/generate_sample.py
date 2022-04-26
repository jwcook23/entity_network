import pandas as pd
from faker import Faker


fake = Faker(locale='en_US')

number_people = 10
number_companies = 10
percent_empty = 0.05
percent_duplicates = 0.2


def random_people():
    df = pd.DataFrame({
        'PersonName': [fake.name() for _ in range(number_people)],
        'CompanyName': [None]*number_people,
        'Email': [fake.free_email_domain() for _ in range(number_people)],
        'HomePhone': [fake.phone_number() for _ in range(number_people)],
        'WorkPhone': [fake.phone_number() for _ in range(number_people)],
        'CellPhone': [fake.phone_number() for _ in range(number_people)],
        'Address': [fake.address() for _ in range(number_people)]
    })

    return df


random_people()