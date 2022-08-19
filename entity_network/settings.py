from entity_network import clean_text

text_comparer = {
    "name": "char",
    "phone": "word",
    "email": "char",
    "email_domain": "char",
    "address": "word"
}

text_cleaner = {
    'name': clean_text.name,
    'phone': clean_text.phone,
    'email': clean_text.email,
    'email_domain': clean_text.email_domain,
    'address': clean_text.address
}