from entity_network import clean_text

text = {
    "name": {"comparer": "char", "cleaner": clean_text.name},
    "phone": {"comparer": "word", "cleaner": clean_text.phone},
    "email": {"comparer": "char", "cleaner": clean_text.email},
    "email_domain": {"comparer": "char", "cleaner": clean_text.email_domain},
    "address": {"comparer": "word", "cleaner": clean_text.address}
}