import pandas as pd

from entity_network.entity_resolver import entity_resolver

# def test_name_only():

#     df = pd.DataFrame({'Full Names': ['Jon Smith', 'John Smith', 'Jane Doe']})

#     er = entity_resolver(df)
#     er.compare('name', 'Full Names', threshold=0.8)
#     er.network()

#     assert 0==1