'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
import pandas as pd
import networkx as nx

from entity_network import _exceptions, _conversion, _core

class entity_resolver():


    def __init__(self, df):

        if df.index.has_duplicates:
            raise _exceptions.DuplicatedIndex('Argument df index must be unique.')

        self.df = df
        self.other_category = []
        self.processed = {}
        self.graph = {}
        self.related = {}
        self.similar = {}


    def compare(self, category, columns, kneighbors, threshold):

        # track other categories compared beside name which form a network
        if category!='name':
            self.other_category.append(category)

        # prepare values for comparison
        self.processed[category] = _core.prepare_values(category, columns)
        self.processed[category].index.names = ('index', 'column')
        self.processed[category].name = category

        # ignore values the processor completely removed
        self.processed[category] = self.processed[category].dropna()

        # compare values on similarity threshold
        related, similar = _core.find_related(category, self.processed[category], kneighbors=kneighbors, threshold=threshold)
        self.related[category] = related
        self.similar[category] = similar


    def entity(self, columns, kneighbors, threshold):

        self.compare_records('name', columns, kneighbors, threshold)

        # form graph of matching names
        self.graph['name'] = _conversion.from_pandas_df_id(self.related['name'], 'name_id')

        # compare name graph to graphs of other features to resolve entities
        self.entity_map = []
        for c in self.other_category:
            self.graph[c] = _conversion.from_pandas_df_id(self.related[c], f'{c}_id')
            other = nx.intersection(self.graph['name'], self.graph[c])
            other.remove_nodes_from(list(nx.isolates(other)))
            other = list(nx.connected_components(other))
            other = pd.DataFrame({'index': other})
            other['Match'] = c
            self.entity_map.append(other)
        self.entity_map = pd.concat(self.entity_map, ignore_index=True)

        # assign entity_id to all records
        self.entity_id = _conversion.from_pandas_series(self.entity_map['index'])
        self.entity_id = pd.DataFrame({'index': list(nx.connected_components(self.entity_id))})
        self.entity_id.index.name = 'entity_id'
        self.entity_id = self.entity_id.explode('index').reset_index()
        unresolved = ~self.related['name']['index'].isin(self.entity_id['index'])
        unresolved = self.related['name'].loc[unresolved, ['index']].drop_duplicates()
        seed = self.entity_id['entity_id'].max()+1
        self.entity_id = pd.concat([self.entity_id, unresolved])
        unresolved = self.entity_id['entity_id'].isna()
        self.entity_id.loc[unresolved,'entity_id'] = range(seed, seed+sum(unresolved))
        self.entity_id['entity_id'] = self.entity_id['entity_id'].astype('int64')

        # assign entity name using most frequent name for each entity_id
        common = self.processed['name'].reset_index()
        common = common.drop_duplicates(subset='index').drop(columns='column')
        common = self.entity_id.merge(common, on='index')
        common = common.value_counts(['entity_id','name']).reset_index()
        common = common.drop(columns=0)
        common = common.drop_duplicates(subset='entity_id')
        common = common.rename(columns={'name': 'entity_name'})
        self.entity_id = self.entity_id.merge(common, on='entity_id')


    def network(self):
        
        network = []
        for c in self.other_category:
            other = self.related[c].groupby(f'{c}_id')
            other = other.agg({'index': 'unique'})
            other = other[other['index'].str.len()>1]
            other['Match'] = c
            network.append(other)
        network = pd.concat(network)
