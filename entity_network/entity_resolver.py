'''Find matching values in a column of data. Matches may be exact or similar according to a threshold.'''
import pandas as pd
import networkx as nx

from entity_network import _exceptions, _conversion, _core

class entity_resolver():


    def __init__(self, df):

        if df.index.has_duplicates:
            raise _exceptions.DuplicatedIndex('Argument df index must be unique.')
        reserved = pd.Series(['entity_id','network_id'])
        if reserved.isin(df.columns).any():
            raise _exceptions.ReservedColumn(f'Argument df cannot contain the columns: {reserved[reserved.isin(df.columns)].to_list()}')

        self.entity_feature = None
        self.entity_id = None
        self.network_feature = None
        self.network_id = None
        self.related = {}
        self.similar = {}

        self._df = df
        self._other = []
        self._processed = {}


    def compare(self, category, columns, kneighbors, threshold, analyzer='default', preprocessor='default'):

        # track other categories compared beside name which form a network
        if category!='name':
            self._other.append(category)

        # prepare values for comparison
        self._processed[category] = _core.prepare_values(self._df, category, columns, preprocessor)
        self._processed[category].index.names = ('index', 'column')
        self._processed[category].name = category

        # ignore values the processor completely removed
        # self._processed[category] = self._processed[category].dropna()

        # compare values on similarity threshold
        related, similar = _core.find_related(category, self._processed[category], kneighbors=kneighbors, threshold=threshold, analyzer=analyzer)
        self.related[category] = related
        self.similar[category] = similar


    def entity(self, columns, kneighbors, threshold, analyzer='default'):

        self.compare('name', columns, kneighbors, threshold, analyzer)

        # form graph of matching names
        name = _conversion.from_pandas_df_id(self.related['name'], 'name_id')

        # compare name graph to graphs of other features to resolve entities
        self.entity_feature = []
        for c in self._other:
            other = _conversion.from_pandas_df_id(self.related[c], f'{c}_id')
            other = nx.intersection(name, other)
            other.remove_nodes_from(list(nx.isolates(other)))
            other = list(nx.connected_components(other))
            other = pd.DataFrame({'index': other})
            other['category'] = c
            self.entity_feature.append(other)
        self.entity_feature = pd.concat(self.entity_feature, ignore_index=True)

        # assign entity_id to all records
        self.entity_id, self.entity_feature = self._assign_id(self.entity_feature, 'entity_id')

        # assign entity_name
        self.entity_id = self._assign_name(self.entity_id, 'entity_id', 'entity_name')

        # assign entity_id to original dataframe
        self._df = self._df.merge(self.entity_id[['entity_id']], left_index=True, right_index=True, how='left')
        self._df['entity_id'] = self._df['entity_id'].astype('Int64')

        return self._df, self.entity_id, self.entity_feature


    def network(self):
        
        # determine network by matching features
        self.network_feature = []
        for c in self._other:
            other = self.related[c].groupby(f'{c}_id')
            other = other.agg({'index': set})
            other = other[other['index'].str.len()>1]
            other['category'] = c
            self.network_feature.append(other)
        self.network_feature = pd.concat(self.network_feature)

        # assign network_id to all records
        self.network_id, self.network_feature = self._assign_id(self.network_feature, 'network_id')

        # assign network name
        self.network_id = self._assign_name(self.network_id, 'network_id', 'network_name')

        # assign network_id to original dataframe
        self._df = self._df.merge(self.network_id[['network_id']], left_index=True, right_index=True)
        self._df['network_id'] = self._df['network_id'].astype('Int64')

        return self._df, self.network_id, self.network_feature

    def _assign_id(self, connected, name_id):

        # assign id to connected indices
        assigned_id = _conversion.from_pandas_series(connected['index'])
        assigned_id = pd.DataFrame({'index': list(nx.connected_components(assigned_id))})
        assigned_id.index.name = name_id
        assigned_id = assigned_id.explode('index').reset_index()

        # assign id to indices that aren't connected
        unassigned = self._df.index[~self._df.index.isin(assigned_id['index'])]
        unassigned = pd.DataFrame({'index': unassigned})
        seed = assigned_id[name_id].max()+1
        assigned_id = pd.concat([assigned_id, unassigned])
        unassigned = assigned_id[name_id].isna()
        assigned_id.loc[unassigned,name_id] = range(seed, seed+sum(unassigned))
        assigned_id[name_id] = assigned_id[name_id].astype('int64')

        # sort by index of original dataframe
        assigned_id = assigned_id.sort_values('index')

        # expand nested input connected features and assign the id
        connected = connected.explode('index')
        connected = connected.merge(assigned_id, on='index')

        # sort by index of original dataframe
        connected = connected.sort_values('index')

        # set index of feature dataframe as original source index
        connected = connected.set_index('index')

        return assigned_id, connected


    def _assign_name(self, df_id, name_id, name_out):

        common = self._processed['name'].reset_index()
        common = common.drop(columns='column')
        common = df_id.merge(common, on='index')
        common = common.value_counts([name_id,'name']).reset_index()
        common = common.drop(columns=0)
        common = common.drop_duplicates(subset=name_id)
        common = common.rename(columns={'name': name_out})
        df_id = df_id.merge(common, on=name_id)

        # sort by index of original dataframe
        df_id = df_id.sort_values('index')

        # set index of id dataframe as original source index
        df_id = df_id.set_index('index')

        return df_id