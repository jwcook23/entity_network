import sample
import pandas as pd
import pytest

from entity_network.entity_resolver import entity_resolver
from entity_network import _exceptions


def test_argument_exceptions():

    with pytest.raises(_exceptions.ReservedColumn):
        df = pd.DataFrame({'ColumnA': ['a','b'], 'entity_id': [0, 1], 'network_id': [0, 1]})
        er = entity_resolver.entity_resolver(df)

    with pytest.raises(_exceptions.DuplicatedIndex):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,1])
        er = entity_resolver.entity_resolver(df)

    with pytest.raises(_exceptions.MissingColumn):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver.entity_resolver(df)
        er.compare('phone', 'ColumnB', kneighbors=10, threshold=1)

    with pytest.raises(_exceptions.InvalidCategory):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver.entity_resolver(df)
        er.compare('foo', 'ColumnB', kneighbors=10, threshold=1)

    with pytest.raises(_exceptions.ThresholdRange):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver.entity_resolver(df)
        er.compare('email', 'ColumnA', kneighbors=10, threshold=100)

    with pytest.raises(_exceptions.KneighborsRange):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver.entity_resolver(df)
        er.compare('email', 'ColumnA', kneighbors=-1.2, threshold=0.8)


def test_network():

    df = sample.unique_records(1000)
    df = sample.duplicate_records(df, 30)

    er = entity_resolver(df)
    er.compare('phone', columns=['HomePhone','WorkPhone','CellPhone'])
