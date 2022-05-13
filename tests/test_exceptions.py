import pandas as pd
import pytest

from entity_network.entity_resolver import entity_resolver
from entity_network import _exceptions

def test_DuplicatedIndex():

    with pytest.raises(_exceptions.DuplicatedIndex):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,1])
        er = entity_resolver(df)

def test_MissingColumn():
    with pytest.raises(_exceptions.MissingColumn):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver(df)
        er.compare('phone', 'ColumnB', kneighbors=10, threshold=1)

def test_InvalidCategory():
    with pytest.raises(_exceptions.InvalidCategory):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver(df)
        er.compare('foo', 'ColumnB', kneighbors=10, threshold=1)

def test_ThresholdRange():
    with pytest.raises(_exceptions.ThresholdRange):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver(df)
        er.compare('email', 'ColumnA', kneighbors=10, threshold=100)

def test_KneighborsRange():
    with pytest.raises(_exceptions.KneighborsRange):
        df = pd.DataFrame({'ColumnA': ['a','b']}, index=[1,2])
        er = entity_resolver(df)
        er.compare('email', 'ColumnA', kneighbors=-1.2, threshold=0.8)