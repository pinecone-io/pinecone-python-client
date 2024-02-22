import pytest
import time

def isNonEmptyString(obj):
    if obj == None:
        return False
    if isinstance(obj, str) == False:
        return False
    if str == '':
        return False
    return True

class TestIndexes():
    def test_list_indexes(self, client, ready_index):
        index_list = client.list_indexes()
        assert ready_index in index_list.names()
        assert len(index_list) > 0

        index_description = index_list.indexes[0]
        assert isNonEmptyString(index_description['name']) == True
        assert isNonEmptyString(index_description['host']) == True
        
    def test_describe_index(self, client, ready_index, dimension, metric, environment):
        description = client.describe_index(ready_index)
        print(description)
        assert description['name'] == ready_index
        assert isNonEmptyString(description['host']) == True
        assert description['status']['ready'] == True
        assert description['status']['state'] == 'Ready'
        assert description['dimension'] == dimension
        assert description['metric'] == metric
        assert description['spec'] != None
        assert description['spec']['pod']['environment'] == environment
        assert description['spec']['pod']['pod_type'] == 'p1.x1'
        assert description['spec']['pod']['replicas'] == 1

    def test_configure_pod_index(self, client, ready_index):
        time.sleep(10) # Wait a little more, just in case.
        client.configure_index(ready_index, replicas=1, pod_type='p1.x1')