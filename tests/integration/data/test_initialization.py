import pytest

class TestIndexClientInitialization:
    def test_index_direct_host_kwarg(client, index_host):
        index = client.Index(host=index_host)
        index.find(ids=['1', '2', '3'])

    def test_index_direct_host_with_https(client, index_host):
        if not index_host.startswith('https://'):
            index_host = 'https://' + index_host
        index = client.Index(index_host)
        index.find(ids=['1', '2', '3'])

    def test_index_direct_host_without_http(client, index_host):
        if index_host.startswith('https://'):
            index_host = index_host[8:]
        index = client.Index(host=index_host)
        index.find(ids=['1', '2', '3'])

    def test_index_by_name_positional(client, index_name):
        index = client.Index(index_name)
        index.find(ids=['1', '2', '3'])

    def test_index_by_name_positional_with_host(client, index_name, index_host):
        index = client.Index(index_name, index_host)
        index.find(ids=['1', '2', '3'])

    def test_index_by_name_kwargs(client, index_name):
        index = client.Index(name=index_name)
        index.find(ids=['1', '2', '3'])

    def test_index_by_name_kwargs_with_host(client, index_name, index_host):
        index = client.Index(name=index_name, host=index_host)
        index.find(ids=['1', '2', '3'])