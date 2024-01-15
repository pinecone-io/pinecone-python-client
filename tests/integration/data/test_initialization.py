import pytest

class TestIndexClientInitialization:
    def test_index_direct_host_kwarg(self, client, index_host):
        index = client.Index(host=index_host)
        index.fetch(ids=['1', '2', '3'])

    def test_index_direct_host_with_https(self, client, index_host):
        if not index_host.startswith('https://'):
            index_host = 'https://' + index_host
        index = client.Index(host=index_host)
        index.fetch(ids=['1', '2', '3'])

    def test_index_direct_host_without_https(self, client, index_host):
        if index_host.startswith('https://'):
            index_host = index_host[8:]
        index = client.Index(host=index_host)
        index.fetch(ids=['1', '2', '3'])

    def test_index_by_name_positional_only(self, client, index_name, index_host):
        index = client.Index(index_name)
        index.fetch(ids=['1', '2', '3'])

    def test_index_by_name_positional_with_host(self, client, index_name, index_host):
        index = client.Index(index_name, index_host)
        index.fetch(ids=['1', '2', '3'])

    def test_index_by_name_kwargs(self, client, index_name):
        index = client.Index(name=index_name)
        index.fetch(ids=['1', '2', '3'])

    def test_index_by_name_kwargs_with_host(self, client, index_name, index_host):
        index = client.Index(name=index_name, host=index_host)
        index.fetch(ids=['1', '2', '3'])

    def test_raises_when_no_name_or_host(self, client, index_host):
        with pytest.raises(ValueError):
            client.Index()