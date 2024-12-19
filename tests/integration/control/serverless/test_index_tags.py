class TestIndexTags:
    def test_index_tags_none_by_default(self, client, index_name):
        client.describe_index(name=index_name)
        assert client.describe_index(name=index_name) is None
        
    def test_add_index_tags(self, client, index_name):
        client.add_index_tags(name=index_name, tags={'foo': 'FOO', 'bar': 'BAR'})
        assert client.describe_index(name=index_name).tags == {'foo': 'FOO', 'bar': 'BAR'}