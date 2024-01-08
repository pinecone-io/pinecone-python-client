from pinecone.utils import normalize_host

def test_when_url_is_none():
    assert normalize_host(None) is None

def test_when_url_is_https():
    assert normalize_host('https://index-name-abcdef.svc.pinecone.io') == 'https://index-name-abcdef.svc.pinecone.io'

def test_when_url_is_http():
    # This should not occur in prod, but if it does, we will leave it alone. 
    # Could be useful when testing with local proxies.
    assert normalize_host('http://index-name-abcdef.svc.pinecone.io') == 'http://index-name-abcdef.svc.pinecone.io'

def test_when_url_is_host_without_protocol():
    assert normalize_host('index-name-abcdef.svc.pinecone.io') == 'https://index-name-abcdef.svc.pinecone.io'

def test_can_be_called_multiple_times():
    assert normalize_host(normalize_host('index-name-abcdef.svc.pinecone.io')) == 'https://index-name-abcdef.svc.pinecone.io'