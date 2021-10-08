import pytest
from urllib3_mock import Responses

import pinecone
from pinecone import ApiTypeError

responses = Responses('requests.packages.urllib3')


@responses.activate
def test_unrecognized_response_field():
    # unrecognized response fields are okay, shouldn't raise an exception
    pinecone.init('example-api-key', environment='example-environment')

    # responses.add('GET', '/actions/whoami',  # fixme: requests-based, so mock fails?
    #               body='{"project_name": "example-project", "user_label": "example-label", "user_name": "test"}',
    #               status=200, content_type='application/json')
    responses.add('DELETE', '/vectors/delete',
                  body='{"deleted_count": 2, "unexpected_key": "xyzzy"}',
                  status=200, content_type='application/json')

    index = pinecone.Index('example-index')
    resp = index.delete(ids=['vec1', 'vec2'])

    # assert len(responses.calls) == 1
    # assert responses.calls[0].request.url == '/vectors/delete?ids=vec1&ids=vec2'
    # assert responses.calls[0].request.host == 'example-index-unknown.svc.example-environment.pinecone.io'
    # assert responses.calls[0].request.scheme == 'https'

    assert resp.deleted_count == 2


@responses.activate
def test_missing_response_field():
    # unrecognized response fields are okay, shouldn't raise an exception
    pinecone.init('example-api-key', environment='example-environment')
    responses.add('DELETE', '/vectors/delete',
                  body='{}',
                  status=200, content_type='application/json')
    index = pinecone.Index('example-index')
    # this should not raise
    index.delete(ids=['vec1', 'vec2'])



@responses.activate
def test_malformed_response_wrong_type():
    # unrecognized response fields are okay, shouldn't raise an exception
    pinecone.init('example-api-key', environment='example-environment')

    responses.add('DELETE', '/vectors/delete',
                  body='{"deleted_count": "foobar"}',
                  status=200, content_type='application/json')

    index = pinecone.Index('example-index')

    with pytest.raises(ApiTypeError) as exc_info:
        resp = index.delete(ids=['vec1', 'vec2'])
        assert resp.deleted_count == 2
