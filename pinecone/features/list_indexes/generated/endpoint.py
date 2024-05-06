from pinecone.exceptions import PineconeApiException, PineconeApiTypeError, PineconeApiValueError
from typing import List, Dict, Optional
import httpx

EXPECTED_RESPONSES = {
    "200": "This operation returns a list of all the indexes that you have previously created, and which are associated with the given project", 
    "401": "Unauthorized. Possible causes: Invalid API key.", 
    "500": "Internal server error."
}

def generated_list_indexes(http_client: httpx.Client):

    request_body = {}
    if request_body == {}:
        r = http_client.request('get', f'/indexes')
    else: 
        r = http_client.request('get', f'/indexes', json=request_body)

    if r.status_code < 300:
        return r.json()
    elif str(r.status_code) in EXPECTED_RESPONSES:
        raise PineconeApiException(f'HTTP Status: {r.status_code}. Message: {EXPECTED_RESPONSES[str(r.status_code)]} ')
    else:
        raise Exception(f'Unknown Error: {r.status_code} {r.text}')
    
