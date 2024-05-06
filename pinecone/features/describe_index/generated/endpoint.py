from pinecone.exceptions import PineconeApiException, PineconeApiTypeError, PineconeApiValueError
from typing import List, Dict, Optional
import httpx

EXPECTED_RESPONSES = {
    "200": "Configuration information and deployment status of the index.", 
    "401": "Unauthorized. Possible causes: Invalid API key.", 
    "404": "Index not found.", 
    "500": "Internal server error."
}

def generated_describe_index(http_client: httpx.Client, index_name: str):

    request_body = {}
    if request_body == {}:
        r = http_client.request('get', f'/indexes/{index_name}')
    else: 
        r = http_client.request('get', f'/indexes/{index_name}', json=request_body)

    if r.status_code < 300:
        return r.json()
    elif str(r.status_code) in EXPECTED_RESPONSES:
        raise PineconeApiException(f'HTTP Status: {r.status_code}. Message: {EXPECTED_RESPONSES[str(r.status_code)]} ')
    else:
        raise Exception(f'Unknown Error: {r.status_code} {r.text}')
    
