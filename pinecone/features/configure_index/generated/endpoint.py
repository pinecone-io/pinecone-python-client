from pinecone.exceptions import PineconeApiException, PineconeApiTypeError, PineconeApiValueError
from typing import List, Dict, Optional
import httpx

EXPECTED_RESPONSES = {
    "202": "The request to configure the index has been accepted. Check the  index status to see when the change has been applied.", 
    "400": "Bad request. The request body included invalid request parameters.", 
    "401": "Unauthorized. Possible causes: Invalid API key.", 
    "403": "You've exceed your pod quota.", 
    "404": "Index not found.", 
    "422": "Unprocessable entity. The request body could not be deserialized.", 
    "500": "Internal server error."
}

def generated_configure_index(http_client: httpx.Client, index_name: str, spec: Dict):
    if not isinstance(spec, Dict):
        raise PineconeApiTypeError("spec must be a Dict")

    request_body = {"spec": spec}
    if request_body == {}:
        r = http_client.request('patch', f'/indexes/{index_name}')
    else: 
        r = http_client.request('patch', f'/indexes/{index_name}', json=request_body)

    if r.status_code < 300:
        return r.json()
    elif str(r.status_code) in EXPECTED_RESPONSES:
        raise PineconeApiException(f'HTTP Status: {r.status_code}. Message: {EXPECTED_RESPONSES[str(r.status_code)]} ')
    else:
        raise Exception(f'Unknown Error: {r.status_code} {r.text}')
    
