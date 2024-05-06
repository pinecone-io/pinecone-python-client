from pinecone.exceptions import PineconeApiException, PineconeApiTypeError, PineconeApiValueError
from typing import List, Dict, Optional
import httpx

EXPECTED_RESPONSES = {
    "201": "The index has been successfully created.", 
    "400": "Bad request. The request body included invalid request parameters.", 
    "401": "Unauthorized. Possible causes: Invalid API key.", 
    "403": "You've exceed your pod quota.", 
    "404": "Unknown cloud or region when creating a serverless index.", 
    "422": "Unprocessable entity. The request body could not be deserialized.", 
    "409": "Index of given name already exists.", 
    "500": "Internal server error."
}

def generated_create_index(http_client: httpx.Client, name: str, dimension: int, spec: Dict, metric: Optional[str] = None):
    if not isinstance(name, str):
        raise PineconeApiTypeError("name must be a str")
    if not isinstance(dimension, int):
        raise PineconeApiTypeError("dimension must be a int")
    if metric is not None and not isinstance(metric, str):
        raise PineconeApiTypeError("metric must be a str")
    if not isinstance(spec, Dict):
        raise PineconeApiTypeError("spec must be a Dict")
    if len(name) < 1:
        raise PineconeApiValueError("name must be at least 1 characters long")
    if len(name) > 45:
        raise PineconeApiValueError("name must be at most 45 characters long")
    if dimension < 1:
        raise PineconeApiValueError("dimension must be at least 1")
    if dimension > 20000:
        raise PineconeApiValueError("dimension must be at most 20000")
    if metric is not None and metric not in ['cosine', 'euclidean', 'dotproduct']:
        raise PineconeApiValueError("metric must be one of ['cosine', 'euclidean', 'dotproduct']")

    request_body = {"name": name, "dimension": dimension, "metric": metric, "spec": spec}
    if request_body == {}:
        r = http_client.request('post', f'/indexes')
    else: 
        r = http_client.request('post', f'/indexes', json=request_body)

    if r.status_code < 300:
        return r.json()
    elif str(r.status_code) in EXPECTED_RESPONSES:
        raise PineconeApiException(f'HTTP Status: {r.status_code}. Message: {EXPECTED_RESPONSES[str(r.status_code)]} ')
    else:
        raise Exception(f'Unknown Error: {r.status_code} {r.text}')
    
