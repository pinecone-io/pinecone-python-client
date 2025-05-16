from http.server import BaseHTTPRequestHandler, HTTPServer
import json

backups_response = {
    "data": [
        {
            "backup_id": "6f52240b-6397-481b-9767-748a2d4d3b65",
            "source_index_name": "jensparse",
            "source_index_id": "71ded150-2b8e-422d-9849-097f2c89d18b",
            "status": "Ready",
            "cloud": "aws",
            "region": "us-east-1",
            "tags": {},
            "name": "sparsebackup",
            "description": "",
            "dimension": 0,
            "record_count": 10000,
            "namespace_count": 1000,
            "size_bytes": 123456,
            "created_at": "2025-05-15T20:55:29.477794Z",
        }
    ]
}

indexes_response = {
    "indexes": [
        {
            "name": "jhamon-20250515-165135548-reorg-create-with-e",
            "metric": "dotproduct",
            "host": "jhamon-20250515-165135548-reorg-create-with-e-bt8x3su.svc.aped-4627-b74a.pinecone.io",
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
            "status": {"ready": True, "state": "Ready"},
            "vector_type": "sparse",
            "dimension": None,
            "deletion_protection": "disabled",
            "tags": {"env": "dev"},
        },
        {
            "name": "unexpected",
            "metric": "newmetric",
            "host": "jhamon-20250515-165135548-reorg-create-with-e-bt8x3su.svc.aped-4627-b74a.pinecone.io",
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
            "status": {"ready": False, "state": "UnknownStatus"},
            "vector_type": "sparse",
            "dimension": -1,
            "deletion_protection": "disabled",
            "tags": {"env": "dev"},
        },
        {
            "name": "wrong-types",
            "metric": 123,
            "host": "jhamon-20250515-165135548-reorg-create-with-e-bt8x3su.svc.aped-4627-b74a.pinecone.io",
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
            "status": {"ready": False, "state": "UnknownStatus"},
            "vector_type": None,
            "dimension": None,
            "deletion_protection": "asdf",
            "tags": None,
        },
    ]
}

index_description_response = {
    "name": "docs-example-dense",
    "vector_type": "dense",
    "metric": "cosine",
    "dimension": 1536,
    "status": {"ready": True, "state": "Ready"},
    "host": "docs-example-dense-govk0nt.svc.aped-4627-b74a.pinecone.io",
    "spec": {"serverless": {"region": "us-east-1", "cloud": "aws"}},
    "deletion_protection": "disabled",
    "tags": {"environment": "development"},
}

upsert_response = {"upsertedCount": 10}

call_count = 0


class MyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global call_count
        call_count += 1

        # Simulate a high rate of 500 errors
        if call_count % 5 != 0:
            self.send_response(500)
            self.end_headers()
            return

        if self.path.startswith("/vectors/upsert"):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = upsert_response
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        global call_count
        call_count += 1

        # Simulate a high rate of 500 errors
        if call_count % 5 != 0:
            self.send_response(500)
            self.end_headers()
            return

        if self.path.startswith("/backups"):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = backups_response
            self.wfile.write(json.dumps(response).encode())
        elif self.path.startswith("/indexes/"):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = index_description_response
            self.wfile.write(json.dumps(response).encode())
        elif self.path.startswith("/indexes"):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = indexes_response
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()


server = HTTPServer(("localhost", 8000), MyHandler)
print("Serving on http://localhost:8000")
server.serve_forever()
