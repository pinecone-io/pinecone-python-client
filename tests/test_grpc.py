import unittest

import numpy as np

from pinecone import utils, grpc


class TestGRPC(unittest.TestCase):
    def test_grpc_index_request_vector_shape_unchanged(self):
        client = grpc.GRPCClient()
        array = np.ones((3, 3, 3))
        req = client.get_index_request(data=array)
        out_array = utils.load_numpy(req.index.data)
        np.testing.assert_array_equal(array, out_array)

    def test_grpc_query_request_vector_shape_unchanged(self):
        client = grpc.GRPCClient()
        array = np.ones((3, 3, 3))
        req = client.get_query_request(data=array)
        out_array = utils.load_numpy(req.query.data)
        np.testing.assert_array_equal(array, out_array)
