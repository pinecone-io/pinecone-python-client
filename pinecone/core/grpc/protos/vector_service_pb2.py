# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vector_service.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2
from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2
from google.api import (
    field_behavior_pb2 as google_dot_api_dot_field__behavior__pb2,
)
from protoc_gen_openapiv2.options import (
    annotations_pb2 as protoc__gen__openapiv2_dot_options_dot_annotations__pb2,
)


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x14vector_service.proto\x1a\x1cgoogle/protobuf/struct.proto\x1a\x1cgoogle/api/annotations.proto\x1a\x1fgoogle/api/field_behavior.proto\x1a.protoc-gen-openapiv2/options/annotations.proto"\x80\x01\n\x0cSparseValues\x12\x36\n\x07indices\x18\x01 \x03(\rB%\x92\x41\x1eJ\x16[1, 312, 822, 14, 980]x\xe8\x07\x80\x01\x01\xe2\x41\x01\x02\x12\x38\n\x06values\x18\x02 \x03(\x02\x42(\x92\x41!J\x19[0.1, 0.2, 0.3, 0.4, 0.5]x\xe8\x07\x80\x01\x01\xe2\x41\x01\x02"\xff\x01\n\x06Vector\x12-\n\x02id\x18\x01 \x01(\tB!\x92\x41\x1aJ\x12"example-vector-1"x\x80\x04\x80\x01\x01\xe2\x41\x01\x02\x12H\n\x06values\x18\x02 \x03(\x02\x42\x38\x92\x41\x31J([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]x\xa0\x9c\x01\x80\x01\x01\xe2\x41\x01\x02\x12$\n\rsparse_values\x18\x04 \x01(\x0b\x32\r.SparseValues\x12V\n\x08metadata\x18\x03 \x01(\x0b\x32\x17.google.protobuf.StructB+\x92\x41(J&{"genre": "documentary", "year": 2019}"\x94\x02\n\x0cScoredVector\x12-\n\x02id\x18\x01 \x01(\tB!\x92\x41\x1aJ\x12"example-vector-1"x\x80\x04\x80\x01\x01\xe2\x41\x01\x02\x12\x18\n\x05score\x18\x02 \x01(\x02\x42\t\x92\x41\x06J\x04\x30.08\x12=\n\x06values\x18\x03 \x03(\x02\x42-\x92\x41*J([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]\x12$\n\rsparse_values\x18\x05 \x01(\x0b\x32\r.SparseValues\x12V\n\x08metadata\x18\x04 \x01(\x0b\x32\x17.google.protobuf.StructB+\x92\x41(J&{"genre": "documentary", "year": 2019}"\x89\x01\n\x0cRequestUnion\x12 \n\x06upsert\x18\x01 \x01(\x0b\x32\x0e.UpsertRequestH\x00\x12 \n\x06\x64\x65lete\x18\x02 \x01(\x0b\x32\x0e.DeleteRequestH\x00\x12 \n\x06update\x18\x03 \x01(\x0b\x32\x0e.UpdateRequestH\x00\x42\x13\n\x11RequestUnionInner"e\n\rUpsertRequest\x12\'\n\x07vectors\x18\x01 \x03(\x0b\x32\x07.VectorB\r\x92\x41\x06x\xe8\x07\x80\x01\x01\xe2\x41\x01\x02\x12+\n\tnamespace\x18\x02 \x01(\tB\x18\x92\x41\x15J\x13"example-namespace""1\n\x0eUpsertResponse\x12\x1f\n\x0eupserted_count\x18\x01 \x01(\rB\x07\x92\x41\x04J\x02\x31\x30"\xb6\x01\n\rDeleteRequest\x12(\n\x03ids\x18\x01 \x03(\tB\x1b\x92\x41\x18J\x10["id-0", "id-1"]x\xe8\x07\x80\x01\x01\x12%\n\ndelete_all\x18\x02 \x01(\x08\x42\x11\x92\x41\x0e:\x05\x66\x61lseJ\x05\x66\x61lse\x12+\n\tnamespace\x18\x03 \x01(\tB\x18\x92\x41\x15J\x13"example-namespace"\x12\'\n\x06\x66ilter\x18\x04 \x01(\x0b\x32\x17.google.protobuf.Struct"\x10\n\x0e\x44\x65leteResponse"i\n\x0c\x46\x65tchRequest\x12,\n\x03ids\x18\x01 \x03(\tB\x1f\x92\x41\x18J\x10["id-0", "id-1"]x\xe8\x07\x80\x01\x01\xe2\x41\x01\x02\x12+\n\tnamespace\x18\x02 \x01(\tB\x18\x92\x41\x15J\x13"example-namespace""\xe1\x01\n\rFetchResponse\x12,\n\x07vectors\x18\x01 \x03(\x0b\x32\x1b.FetchResponse.VectorsEntry\x12+\n\tnamespace\x18\x02 \x01(\tB\x18\x92\x41\x15J\x13"example-namespace"\x12\x32\n\x05usage\x18\x03 \x01(\x0b\x32\x06.UsageB\x16\x92\x41\x13J\x11{"read_units": 5}H\x00\x88\x01\x01\x1a\x37\n\x0cVectorsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x16\n\x05value\x18\x02 \x01(\x0b\x32\x07.Vector:\x02\x38\x01\x42\x08\n\x06_usage"\xf8\x01\n\x0bListRequest\x12,\n\x06prefix\x18\x01 \x01(\tB\x17\x92\x41\x14J\x0c"document1#"x\xe8\x07\x80\x01\x01H\x00\x88\x01\x01\x12 \n\x05limit\x18\x02 \x01(\rB\x0c\x92\x41\t:\x03\x31\x30\x30J\x02\x31\x32H\x01\x88\x01\x01\x12\x42\n\x10pagination_token\x18\x03 \x01(\tB#\x92\x41 J\x1e"Tm90aGluZyB0byBzZWUgaGVyZQo="H\x02\x88\x01\x01\x12+\n\tnamespace\x18\x04 \x01(\tB\x18\x92\x41\x15J\x13"example-namespace"B\t\n\x07_prefixB\x08\n\x06_limitB\x13\n\x11_pagination_token"?\n\nPagination\x12\x31\n\x04next\x18\x01 \x01(\tB#\x92\x41 J\x1e"Tm90aGluZyB0byBzZWUgaGVyZQo="",\n\x08ListItem\x12 \n\x02id\x18\x01 \x01(\tB\x14\x92\x41\x11J\x0f"document1#abb""\x83\x02\n\x0cListResponse\x12S\n\x07vectors\x18\x01 \x03(\x0b\x32\t.ListItemB7\x92\x41\x34J2[{"id": "document1#abb"}, {"id": "document1#abc"}]\x12$\n\npagination\x18\x02 \x01(\x0b\x32\x0b.PaginationH\x00\x88\x01\x01\x12+\n\tnamespace\x18\x03 \x01(\tB\x18\x92\x41\x15J\x13"example-namespace"\x12\x32\n\x05usage\x18\x04 \x01(\x0b\x32\x06.UsageB\x16\x92\x41\x13J\x11{"read_units": 1}H\x01\x88\x01\x01\x42\r\n\x0b_paginationB\x08\n\x06_usage"\xd1\x02\n\x0bQueryVector\x12H\n\x06values\x18\x01 \x03(\x02\x42\x38\x92\x41\x31J([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]x\xa0\x9c\x01\x80\x01\x01\xe2\x41\x01\x02\x12$\n\rsparse_values\x18\x05 \x01(\x0b\x32\r.SparseValues\x12(\n\x05top_k\x18\x02 \x01(\rB\x19\x92\x41\x16J\x02\x31\x30Y\x00\x00\x00\x00\x00\x88\xc3@i\x00\x00\x00\x00\x00\x00\xf0?\x12+\n\tnamespace\x18\x03 \x01(\tB\x18\x92\x41\x15J\x13"example-namespace"\x12{\n\x06\x66ilter\x18\x04 \x01(\x0b\x32\x17.google.protobuf.StructBR\x92\x41OJM{"genre": {"$in": ["comedy", "documentary", "drama"]}, "year": {"$eq": 2019}}"\xfb\x03\n\x0cQueryRequest\x12+\n\tnamespace\x18\x01 \x01(\tB\x18\x92\x41\x15J\x13"example-namespace"\x12,\n\x05top_k\x18\x02 \x01(\rB\x1d\x92\x41\x16J\x02\x31\x30Y\x00\x00\x00\x00\x00\x88\xc3@i\x00\x00\x00\x00\x00\x00\xf0?\xe2\x41\x01\x02\x12{\n\x06\x66ilter\x18\x03 \x01(\x0b\x32\x17.google.protobuf.StructBR\x92\x41OJM{"genre": {"$in": ["comedy", "documentary", "drama"]}, "year": {"$eq": 2019}}\x12(\n\x0einclude_values\x18\x04 \x01(\x08\x42\x10\x92\x41\r:\x05\x66\x61lseJ\x04true\x12*\n\x10include_metadata\x18\x05 \x01(\x08\x42\x10\x92\x41\r:\x05\x66\x61lseJ\x04true\x12)\n\x07queries\x18\x06 \x03(\x0b\x32\x0c.QueryVectorB\n\x18\x01\x92\x41\x05x\n\x80\x01\x01\x12\x44\n\x06vector\x18\x07 \x03(\x02\x42\x34\x92\x41\x31J([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]x\xa0\x9c\x01\x80\x01\x01\x12$\n\rsparse_vector\x18\t \x01(\x0b\x32\r.SparseValues\x12&\n\x02id\x18\x08 \x01(\tB\x1a\x92\x41\x17J\x12"example-vector-1"x\x80\x04"a\n\x12SingleQueryResults\x12\x1e\n\x07matches\x18\x01 \x03(\x0b\x32\r.ScoredVector\x12+\n\tnamespace\x18\x02 \x01(\tB\x18\x92\x41\x15J\x13"example-namespace""\xaa\x01\n\rQueryResponse\x12(\n\x07results\x18\x01 \x03(\x0b\x32\x13.SingleQueryResultsB\x02\x18\x01\x12\x1e\n\x07matches\x18\x02 \x03(\x0b\x32\r.ScoredVector\x12\x11\n\tnamespace\x18\x03 \x01(\t\x12\x32\n\x05usage\x18\x04 \x01(\x0b\x32\x06.UsageB\x16\x92\x41\x13J\x11{"read_units": 5}H\x00\x88\x01\x01\x42\x08\n\x06_usage"7\n\x05Usage\x12\x1f\n\nread_units\x18\x01 \x01(\rB\x06\x92\x41\x03J\x01\x35H\x00\x88\x01\x01\x42\r\n\x0b_read_units"\xb3\x02\n\rUpdateRequest\x12-\n\x02id\x18\x01 \x01(\tB!\x92\x41\x1aJ\x12"example-vector-1"x\x80\x04\x80\x01\x01\xe2\x41\x01\x02\x12\x44\n\x06values\x18\x02 \x03(\x02\x42\x34\x92\x41\x31J([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]x\xa0\x9c\x01\x80\x01\x01\x12$\n\rsparse_values\x18\x05 \x01(\x0b\x32\r.SparseValues\x12Z\n\x0cset_metadata\x18\x03 \x01(\x0b\x32\x17.google.protobuf.StructB+\x92\x41(J&{"genre": "documentary", "year": 2019}\x12+\n\tnamespace\x18\x04 \x01(\tB\x18\x92\x41\x15J\x13"example-namespace""\x10\n\x0eUpdateResponse"D\n\x19\x44\x65scribeIndexStatsRequest\x12\'\n\x06\x66ilter\x18\x01 \x01(\x0b\x32\x17.google.protobuf.Struct"4\n\x10NamespaceSummary\x12 \n\x0cvector_count\x18\x01 \x01(\rB\n\x92\x41\x07J\x05\x35\x30\x30\x30\x30"\x9a\x03\n\x1a\x44\x65scribeIndexStatsResponse\x12?\n\nnamespaces\x18\x01 \x03(\x0b\x32+.DescribeIndexStatsResponse.NamespacesEntry\x12\x1c\n\tdimension\x18\x02 \x01(\rB\t\x92\x41\x06J\x04\x31\x30\x32\x34\x12 \n\x0eindex_fullness\x18\x03 \x01(\x02\x42\x08\x92\x41\x05J\x03\x30.4\x12&\n\x12total_vector_count\x18\x04 \x01(\rB\n\x92\x41\x07J\x05\x38\x30\x30\x30\x30\x1a\x44\n\x0fNamespacesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.NamespaceSummary:\x02\x38\x01:\x8c\x01\x92\x41\x88\x01\x32\x85\x01{"namespaces": {"": {"vectorCount": 50000}, "example-namespace-2": {"vectorCount": 30000}}, "dimension": 1024, "index_fullness": 0.4}2\x95\x06\n\rVectorService\x12\x63\n\x06Upsert\x12\x0e.UpsertRequest\x1a\x0f.UpsertResponse"8\x92\x41\x1b\n\x11Vector Operations*\x06upsert\x82\xd3\xe4\x93\x02\x14"\x0f/vectors/upsert:\x01*\x12v\n\x06\x44\x65lete\x12\x0e.DeleteRequest\x1a\x0f.DeleteResponse"K\x92\x41\x1b\n\x11Vector Operations*\x06\x64\x65lete\x82\xd3\xe4\x93\x02\'"\x0f/vectors/delete:\x01*Z\x11*\x0f/vectors/delete\x12[\n\x05\x46\x65tch\x12\r.FetchRequest\x1a\x0e.FetchResponse"3\x92\x41\x1a\n\x11Vector Operations*\x05\x66\x65tch\x82\xd3\xe4\x93\x02\x10\x12\x0e/vectors/fetch\x12V\n\x04List\x12\x0c.ListRequest\x1a\r.ListResponse"1\x92\x41\x19\n\x11Vector Operations*\x04list\x82\xd3\xe4\x93\x02\x0f\x12\r/vectors/list\x12V\n\x05Query\x12\r.QueryRequest\x1a\x0e.QueryResponse".\x92\x41\x1a\n\x11Vector Operations*\x05query\x82\xd3\xe4\x93\x02\x0b"\x06/query:\x01*\x12\x63\n\x06Update\x12\x0e.UpdateRequest\x1a\x0f.UpdateResponse"8\x92\x41\x1b\n\x11Vector Operations*\x06update\x82\xd3\xe4\x93\x02\x14"\x0f/vectors/update:\x01*\x12\xb4\x01\n\x12\x44\x65scribeIndexStats\x12\x1a.DescribeIndexStatsRequest\x1a\x1b.DescribeIndexStatsResponse"e\x92\x41)\n\x11Vector Operations*\x14\x64\x65scribe_index_stats\x82\xd3\xe4\x93\x02\x33"\x15/describe_index_stats:\x01*Z\x17\x12\x15/describe_index_statsB\x8f\x03\n\x11io.pinecone.protoP\x01Z+github.com/pinecone-io/go-pinecone/pinecone\x92\x41\xc9\x02\x12K\n\x0cPinecone API";\n\x0fPinecone.io Ops\x12\x13https://pinecone.io\x1a\x13support@pinecone.io\x1a\x0c{index_host}*\x01\x02\x32\x10\x61pplication/json:\x10\x61pplication/jsonZx\nv\n\nApiKeyAuth\x12h\x08\x02\x12YAn API Key is required to call Pinecone APIs. Get yours at https://www.pinecone.io/start/\x1a\x07\x41pi-Key \x02\x62\x10\n\x0e\n\nApiKeyAuth\x12\x00r9\n\x19More Pinecone.io API docs\x12\x1chttps://www.pinecone.io/docsb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, "vector_service_pb2", _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
    _globals["DESCRIPTOR"]._options = None
    _globals["DESCRIPTOR"]._serialized_options = (
        b'\n\021io.pinecone.protoP\001Z+github.com/pinecone-io/go-pinecone/pinecone\222A\311\002\022K\n\014Pinecone API";\n\017Pinecone.io Ops\022\023https://pinecone.io\032\023support@pinecone.io\032\014{index_host}*\001\0022\020application/json:\020application/jsonZx\nv\n\nApiKeyAuth\022h\010\002\022YAn API Key is required to call Pinecone APIs. Get yours at https://www.pinecone.io/start/\032\007Api-Key \002b\020\n\016\n\nApiKeyAuth\022\000r9\n\031More Pinecone.io API docs\022\034https://www.pinecone.io/docs'
    )
    _globals["_SPARSEVALUES"].fields_by_name["indices"]._options = None
    _globals["_SPARSEVALUES"].fields_by_name[
        "indices"
    ]._serialized_options = b"\222A\036J\026[1, 312, 822, 14, 980]x\350\007\200\001\001\342A\001\002"
    _globals["_SPARSEVALUES"].fields_by_name["values"]._options = None
    _globals["_SPARSEVALUES"].fields_by_name[
        "values"
    ]._serialized_options = b"\222A!J\031[0.1, 0.2, 0.3, 0.4, 0.5]x\350\007\200\001\001\342A\001\002"
    _globals["_VECTOR"].fields_by_name["id"]._options = None
    _globals["_VECTOR"].fields_by_name[
        "id"
    ]._serialized_options = b'\222A\032J\022"example-vector-1"x\200\004\200\001\001\342A\001\002'
    _globals["_VECTOR"].fields_by_name["values"]._options = None
    _globals["_VECTOR"].fields_by_name[
        "values"
    ]._serialized_options = b"\222A1J([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]x\240\234\001\200\001\001\342A\001\002"
    _globals["_VECTOR"].fields_by_name["metadata"]._options = None
    _globals["_VECTOR"].fields_by_name[
        "metadata"
    ]._serialized_options = b'\222A(J&{"genre": "documentary", "year": 2019}'
    _globals["_SCOREDVECTOR"].fields_by_name["id"]._options = None
    _globals["_SCOREDVECTOR"].fields_by_name[
        "id"
    ]._serialized_options = b'\222A\032J\022"example-vector-1"x\200\004\200\001\001\342A\001\002'
    _globals["_SCOREDVECTOR"].fields_by_name["score"]._options = None
    _globals["_SCOREDVECTOR"].fields_by_name["score"]._serialized_options = b"\222A\006J\0040.08"
    _globals["_SCOREDVECTOR"].fields_by_name["values"]._options = None
    _globals["_SCOREDVECTOR"].fields_by_name[
        "values"
    ]._serialized_options = b"\222A*J([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]"
    _globals["_SCOREDVECTOR"].fields_by_name["metadata"]._options = None
    _globals["_SCOREDVECTOR"].fields_by_name[
        "metadata"
    ]._serialized_options = b'\222A(J&{"genre": "documentary", "year": 2019}'
    _globals["_UPSERTREQUEST"].fields_by_name["vectors"]._options = None
    _globals["_UPSERTREQUEST"].fields_by_name[
        "vectors"
    ]._serialized_options = b"\222A\006x\350\007\200\001\001\342A\001\002"
    _globals["_UPSERTREQUEST"].fields_by_name["namespace"]._options = None
    _globals["_UPSERTREQUEST"].fields_by_name["namespace"]._serialized_options = b'\222A\025J\023"example-namespace"'
    _globals["_UPSERTRESPONSE"].fields_by_name["upserted_count"]._options = None
    _globals["_UPSERTRESPONSE"].fields_by_name["upserted_count"]._serialized_options = b"\222A\004J\00210"
    _globals["_DELETEREQUEST"].fields_by_name["ids"]._options = None
    _globals["_DELETEREQUEST"].fields_by_name[
        "ids"
    ]._serialized_options = b'\222A\030J\020["id-0", "id-1"]x\350\007\200\001\001'
    _globals["_DELETEREQUEST"].fields_by_name["delete_all"]._options = None
    _globals["_DELETEREQUEST"].fields_by_name["delete_all"]._serialized_options = b"\222A\016:\005falseJ\005false"
    _globals["_DELETEREQUEST"].fields_by_name["namespace"]._options = None
    _globals["_DELETEREQUEST"].fields_by_name["namespace"]._serialized_options = b'\222A\025J\023"example-namespace"'
    _globals["_FETCHREQUEST"].fields_by_name["ids"]._options = None
    _globals["_FETCHREQUEST"].fields_by_name[
        "ids"
    ]._serialized_options = b'\222A\030J\020["id-0", "id-1"]x\350\007\200\001\001\342A\001\002'
    _globals["_FETCHREQUEST"].fields_by_name["namespace"]._options = None
    _globals["_FETCHREQUEST"].fields_by_name["namespace"]._serialized_options = b'\222A\025J\023"example-namespace"'
    _globals["_FETCHRESPONSE_VECTORSENTRY"]._options = None
    _globals["_FETCHRESPONSE_VECTORSENTRY"]._serialized_options = b"8\001"
    _globals["_FETCHRESPONSE"].fields_by_name["namespace"]._options = None
    _globals["_FETCHRESPONSE"].fields_by_name["namespace"]._serialized_options = b'\222A\025J\023"example-namespace"'
    _globals["_FETCHRESPONSE"].fields_by_name["usage"]._options = None
    _globals["_FETCHRESPONSE"].fields_by_name["usage"]._serialized_options = b'\222A\023J\021{"read_units": 5}'
    _globals["_LISTREQUEST"].fields_by_name["prefix"]._options = None
    _globals["_LISTREQUEST"].fields_by_name[
        "prefix"
    ]._serialized_options = b'\222A\024J\014"document1#"x\350\007\200\001\001'
    _globals["_LISTREQUEST"].fields_by_name["limit"]._options = None
    _globals["_LISTREQUEST"].fields_by_name["limit"]._serialized_options = b"\222A\t:\003100J\00212"
    _globals["_LISTREQUEST"].fields_by_name["pagination_token"]._options = None
    _globals["_LISTREQUEST"].fields_by_name[
        "pagination_token"
    ]._serialized_options = b'\222A J\036"Tm90aGluZyB0byBzZWUgaGVyZQo="'
    _globals["_LISTREQUEST"].fields_by_name["namespace"]._options = None
    _globals["_LISTREQUEST"].fields_by_name["namespace"]._serialized_options = b'\222A\025J\023"example-namespace"'
    _globals["_PAGINATION"].fields_by_name["next"]._options = None
    _globals["_PAGINATION"].fields_by_name["next"]._serialized_options = b'\222A J\036"Tm90aGluZyB0byBzZWUgaGVyZQo="'
    _globals["_LISTITEM"].fields_by_name["id"]._options = None
    _globals["_LISTITEM"].fields_by_name["id"]._serialized_options = b'\222A\021J\017"document1#abb"'
    _globals["_LISTRESPONSE"].fields_by_name["vectors"]._options = None
    _globals["_LISTRESPONSE"].fields_by_name[
        "vectors"
    ]._serialized_options = b'\222A4J2[{"id": "document1#abb"}, {"id": "document1#abc"}]'
    _globals["_LISTRESPONSE"].fields_by_name["namespace"]._options = None
    _globals["_LISTRESPONSE"].fields_by_name["namespace"]._serialized_options = b'\222A\025J\023"example-namespace"'
    _globals["_LISTRESPONSE"].fields_by_name["usage"]._options = None
    _globals["_LISTRESPONSE"].fields_by_name["usage"]._serialized_options = b'\222A\023J\021{"read_units": 1}'
    _globals["_QUERYVECTOR"].fields_by_name["values"]._options = None
    _globals["_QUERYVECTOR"].fields_by_name[
        "values"
    ]._serialized_options = b"\222A1J([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]x\240\234\001\200\001\001\342A\001\002"
    _globals["_QUERYVECTOR"].fields_by_name["top_k"]._options = None
    _globals["_QUERYVECTOR"].fields_by_name[
        "top_k"
    ]._serialized_options = b"\222A\026J\00210Y\000\000\000\000\000\210\303@i\000\000\000\000\000\000\360?"
    _globals["_QUERYVECTOR"].fields_by_name["namespace"]._options = None
    _globals["_QUERYVECTOR"].fields_by_name["namespace"]._serialized_options = b'\222A\025J\023"example-namespace"'
    _globals["_QUERYVECTOR"].fields_by_name["filter"]._options = None
    _globals["_QUERYVECTOR"].fields_by_name[
        "filter"
    ]._serialized_options = b'\222AOJM{"genre": {"$in": ["comedy", "documentary", "drama"]}, "year": {"$eq": 2019}}'
    _globals["_QUERYREQUEST"].fields_by_name["namespace"]._options = None
    _globals["_QUERYREQUEST"].fields_by_name["namespace"]._serialized_options = b'\222A\025J\023"example-namespace"'
    _globals["_QUERYREQUEST"].fields_by_name["top_k"]._options = None
    _globals["_QUERYREQUEST"].fields_by_name[
        "top_k"
    ]._serialized_options = b"\222A\026J\00210Y\000\000\000\000\000\210\303@i\000\000\000\000\000\000\360?\342A\001\002"
    _globals["_QUERYREQUEST"].fields_by_name["filter"]._options = None
    _globals["_QUERYREQUEST"].fields_by_name[
        "filter"
    ]._serialized_options = b'\222AOJM{"genre": {"$in": ["comedy", "documentary", "drama"]}, "year": {"$eq": 2019}}'
    _globals["_QUERYREQUEST"].fields_by_name["include_values"]._options = None
    _globals["_QUERYREQUEST"].fields_by_name["include_values"]._serialized_options = b"\222A\r:\005falseJ\004true"
    _globals["_QUERYREQUEST"].fields_by_name["include_metadata"]._options = None
    _globals["_QUERYREQUEST"].fields_by_name["include_metadata"]._serialized_options = b"\222A\r:\005falseJ\004true"
    _globals["_QUERYREQUEST"].fields_by_name["queries"]._options = None
    _globals["_QUERYREQUEST"].fields_by_name["queries"]._serialized_options = b"\030\001\222A\005x\n\200\001\001"
    _globals["_QUERYREQUEST"].fields_by_name["vector"]._options = None
    _globals["_QUERYREQUEST"].fields_by_name[
        "vector"
    ]._serialized_options = b"\222A1J([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]x\240\234\001\200\001\001"
    _globals["_QUERYREQUEST"].fields_by_name["id"]._options = None
    _globals["_QUERYREQUEST"].fields_by_name["id"]._serialized_options = b'\222A\027J\022"example-vector-1"x\200\004'
    _globals["_SINGLEQUERYRESULTS"].fields_by_name["namespace"]._options = None
    _globals["_SINGLEQUERYRESULTS"].fields_by_name[
        "namespace"
    ]._serialized_options = b'\222A\025J\023"example-namespace"'
    _globals["_QUERYRESPONSE"].fields_by_name["results"]._options = None
    _globals["_QUERYRESPONSE"].fields_by_name["results"]._serialized_options = b"\030\001"
    _globals["_QUERYRESPONSE"].fields_by_name["usage"]._options = None
    _globals["_QUERYRESPONSE"].fields_by_name["usage"]._serialized_options = b'\222A\023J\021{"read_units": 5}'
    _globals["_USAGE"].fields_by_name["read_units"]._options = None
    _globals["_USAGE"].fields_by_name["read_units"]._serialized_options = b"\222A\003J\0015"
    _globals["_UPDATEREQUEST"].fields_by_name["id"]._options = None
    _globals["_UPDATEREQUEST"].fields_by_name[
        "id"
    ]._serialized_options = b'\222A\032J\022"example-vector-1"x\200\004\200\001\001\342A\001\002'
    _globals["_UPDATEREQUEST"].fields_by_name["values"]._options = None
    _globals["_UPDATEREQUEST"].fields_by_name[
        "values"
    ]._serialized_options = b"\222A1J([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]x\240\234\001\200\001\001"
    _globals["_UPDATEREQUEST"].fields_by_name["set_metadata"]._options = None
    _globals["_UPDATEREQUEST"].fields_by_name[
        "set_metadata"
    ]._serialized_options = b'\222A(J&{"genre": "documentary", "year": 2019}'
    _globals["_UPDATEREQUEST"].fields_by_name["namespace"]._options = None
    _globals["_UPDATEREQUEST"].fields_by_name["namespace"]._serialized_options = b'\222A\025J\023"example-namespace"'
    _globals["_NAMESPACESUMMARY"].fields_by_name["vector_count"]._options = None
    _globals["_NAMESPACESUMMARY"].fields_by_name["vector_count"]._serialized_options = b"\222A\007J\00550000"
    _globals["_DESCRIBEINDEXSTATSRESPONSE_NAMESPACESENTRY"]._options = None
    _globals["_DESCRIBEINDEXSTATSRESPONSE_NAMESPACESENTRY"]._serialized_options = b"8\001"
    _globals["_DESCRIBEINDEXSTATSRESPONSE"].fields_by_name["dimension"]._options = None
    _globals["_DESCRIBEINDEXSTATSRESPONSE"].fields_by_name["dimension"]._serialized_options = b"\222A\006J\0041024"
    _globals["_DESCRIBEINDEXSTATSRESPONSE"].fields_by_name["index_fullness"]._options = None
    _globals["_DESCRIBEINDEXSTATSRESPONSE"].fields_by_name["index_fullness"]._serialized_options = b"\222A\005J\0030.4"
    _globals["_DESCRIBEINDEXSTATSRESPONSE"].fields_by_name["total_vector_count"]._options = None
    _globals["_DESCRIBEINDEXSTATSRESPONSE"].fields_by_name[
        "total_vector_count"
    ]._serialized_options = b"\222A\007J\00580000"
    _globals["_DESCRIBEINDEXSTATSRESPONSE"]._options = None
    _globals["_DESCRIBEINDEXSTATSRESPONSE"]._serialized_options = (
        b'\222A\210\0012\205\001{"namespaces": {"": {"vectorCount": 50000}, "example-namespace-2": {"vectorCount": 30000}}, "dimension": 1024, "index_fullness": 0.4}'
    )
    _globals["_VECTORSERVICE"].methods_by_name["Upsert"]._options = None
    _globals["_VECTORSERVICE"].methods_by_name[
        "Upsert"
    ]._serialized_options = (
        b'\222A\033\n\021Vector Operations*\006upsert\202\323\344\223\002\024"\017/vectors/upsert:\001*'
    )
    _globals["_VECTORSERVICE"].methods_by_name["Delete"]._options = None
    _globals["_VECTORSERVICE"].methods_by_name[
        "Delete"
    ]._serialized_options = b"\222A\033\n\021Vector Operations*\006delete\202\323\344\223\002'\"\017/vectors/delete:\001*Z\021*\017/vectors/delete"
    _globals["_VECTORSERVICE"].methods_by_name["Fetch"]._options = None
    _globals["_VECTORSERVICE"].methods_by_name[
        "Fetch"
    ]._serialized_options = b"\222A\032\n\021Vector Operations*\005fetch\202\323\344\223\002\020\022\016/vectors/fetch"
    _globals["_VECTORSERVICE"].methods_by_name["List"]._options = None
    _globals["_VECTORSERVICE"].methods_by_name[
        "List"
    ]._serialized_options = b"\222A\031\n\021Vector Operations*\004list\202\323\344\223\002\017\022\r/vectors/list"
    _globals["_VECTORSERVICE"].methods_by_name["Query"]._options = None
    _globals["_VECTORSERVICE"].methods_by_name[
        "Query"
    ]._serialized_options = b'\222A\032\n\021Vector Operations*\005query\202\323\344\223\002\013"\006/query:\001*'
    _globals["_VECTORSERVICE"].methods_by_name["Update"]._options = None
    _globals["_VECTORSERVICE"].methods_by_name[
        "Update"
    ]._serialized_options = (
        b'\222A\033\n\021Vector Operations*\006update\202\323\344\223\002\024"\017/vectors/update:\001*'
    )
    _globals["_VECTORSERVICE"].methods_by_name["DescribeIndexStats"]._options = None
    _globals["_VECTORSERVICE"].methods_by_name[
        "DescribeIndexStats"
    ]._serialized_options = b'\222A)\n\021Vector Operations*\024describe_index_stats\202\323\344\223\0023"\025/describe_index_stats:\001*Z\027\022\025/describe_index_stats'
    _globals["_SPARSEVALUES"]._serialized_start = 166
    _globals["_SPARSEVALUES"]._serialized_end = 294
    _globals["_VECTOR"]._serialized_start = 297
    _globals["_VECTOR"]._serialized_end = 552
    _globals["_SCOREDVECTOR"]._serialized_start = 555
    _globals["_SCOREDVECTOR"]._serialized_end = 831
    _globals["_REQUESTUNION"]._serialized_start = 834
    _globals["_REQUESTUNION"]._serialized_end = 971
    _globals["_UPSERTREQUEST"]._serialized_start = 973
    _globals["_UPSERTREQUEST"]._serialized_end = 1074
    _globals["_UPSERTRESPONSE"]._serialized_start = 1076
    _globals["_UPSERTRESPONSE"]._serialized_end = 1125
    _globals["_DELETEREQUEST"]._serialized_start = 1128
    _globals["_DELETEREQUEST"]._serialized_end = 1310
    _globals["_DELETERESPONSE"]._serialized_start = 1312
    _globals["_DELETERESPONSE"]._serialized_end = 1328
    _globals["_FETCHREQUEST"]._serialized_start = 1330
    _globals["_FETCHREQUEST"]._serialized_end = 1435
    _globals["_FETCHRESPONSE"]._serialized_start = 1438
    _globals["_FETCHRESPONSE"]._serialized_end = 1663
    _globals["_FETCHRESPONSE_VECTORSENTRY"]._serialized_start = 1598
    _globals["_FETCHRESPONSE_VECTORSENTRY"]._serialized_end = 1653
    _globals["_LISTREQUEST"]._serialized_start = 1666
    _globals["_LISTREQUEST"]._serialized_end = 1914
    _globals["_PAGINATION"]._serialized_start = 1916
    _globals["_PAGINATION"]._serialized_end = 1979
    _globals["_LISTITEM"]._serialized_start = 1981
    _globals["_LISTITEM"]._serialized_end = 2025
    _globals["_LISTRESPONSE"]._serialized_start = 2028
    _globals["_LISTRESPONSE"]._serialized_end = 2287
    _globals["_QUERYVECTOR"]._serialized_start = 2290
    _globals["_QUERYVECTOR"]._serialized_end = 2627
    _globals["_QUERYREQUEST"]._serialized_start = 2630
    _globals["_QUERYREQUEST"]._serialized_end = 3137
    _globals["_SINGLEQUERYRESULTS"]._serialized_start = 3139
    _globals["_SINGLEQUERYRESULTS"]._serialized_end = 3236
    _globals["_QUERYRESPONSE"]._serialized_start = 3239
    _globals["_QUERYRESPONSE"]._serialized_end = 3409
    _globals["_USAGE"]._serialized_start = 3411
    _globals["_USAGE"]._serialized_end = 3466
    _globals["_UPDATEREQUEST"]._serialized_start = 3469
    _globals["_UPDATEREQUEST"]._serialized_end = 3776
    _globals["_UPDATERESPONSE"]._serialized_start = 3778
    _globals["_UPDATERESPONSE"]._serialized_end = 3794
    _globals["_DESCRIBEINDEXSTATSREQUEST"]._serialized_start = 3796
    _globals["_DESCRIBEINDEXSTATSREQUEST"]._serialized_end = 3864
    _globals["_NAMESPACESUMMARY"]._serialized_start = 3866
    _globals["_NAMESPACESUMMARY"]._serialized_end = 3918
    _globals["_DESCRIBEINDEXSTATSRESPONSE"]._serialized_start = 3921
    _globals["_DESCRIBEINDEXSTATSRESPONSE"]._serialized_end = 4331
    _globals["_DESCRIBEINDEXSTATSRESPONSE_NAMESPACESENTRY"]._serialized_start = 4120
    _globals["_DESCRIBEINDEXSTATSRESPONSE_NAMESPACESENTRY"]._serialized_end = 4188
    _globals["_VECTORSERVICE"]._serialized_start = 4334
    _globals["_VECTORSERVICE"]._serialized_end = 5123
# @@protoc_insertion_point(module_scope)
