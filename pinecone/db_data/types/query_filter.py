from typing import Literal

FieldValue = str | int | float | bool

ExactMatchFilter = dict[str, FieldValue]

EqFilter = dict[Literal["$eq"], FieldValue]
NeFilter = dict[Literal["$ne"], FieldValue]

NumericFieldValue = int | float
GtFilter = dict[Literal["$gt"], NumericFieldValue]
GteFilter = dict[Literal["$gte"], NumericFieldValue]
LtFilter = dict[Literal["$lt"], NumericFieldValue]
LteFilter = dict[Literal["$lte"], NumericFieldValue]

InFilter = dict[Literal["$in"], list[FieldValue]]
NinFilter = dict[Literal["$nin"], list[FieldValue]]
ExistsFilter = dict[Literal["$exists"], bool]

# Operator-only filters (e.g., {"$eq": "value"})
OperatorFilter = (
    EqFilter
    | NeFilter
    | GtFilter
    | GteFilter
    | LtFilter
    | LteFilter
    | InFilter
    | NinFilter
    | ExistsFilter
)

# Field-level filters that can use operators or exact match (e.g., {"field": {"$eq": "value"}})
FieldFilter = dict[str, OperatorFilter | FieldValue]

SimpleFilter = (
    ExactMatchFilter
    | EqFilter
    | NeFilter
    | GtFilter
    | GteFilter
    | LtFilter
    | LteFilter
    | InFilter
    | NinFilter
    | ExistsFilter
    | FieldFilter
)
AndFilter = dict[Literal["$and"], list[SimpleFilter]]
OrFilter = dict[Literal["$or"], list[SimpleFilter]]

FilterTypedDict = SimpleFilter | AndFilter | OrFilter
