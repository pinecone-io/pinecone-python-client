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
)
AndFilter = dict[Literal["$and"], list[SimpleFilter]]
OrFilter = dict[Literal["$or"], list[SimpleFilter]]

FilterTypedDict = SimpleFilter | AndFilter | OrFilter
