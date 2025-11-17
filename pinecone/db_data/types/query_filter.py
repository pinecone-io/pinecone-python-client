from typing import Literal, Dict, List

FieldValue = str | int | float | bool

ExactMatchFilter = Dict[str, FieldValue]

EqFilter = Dict[Literal["$eq"], FieldValue]
NeFilter = Dict[Literal["$ne"], FieldValue]

NumericFieldValue = int | float
GtFilter = Dict[Literal["$gt"], NumericFieldValue]
GteFilter = Dict[Literal["$gte"], NumericFieldValue]
LtFilter = Dict[Literal["$lt"], NumericFieldValue]
LteFilter = Dict[Literal["$lte"], NumericFieldValue]

InFilter = Dict[Literal["$in"], List[FieldValue]]
NinFilter = Dict[Literal["$nin"], List[FieldValue]]
ExistsFilter = Dict[Literal["$exists"], bool]

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
AndFilter = Dict[Literal["$and"], List[SimpleFilter]]
OrFilter = Dict[Literal["$or"], List[SimpleFilter]]

FilterTypedDict = SimpleFilter | AndFilter | OrFilter
