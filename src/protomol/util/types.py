from typing import Annotated

import numpy
from pydantic import BeforeValidator, GetPydanticSchema, PlainSerializer, SkipValidation
from pydantic_core import core_schema


# Source: https://github.com/coltonbh/qcio/blob/master/qcio/helper_types.py
NDArray = Annotated[
    SkipValidation[numpy.ndarray],
    BeforeValidator(lambda x: numpy.array(x, dtype=numpy.float64)),
    PlainSerializer(lambda x: numpy.array(x).tolist()),
    GetPydanticSchema(
        lambda _, handler: core_schema.with_default_schema(handler(list[float]))
    ),
]
