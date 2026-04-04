from oddments.frame._constants import (
    PANDAS_TYPES,
    POLARS_TYPES,
    )


def _is_frame(obj):
    return isinstance(obj, PANDAS_TYPES + POLARS_TYPES)