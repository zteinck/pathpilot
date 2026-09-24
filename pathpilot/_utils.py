from oddments import Validator


def _validate_df_backend(value):
    (
    Validator(
        types=str,
        whitelist=['pandas','polars'],
        )
    .validate(value, 'df_backend')
    )