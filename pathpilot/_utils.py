import oddments as odd


def _validate_df_backend(value):

    odd.validate_value(
        value=value,
        name='df_backend',
        types=str,
        whitelist=['pandas','polars'],
        )