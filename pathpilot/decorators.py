import numpy as np

from .exceptions import ReadOnlyError



def check_read_only(func):

    def wrapper(self, *args, **kwargs):
        if self.read_only: raise ReadOnlyError
        return func(self, *args, **kwargs)

    return wrapper



def purge_whitespace(func):
    ''' wrapper function that purges unwanted whitespace from a DataFrame '''

    def wrapper(*args, **kwargs):

        def strip_or_skip(x):
            try:
                x = x.strip()
                if x == '': return np.nan
            except:
                pass
            return x

        df = func(*args, **kwargs)

        # clean column names by trimming leading/trailing whitespace and removing new lines and consecutive spaces
        df.rename(columns={k: ' '.join(k.split()) for k in df.columns if isinstance(k, str)}, inplace=True)

        # replace None with np.nan
        for k in df.columns:
            try:
                df[k] = df[k].fillna(np.nan)
            except:
                pass

        # trim leading/trailing whitespace and replace whitespace-only values with NaN
        for k in df.select_dtypes(include=['object']).columns:
            # df[k] = df[k].replace(to_replace=r'^\s*$', value=np.nan, regex=True)

            # using the vectorized string method str.strip() is faster but object-type columns can have mixed data types
            df[k] = df[k].apply(strip_or_skip) #.str.strip()

        # df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        return df

    return wrapper