import oddments as odd
import clockwork as cw
import polars as pl


class FolderContents(object):
    '''
    Description
    --------------------
    Class that enables operations on all contents of a folder collectively,
    including obtaining metadata, filtering, deleting, and more. This class
    encompasses both files and folders but they can be accessed separetly via
    the Files and Folders subclasses, respectively.

    Note: This class and its subclasses are intended to be accessed indirectly
    by using the 'contents', 'folders', and 'files' Folder cached properties.

    Class Attributes
    --------------------
    None

    Instance Attributes
    --------------------
    folder : Folder
        Folder object in which the contents reside.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, folder):
        self.folder = folder


    #╭-------------------------------------------------------------------------╮
    #| Properties                                                              |
    #╰-------------------------------------------------------------------------╯

    @property
    def meta_data(self):

        schema = {
            'label': pl.Categorical,
            'type': pl.Categorical,
            'hash_value': pl.String,
            'path': pl.String,
            'directory': pl.String,
            'full_name': pl.String,
            'name': pl.String,
            'read_only': pl.Boolean,
            'exists': pl.Boolean,
            'empty': pl.Boolean,
            'file_count': pl.Int64,
            'folder_count': pl.Int64,
            'created_date': pl.Datetime,
            'modified_date': pl.Datetime,
            'extension': pl.Categorical,
            'size': pl.Int64,
            'size_label': pl.String,
            }

        data = []
        for obj in self:
            meta_data = obj.meta_data
            data.append({
                k: meta_data.get(k)
                for k in schema
                })

        df = pl.DataFrame(data=data, schema=schema)
        df = df.sort(by=['label','full_name'])

        return df


    @property
    def count(self):
        ''' the number of objects that currently exist in the folder '''
        return len(self.to_list())


    @property
    def exists(self):
        ''' True if any such objects exist '''
        return self.count > 0


    @property
    def read_only(self):
        return self.folder.read_only


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def to_list(self):
        ''' sorted list of all the objects in the folder '''
        if not self.folder.exists:
            return []

        return self.folder.sort(self._to_list())


    def to_dict(self):
        ''' dictionary where keys are object hash values and values are
            the objects themselves '''
        return {obj.hash_value: obj for obj in self}


    def delete(self):
        ''' delete folder contents '''
        self.folder.clear()


    def filter(
        self,
        name_pattern=None,
        name_literal=True,
        date_pattern=None,
        date_format=None,
        extension=None,
        index=None,
        sort_by=None,
        descending=True,
        errors='raise',
        ):
        '''
        Description
        ------------
        Filter folder objects based on user-defined criteria.

        Parameters
        ------------
        name_pattern : str
            If not None, only object names meeting this pattern will be
            considered candidates.
        name_literal
            Determines how 'name_pattern' is interpreted:
                • True → literal string
                • False → regular expression
        date_pattern : str
            If not None, only those object names containing a timestamp
            matching this regular expression pattern will be considered
            candidates.

            Example:
            ------------
            Consider a folder that contains multiple files that include a
            YYYY-MM-DD timestamp in the name such as '2024-06-23 Budget.xlsx'.
            In this case you might pass date_pattern=r'\d{4}\-\d{2}\-\d{2}'

        date_format : str
            Defines the object names' timestamp format (e.g. '%Y-%m-%d').
            There are a few scenarios to consider:
                • 'date_pattern' is None → an attempt will be made to derive
                    the regex pattern using 'date_format' as the template.
                • 'date_pattern' is not None → 'date_format' is optional.

            Example:
            ------------
            Dealing with the same example from above, all the following
            combinations will achieve the same result since the None values
            being inferred:
                • date_pattern=r'\d{4}\-\d{2}\-\d{2}', date_format='%Y-%m-%d'
                • date_pattern=None, date_format='%Y-%m-%d'
                • date_pattern=r'\d{4}\-\d{2}\-\d{2}', date_format=None

        extension : str
            If not None, only those file objects with this file extension
            will be considered candidates. This argument may be passed with
            or without a period and is not case sensitive (e.g. '.txt',
            'txt', '.TXT' are all valid).
        index : int
            If not None, the object at this integer index in the filtered
            and sorted data is returned.
        sort_by : str
            Name of self.meta_data column to sort by. This argument must
            be None if the 'date_pattern' or 'date_format' arguments are
            not None. Defaults to the object created date if None.
        descending : bool
            If True, data is sorted in descending order, otherwise,
            ascending order. If the 'date_pattern' or 'date_format'
            arguments are not None, the sort will occur on the timestamps
            extracted from the object names, otherwise, the column denoted
            by the 'sort_by' argument is used.
        errors : bool
            determines how exceptions are handled:
                • 'raise' → exception is raised
                • 'ignore' → None is returned

        Returns
        ------------
        out : pl.DataFrame | pd.DataFrame | Folder | File | None
            If 'index' argument is not None, the corresponding file or
            folder object is returned. Otherwise, the filtered and sorted
            meta data DataFrame is returned. If an error is encountered and
            errors='ignore', None is returned.
        '''

        def format_date_pattern(x):
            start = x.startswith('(')
            end = x.endswith(')')

            if start != end:
                raise ValueError(
                    "'date_pattern' is malformed: "
                    f"{date_pattern!r}"
                    )

            return x if start and end else f'({x})'


        odd.validate_value(
            value=index,
            name='index',
            types=int,
            none_ok=True
            )

        odd.validate_value(
            value=sort_by,
            name='sort_by',
            types=(str, list),
            none_ok=True,
            empty_ok=False
            )

        odd.validate_value(
            value=errors,
            name='errors',
            types=str,
            whitelist=['raise','ignore']
            )

        if sort_by is None:
            sort_by = ['created_date']
        else:
            if date_format or date_pattern:
                raise ValueError(
                    "Cannot pass 'sort_by' argument if either "
                    "'date_format' or 'date_pattern' arguments "
                    "are not None."
                    )

        if isinstance(sort_by, str):
            sort_by = [sort_by]

        kind = self.__class__.__name__.lower()

        if not self.exists:
            if errors == 'ignore':
                return None
            raise ValueError(
                f'No {kind} exist to filter.'
                )

        df = self.meta_data.sort(
            by=sort_by,
            descending=descending
            )

        if extension is not None:
            if kind == 'folders':
                raise ValueError(
                    "Cannot pass 'extension' argument when "
                    "accessing 'folders' property."
                    )
            extension = extension.replace('.', '').lower()

            df = df.filter((
                pl.col('extension') == extension
                ))

            if df.is_empty():
                if errors == 'ignore':
                    return None
                raise ValueError(
                    f"No files with extension '.{extension}' found."
                    )

        if name_pattern is not None:
            df = df.filter((
                pl.col('name').str.contains(
                    pattern=name_pattern,
                    literal=name_literal,
                    )
                ))

            if df.is_empty():
                if errors == 'ignore':
                    return None
                message = [
                    f'No {kind} matching name pattern {name_pattern!r}'
                    ]
                if extension is not None:
                    message.append(f"with extension '.{extension}'")
                raise ValueError(
                    ' '.join(message + ['found.'])
                    )

        if date_format is not None and date_pattern is None:
            date_pattern = cw.temporal_format_to_regex(date_format)

        if date_pattern is not None:
            date_pattern = format_date_pattern(date_pattern)
            date_alias = 'name_timestamp'

            df = (
                df
                .lazy()
                .with_columns(
                    pl.col('name')
                    .str.extract(
                        pattern=date_pattern,
                        group_index=1
                        )
                    .str.to_datetime(
                        format=date_format
                        )
                    .alias(date_alias)
                    )
                .drop_nulls(
                    subset=date_alias
                    )
                .sort(
                    by=[date_alias, *sort_by],
                    descending=descending
                    )
                .collect()
                )

            if df.is_empty():
                if errors == 'ignore':
                    return None
                raise ValueError(
                    'Timestamp extraction failed for pattern: '
                    f'{date_pattern!r}'
                    )

        if index is None:
            return df

        try:
            hash_value = df['hash_value'][index]
            return self.to_dict()[hash_value]
        except Exception as error:
            if errors == 'ignore':
                return None
            raise error


    def _to_list(self):
        return [
            *self.folder.folders.to_list(),
            *self.folder.files.to_list()
            ]


    #╭-------------------------------------------------------------------------╮
    #| Magic Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    def __repr__(self):
        return '\n'.join(
            obj.path.replace('/','\\')
            for obj in self
            )


    def __bool__(self):
        return self.count > 0


    def __iter__(self):
        for x in self.to_list():
            yield x


    def __len__(self):
        return self.count


    def __getitem__(self, key):
        ''' implements slicing & indexing '''
        objs = self.to_list()

        if isinstance(key, (int, slice)):
            return objs[key]
        elif isinstance(key, str):
            for obj in objs:
                if key in {
                    obj.name,
                    obj.full_name,
                    obj.path
                    }:
                    return obj
            raise KeyError(key)
        else:
            raise TypeError(
                f'Invalid argument type: <{type(key).__name__}>'
                )


    def __contains__(self, key):
        for obj in self.to_list():
            if key in {
                obj.name,
                obj.full_name,
                obj.path
                }:
                return True

        return False