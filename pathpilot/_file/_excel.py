from collections import OrderedDict
from functools import cached_property
from copy import deepcopy
import datetime
import sys
import re
import pandas as pd
import numpy as np
import oddments as odd
from cachegrab import sha256

from .base import FileBase
from .utils import get_size_label


class ExcelFile(FileBase):
    '''
    Description
    --------------------
    Excel file object

    Class Attributes
    --------------------
    See parent class documentation.

    Instance Attributes
    --------------------
    writer : ExcelWriter
        pd.ExcelWriter instance
    active_sheet : xlsxwriter.worksheet.Worksheet
        The active worksheet
    sheet_cache : dict
        Dictionary where the keys are worksheet names and values are the
        worksheet objects. It is important to note that the keys are the
        original sheet names passed by the user as opposed to the actual name
        that appears in the excel file and worksheet.name attribute because the
        latter is truncated by the 31 character limit and/or will have a page
        prefix added per the 'number_tabs' argument. This setup allows the user
        to access the sheet using the original name.
    formats : dict
        Keys are strings representing names of formats and values are
        dictionaries containing format parameters. For Example, {{'bold':
        {'bold': True}}. These are intended to be frequently used formats that
        can be both used individually and as building blocks for more complex
        formats (see self.add_format for more information.)
    format_cache : dict
        keys are hashes of sorted dictionaries containing format parameters
        and values are xlsxwriter.format.Format. This is done so that only one
        format instance will be created per unique format used.
    number_tabs : bool
        if True, tabs in the workbook will have a number prefix. For example,
        'MyTab' would be appear as '1 - MyTab' in the workbook.
    page : int
        keeps track of the current tab number
    verbose : bool
        if True, information is printed when a worksheet is created or written
        to.

    Note:
    --------------------
    This URL contains instructions on how to get LibreOffice to calculate
    formulas (0 by default):
    https://stackoverflow.com/questions/32205927/xlsxwriter-and-libreoffice-not-showing-formulas-result
    '''

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(
        self,
        f,
        number_tabs=False,
        verbose=True,
        **kwargs
        ):
        super().__init__(f, **kwargs)
        self.format_cache = dict()
        self.sheet_cache = dict()
        self.number_tabs = number_tabs
        self.verbose = verbose
        self.page = 0


    #╭-------------------------------------------------------------------------╮
    #| Static Methods                                                          |
    #╰-------------------------------------------------------------------------╯

    @staticmethod
    def get_column_letter(index):
        ''' return the Excel column letter sequence for a given index (e.g.
            index=25 returns 'Z') '''
        letters = []
        while index >= 0:
            letters.append(chr(index % 26 + ord('A')))
            index = index // 26 - 1
        return ''.join(reversed(letters))


    @staticmethod
    def get_column_index(column):
        ''' return the Excel column index for a given letter sequence (e.g.
            column='Z' returns 25) '''
        index = 0
        for i, char in enumerate(reversed(column.upper())):
            index += (ord(char) - ord('A') + 1) * (26 ** i)
        return index - 1


    @staticmethod
    def parse_start_cell(start_cell):
        '''
        Description
        ------------
        Converts conventional representations of start cells into the zero-based
        numbering format required by xlsxwriter. For example, 'A1' would be
        converted to (0, 0).

        Parameters
        ------------
        start_cell : str | tuple | list
            Individual cell to be used as the starting position for writing data
            to an Excel worksheet. May be a standard excel range (e.g. 'A2') or
            a tuple of length two (column, row) (e.g. (1, 2) | ('A', 2)).

        Returns
        ------------
        out : tuple
            zero-based numbering row and column index (e.g. (0, 0)).
        '''
        if isinstance(start_cell, (tuple, list)):
            col, row = start_cell
            if isinstance(col, int):
                col -= 1
            else:
                col = ExcelFile.get_column_index(col)

        elif isinstance(start_cell, str):
            col = ExcelFile.get_column_index(
                re.findall(r'[a-zA-Z]+', start_cell)[0]
                )
            row = re.findall(r'\d+', start_cell)[0]

        else:
            raise TypeError(
                "start_cell must be of type 'str' or 'tuple', got: "
                f"{type(start_cell).__name__}"
                )

        row = int(row) - 1

        if row < 0 or col < 0:
            raise ValueError(
                f"invalid start_cell argument {start_cell}. "
                "Zero-based indices are not allowed."
                )

        return col, row


    @staticmethod
    def rgb_to_hex(red, green, blue):
        ''' convert RGB color to its hex equivalent '''
        return '#%02x%02x%02x' % (red, green, blue)


    #╭-------------------------------------------------------------------------╮
    #| Properties                                                              |
    #╰-------------------------------------------------------------------------╯

    @property
    def workbook(self):
        return self.writer.book


    @property
    def sheet_names(self):
        ''' list of the actual sheet names '''
        return [sheet.name for sheet in self.sheets]


    @property
    def sheet_name_map(self):
        ''' dictionary mapping user's worksheet names to their truncated
            counterparts '''
        return {k: v.name for k, v in self.sheet_cache.items()}


    @property
    def sheets(self):
        ''' list of worksheet objects '''
        return list(self.sheet_cache.values())


    @cached_property
    def writer(self):
        if self.verbose: print(f'\nCreating {self.name_ext}')
        return pd.ExcelWriter(self.path)


    @property
    def active_sheet(self):
        ''' current active worksheet '''
        # used __dict__ because hasattr calls __getattr__
        # when attr not found in __dict__
        if '_active_sheet' not in self.__dict__:
            self.create_worksheet('Sheet1')
        return self._active_sheet


    @active_sheet.setter
    @odd.validate_setter(types=str, call_func=True)
    def active_sheet(self, value):
        if value in self.sheet_cache:
            self._active_sheet = self.sheet_cache[value]
        else:
            self.create_worksheet(value)


    @cached_property
    def formats(self):

        header_body_presets = {
            'text_wrap': True,
            'border': True,
            'align': 'left',
            'valign': 'top'
            }

        format_presets = {
            # Font
            'bold': {'bold': True},

            # Alignment
            'text_wrap': {'text_wrap': True},

            # Pattern
            'highlight': {'fg_color': 'yellow'},

            # Protection
            'unlocked': {'locked': 0},

            # Number
            'commas': {'num_format': '#,##0'},
            'commas_two_decimals': {'num_format': '#,##0.00'},

            # Percent
            'percent': {'num_format': '0%'},
            'percent_one_decimals': {'num_format': '0.0%'},
            'percent_two_decimals': {'num_format': '0.00%'},

            # Date
            'date': {'num_format': 'mm/dd/yyyy'},
            'datetime': {'num_format': 'mm/dd/yyyy hh:mm:ss'},
            # 'datetime': {'num_format': 'mm/dd/yyyy hh:mm:ss.000 AM/PM'},

            # Border
            'top': {'top': True},
            'bottom': {'bottom': True},
            'left': {'left': True},
            'right': {'right': True},

            # Headers
            'pandas_header': {
                'border': True,
                'align': 'center',
                'valign': 'vcenter',
                'bold': True
                },

            'black_header': {
                'text_wrap': True,
                'fg_color': 'black',
                'border': True,
                'align': 'center',
                'valign': 'vcenter',
                'bold': True,
                'font_color': 'white'
                },

            'white_body': header_body_presets,
            'gold_body': {**header_body_presets, 'fg_color': '#FFE265'},
            'turqoise_body': {**header_body_presets, 'fg_color': '#CCFFFF'},

            # Miscellaneous
            'default_merge': {
                'border': True,
                'align': 'top',
                'text_wrap': True
                },

            'gold_wrap': {
                'border': True,
                'align': 'top',
                'text_wrap': True,
                'fg_color': '#FFE265'
                },

            'bold_total': {'align': 'right', 'bold': True},
            'header': {'bold': True,'bottom': True, 'text_wrap': True},

            # Conditional
            'conditional_red': {
                'bg_color': '#FFC7CE',
                'font_color': '#9C0006'
                },
            'conditional_green': {
                'bg_color': '#C6EFCE',
                'font_color': '#006100'
                },
            'conditional_yellow': {
                'bg_color': '#FFEB9C',
                'font_color': '#9C6500'
                },
            }

        return format_presets


    #╭-------------------------------------------------------------------------╮
    #| Magic Methods                                                           |
    #╰-------------------------------------------------------------------------╯

    def __getattr__(self, name):
        '''
        Description
        ------------
        Called when an attribute is referenced that does not exist. In this
        case, attributes that do not exist will be assumed to be active
        worksheet object attributes (e.g. '.hide_gridlines').

        Parameters
        ------------
        name : str
            xlsxwriter.worksheet.Worksheet attribute name

        Returns
        ------------
        out : *
            xlsxwriter.worksheet.Worksheet attribute
        '''
        return getattr(self.active_sheet, name)


    def __getitem__(self, key):
        '''
        Description
        ------------
        Indexing and slicing is applied to existing worksheets.

        Parameters
        ------------
        key : int | slice | str
            • if int or slice, applied to list of worksheet objects
            • if str, treated like a sheet_cache key (i.e. sheet name)

        Returns
        ------------
        out : xlsxwriter.worksheet.Worksheet | list
             one or more worksheet objects
        '''
        if isinstance(key, (int, slice)):
            return self.sheets[key]
        elif isinstance(key, str):
            return self.sheet_cache[key]
        else:
            raise TypeError(f"Invalid argument type: '{type(key)}'")


    def __iter__(self):
        ''' iterate through worksheet objects '''
        for key, sheet in self.sheet_cache.items():
            yield key, sheet


    def __contains__(self, key):
        ''' returns True if worksheet exists in workbook '''
        return key in self.sheet_cache


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    @odd.purge_whitespace
    def read(self, **kwargs):
        kwargs.setdefault('keep_default_na', False)
        return pd.read_excel(self.path, **kwargs)


    def _save(self, args=None, sheets=None, **kwargs):
        '''
        Description
        ------------
        Creates the excel file.

        Parameters
        ------------
        args : pd.DataFrame | list | dict
            • if list, each dataframe in the list is written to a separate
                worksheet.
            • if dict, the keys are considered sheet names and the values
                should be DataFrames. (i.e. {'My Sheet 1': df1, 'My Sheet 2':
                df2}). You may not pass a sheets argument when passing a
                dictionary as the args argument.
        sheets : list | str | None
            Mist of sheet names. Must be None when 'args' is a dictionary.
        kwargs : dict
            keyword arguments passed to write_df() method.

        Returns
        ------------
        None
        '''

        if args is not None:

            if isinstance(args, dict):
                if sheets is not None:
                    raise ValueError(
                        "'sheets' argument must be None "
                        "when a dictionary is passed"
                        )
                sheets = list(args.keys())
                args = list(args.values())
            else:
                args = odd.ensure_list(args)

            if sheets is not None:
                sheets = odd.ensure_list(sheets)
                if len(args) != len(sheets):
                    raise ValueError(
                        f"length of 'args' ({len(args)}) != length of 'sheets' "
                        f"({len(sheets)})."
                        )

            for i, arg in enumerate(args):
                self.write_df(
                    df=arg,
                    sheet=sheets[i] if sheets else f'Sheet{i + 1}',
                    **kwargs
                    )

        self.writer.close()


    def create_worksheet(self, name):
        ''' create a new worksheet and set it as the active worksheet '''

        key = name[:]

        if key in self.sheet_cache:
            raise ValueError(
                f"a worksheet with name='{name}' was already created."
                )

        if self.number_tabs:
            self.page += 1
            name = f'{self.page} - {name}'

        # Excel has a 31 character limit
        char_limit = 31
        name = name[:char_limit]

        # ensure truncated sheet names are unique
        sheet_names = set(self.sheet_names)

        counter = 2
        while name in sheet_names:
            index = char_limit - len(str(int(counter)))
            name = f'{name[:index]}{counter}'
            counter += 1

        self.sheet_cache[key] = self.workbook.add_worksheet(name)
        self.active_sheet = key


    def write(
        self,
        start_cell,
        data,
        formatting=None,
        inverse=False,
        repeat=None,
        sheet=None,
        outer_border=False
        ):
        '''
        Description
        ------------
        writes data to a worksheet

        Parameters
        ------------
        start_cell : str | tuple
            Individual cell range to be used as the starting position for the
            fill. May be a standard excel range (e.g. 'A2') or a tuple of
            length two (column, row) (e.g. (1, 2) | ('A', 2)).
        data : 1D list | 2D list | single value | pd.DataFrame | pd.Series
            Data to be written to excel worksheet starting in start_cell. Each
            sub-list is treated as row data and written in sequential order.
            For example, if start_cell = 'A1' and data = [[1, 2, 3], ['a', 'b',
            'c']], then row 1 and 2 will be filled with 1, 2, 3 and 'a', 'b',
            'c', respectively.
        formatting : str | dict | list(1D) | tuple(1D) | None
            Excel formatting to be applied to data. Supported types include:
                • str ➜ Strings are interpreted as self.formats dictionary
                    keys. Passing a single string (e.g. formatting='bold')
                    causes the corresponding format to be be universally
                    applied to all data cells.
                • dict ➜ Dictionaries are interpreted as format parameters.
                    Passing a single dictionary (e.g. formatting = {'bold':
                    True, 'num_format': '#,##0'}) causes the corresponding
                    format to be be universally applied to all data cells.
                • list | tuple ➜ formats included in a list/tuuple are applied
                    to columns in sequential order. If the length of the list
                    or tuple is shorter than the number of columns then no
                    format is applied to the remaining columns.
                • None ➜ no formatting is applied.
        inverse : bool
            If True, the 2D data is inverted such that each sub-list is
            treated as column data as opposed to row data under the default
            behavior. For example:
            [[1, 2, 3], ['a', 'b', 'c']] ➜ [[1, 'a'], [2, 'b'], [3, 'c']]
        repeat : int
            If 'data' argument is a single data element then it will be repeated
            or duplicated this number of times.
        sheet : str
            If None, self.active_sheet is utilized. If self.active_sheet has
                not yet been assigned, it will be assigned to a newly created
                blank worksheet named 'Sheet1'.
            If not None, self.active_sheet will be assigned to the passed
                sheet name. If the passed name does not correspond to an
                existing sheet, it will be created.
        outer_border : bool
            If True, data is encased in an outer border.

        Returns
        ------------
        None
        '''

        def process_format(x):
            ''' return dictionary representing format parameters '''
            if x is None:
                return
            elif isinstance(x, (str, tuple, list)):
                name = self.add_format(x)
                return self.formats[name]
            elif isinstance(x, dict):
                return x
            else:
                raise TypeError(
                    "format must be of type 'str' or 'dict', "
                    f"not: {type(x).__name__}"
                    )


        def format_builder(col, row):
            fmt = deepcopy(formatting[col]) or dict()
            updates = dict()

            if outer_border:
                if col == 0:
                    updates['left'] = 1
                if col == n_cols - 1:
                    updates['right'] = 1
                if row == 0:
                    updates['top'] = 1
                if row == n_rows - 1:
                    updates['bottom'] = 1
                fmt.update(updates)

            return self.get_format(fmt)


        if sheet is not None:
            self.active_sheet = sheet

        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.write_df(
                df=data,
                sheet=sheet,
                start_cell=start_cell,
                data_format=formatting,
                inverse=inverse,
                repeat=repeat,
                outer_border=outer_border,
                )
            return

        if self.verbose:
             size_label = get_size_label(sys.getsizeof(data))
             print(
                f'\twriting {size_label} to '
                f'{self.active_sheet.name!r} tab',
                end='... '
                )

        if not data:
            if self.verbose:
                print('SKIPPED')
            return

        start_col, start_row = self.parse_start_cell(start_cell)

        is_iter = lambda x: isinstance(x, (list, tuple))

        if not is_iter(data):
            # data is non-iterable
            data = [[data] * (repeat or 1)]
        else:
            if repeat: raise NotImplementedError
            # data is a one-dimensional iterable
            if not is_iter(data[0]):
                data = [data]

        if inverse: data = list(zip(*data))

        n_rows, n_cols = np.shape(data)

        formatting = list(map(process_format, formatting)) + \
                    ([None] * (n_cols - len(formatting))) \
                    if is_iter(formatting) \
                    else [process_format(formatting)] * n_cols

        for row_idx, row in enumerate(data):
            for col_idx, cell in enumerate(row):
                self.active_sheet.write(
                    start_row + row_idx,
                    start_col + col_idx,
                    cell if pd.notnull(cell) else None,
                    format_builder(col_idx, row_idx)
                    )

        if self.verbose:
            print('DONE')


    def write_df(
        self,
        df,
        sheet=None,
        start_cell='A1',
        header_format='pandas_header',
        data_format='auto',
        column_widths='auto',
        date_format=None,
        normalize=True,
        autofilter=False,
        raise_on_empty=True,
        total_row=False,
        total_row_format=None,
        total_column=False,
        total_column_format=None,
        **kwargs
        ):
        '''
        Description
        ------------
        Writes a DataFrame to an Excel worksheet. This is an alternative to
        df.to_excel() that addresses some of its limitations such as not being
        able format cells that already have a format including the index,
        headers, and cells that contain dates or datetimes.

        Parameters
        ------------
        df : pd.DataFrame | pd.Series
            DataFrame to be written to worksheet. If a Series is passed it will
            be converted to a DataFrame. Note: a copy is created so the original
            object will be unchanged.
        sheet : ↑
            See self.write documentation.
        start_cell : ↑
            See self.write documentation.
        header_format : ↑
            See self.write 'formatting' argument documentation.
        data_format : ↑
            Special cases:
            • 'auto' ➜ formatting is automatically applied to numeric, percent,
                and date fields.
            • dict ➜ if the dictionary keys are only comprised of 'df' index or
                column names then the values are treated like format parameters
                and formatting is only applied to those columns included in the
                keys. If not all the key values are column names then the
                dictionary receives the default treatment outlined in the
                self.write documentation. (e.g.{'Price': 'commas', 'Total':
                {'bold': True}})
            Other cases:
                see self.write 'formatting' argument documentation.
        column_widths : 'auto' | list | tuple | dict
            • 'auto' ➜ xlsxwriter does not support auto-fitting column widths
                so this attempts replicate it by setting the column width
                according to the length of the values in each column (up to a
                certain limit).
            • list | tuple ➜ widths are applied to columns in sequential order.
                If the length of the list/tuple is shorter than the
                number of columns then the width is not set on the
                remaining columns.
            • dict ➜ dictionary where keys are DataFrame column names and values
                are column widths. Any column names excluded from the dictionary
                will not have their widths set.
        date_format : None | str | dict
            Defines how date-like columns are parsed when data_format='auto'
            (e.g. '%Y-%m-%d'). Options include:
            • None ➜ format is inferred.
            • str ➜ used for all date-like columns.
            • dict ➜ dictionary where keys are DataFrame column names and values
                are formats. Any column names excluded from the dictionary
                default to None.
        normalize : bool
            if True, any date columns where the hours, minutes, seconds,
                microseconds are all set to zero (midnight) will be converted
                from a datetime to date.
        autofilter : bool
            if True, a filter will be applied to the column headers.
        raise_on_empty : bool
            if True and the 'df' argument is empty, an exception will be raised.
            if False, the 'df' columns will be written to an otherwise empty
                worksheet.
        total_row : bool
            if True, a row is added at the bottom reflecting the sum of each
                numeric column.
        total_row_format : ^
            format applied if 'total_row' is True. If None and data_format =
            'auto', then the same formatting will be applied to the total row
            plus bold and a top border.
        total_column : bool
            if True, a column is added at the end reflecting the sum of all
            numeric values in each row.
        total_column_format : ^
            format applied if 'total_column' is True. If None and data_format =
            'auto', then the same formatting will be applied to the total column
            plus bold and a left border.

        Returns
        ------------
        None
        '''

        df = odd.coerce_dataframe(df)

        odd.validate_value(
            value=date_format,
            name='date_format',
            types=(str, dict),
            none_ok=True
            )

        # kwargs housekeeping
        if kwargs.get('inverse'):
            raise NotImplementedError

        # Reset index
        if odd.has_named_index(df):
            df.reset_index(inplace=True)

        # Check if empty
        if df.empty:
            if raise_on_empty:
                raise ValueError(
                    "'df' argument cannot be empty."
                    )
            if len(df.columns) == 0:
                raise ValueError(
                    "'df' argument must have "
                    "an index or columns."
                    )
            total_row = False
            total_column = False

        # Check for duplicate column names
        odd.verify_unique(df, column_names=True)

        # Add a total column to dataframe
        if total_column:
            total_column_name = 'Total'
            if total_column_name in df.columns:
                raise ValueError(
                    "'df' already has a column "
                    f"named {total_column_name!r}"
                    )
            df[total_column_name] = df.sum(axis=1, numeric_only=True)

        # Categorize columns
        numeric_columns = set(
            df.select_dtypes(include=['number']).columns.tolist()
            )

        percent_columns = set([
            k for k in numeric_columns
            if any(z in k.lower() for z in ['%','percent'])
            ])

        numeric_columns -= percent_columns

        datelike_columns = set(
            df.select_dtypes(include=[np.datetime64])\
            .columns.tolist()
            )

        for k in df.columns:
            if isinstance(k, str) \
            and odd.column_name_is_datelike(k) \
            and not np.issubdtype(df[k].dtype, np.timedelta64):
                datelike_columns.add(k)

        # Parse start cell
        start_col, start_row = self.parse_start_cell(start_cell)

        # Write header
        self.write(
            start_cell=(start_col + 1, start_row + 1),
            data=df.columns.tolist(),
            formatting=header_format,
            sheet=sheet,
            **kwargs
            )

        # Force data_format to comply with the standard {column name : format}
        if isinstance(data_format, dict):
            # check if diciontary keys are index/column names
            if not all(k in df.columns for k in data_format):
                if any(isinstance(v, (list, tuple, dict))
                       for v in data_format.values()):
                    raise ValueError
                data_format = {k: data_format for k in df.columns}

        # Auto-detects the best format for each DataFrame column
        if data_format == 'auto':

            # cascade auto formatting
            if total_row_format is None:
                total_row_format = data_format[:]

            if total_column_format is None:
                total_column_format = data_format[:]

            data_format = {}

            infer_format = lambda fmt, s: \
                fmt if s.sum() - s.round().sum() == 0 \
                else f'{fmt}_two_decimals'

            clean_abs_numeric = lambda x: \
                pd.to_numeric(x, errors='coerce').dropna().abs()

            for k in numeric_columns:
                if not df[k].isna().all():
                    s = clean_abs_numeric(df[k])
                    abs_max = s.max()
                    if pd.notna(abs_max) and abs_max >= 1000:
                        data_format[k] = infer_format('commas', s)

            for k in percent_columns:
                if not df[k].isna().all():
                    s = clean_abs_numeric(df[k])
                    abs_min = s.where(s > 0).min()
                    if pd.notna(abs_min) and abs_min >= 1:
                        df[k] /= 100
                    else:
                        s *= 100
                    data_format[k] = infer_format('percent', s)

            if isinstance(date_format, dict):
                date_formats = {}
                for k in datelike_columns:
                    v = date_format.get(k)
                    if v is not None:
                        odd.validate_value(value=v, types=str)
                        date_formats[k] = v
            else:
                date_formats = {k: date_format for k in datelike_columns}

            datetime_columns = set()

            for k in list(datelike_columns):
                if np.issubdtype(df[k].dtype, np.datetime64) \
                    or df[k].isna().all():
                    continue

                fmt = date_formats.get(k)
                to_dt = lambda x: pd.to_datetime(x, format=fmt)

                try:
                    df[k] = to_dt(df[k])
                except:
                    try:
                        # handles mixed dtypes that will convert individually
                        df[k] = df[k].apply(odd.ignore_nan(to_dt))
                    except:
                        types = set(type(x) for x in df[k].dropna())
                        if types == {str}:
                            if fmt is None:
                                datelike_columns.remove(k)
                            else:
                                df[k] = df[k].apply(odd.ignore_nan(
                                    lambda x: datetime.datetime.strptime(x, fmt)
                                    ))
                                datetime_columns.add(k)
                        elif all(
                            issubclass(x, datetime.datetime)
                            for x in types
                            ):
                            datetime_columns.add(k)
                        else:
                            datelike_columns.remove(k)

            for k in datelike_columns:
                data_format[k] = 'datetime'
                if not normalize: continue
                s = df[k].dropna()
                if s.empty: continue
                if k in datetime_columns:
                    if all(x.time() == datetime.time.min for x in s.values):
                        df[k] = df[k].apply(odd.ignore_nan(lambda x: x.date()))
                        data_format[k] = 'date'
                else:
                    if (s == s.dt.normalize()).all():
                        df[k] = df[k].dt.date
                        data_format[k] = 'date'

        if isinstance(data_format, str):
            data_format = {k: data_format for k in df.columns}

        if isinstance(data_format, (list, tuple)):
            data_format = {k: v for k, v in zip(df.columns, data_format)}

        if data_format is not None and not isinstance(data_format, dict):
            raise TypeError(
                "'data_format' argument does not support type: "
                f"{type(data_format).__name__}."
                )

        # Change total column to formulas
        if total_column:
            cols = []
            for k in numeric_columns:
                if k != total_column_name:
                    col = self.get_column_letter(
                        start_col + df.columns.get_loc(k)
                        )
                    cols.append(col)

            cell_blocks = []
            for i in range(len(df)):
                row = start_row + i + 2
                cell_blocks.append([f'{col}{row}' for col in cols])

            df[total_column_name] = [
                f"=SUM({','.join(cells)}" for cells in cell_blocks
                ]

            if total_column_format == 'auto':
                data_format[total_column_name] = \
                    self.add_format(odd.ensure_list(
                        data_format.pop(total_column_name, [])
                        ) + ['bold','left'])
            else:
                if total_column_format:
                    if data_format:
                        data_format[total_column_name] = total_column_format
                    else:
                        data_format = {total_column_name: total_column_format}

        # Write data
        self.write(
            start_cell=(start_col + 1, start_row + 2),
            data=df.replace([np.inf, -np.inf], np.nan)\
                   .where(df.notnull(), None).values.tolist(),
            formatting=None if data_format is None else \
                       [data_format.get(k) for k in df.columns],
            sheet=sheet,
            **kwargs
            )

        # Add a total row
        if total_row:
            total_row = []
            first_row = start_row + 2
            last_row = first_row + len(df) - 1
            for k in df.columns:
                if k in numeric_columns:
                    col = self.get_column_letter(
                        start_col + df.columns.get_loc(k)
                        )
                    total_row.append(
                        f'=SUM({col}{first_row}:{col}{last_row})'
                        )
                else:
                    total_row.append(None)

            if total_row_format == 'auto':
                total_row_format = []
                for k in df.columns:
                    fmt = ['bold','top']
                    if total_column and k == total_column_name:
                        fmt.append('left')
                    if k in data_format:
                        fmt.append(data_format[k])
                    total_row_format.append(fmt)

            if total_row_format is None and data_format is not None:
                total_row_format = [data_format.get(k) for k in df.columns]

            self.write(
                start_cell=(start_col + 1, last_row + 1),
                data=total_row,
                formatting=total_row_format,
                sheet=sheet,
                **kwargs
                )

        # Set column widths
        set_column_width = lambda i, w: self.set_column(
            '{0}:{0}'.format(self.get_column_letter(start_col + i)), w
            )

        if column_widths == 'auto':
            column_widths = []
            for k in df.columns:
                name_width = len(str(k))
                value_width = df[k].astype(str).str.len().max()
                max_width = max(name_width, value_width) + 1
                column_widths.append(min(max_width, 50)) # setting cap at 50

        if isinstance(column_widths, (list, tuple)):
            for i, w in enumerate(column_widths):
                set_column_width(i, w)
        elif isinstance(column_widths, dict):
            for k, w in column_widths.items():
                set_column_width(df.columns.get_loc(k), w)
        elif isinstance(column_widths, (int, float)):
            for i in range(len(df.columns)):
                set_column_width(i, column_widths)
        else:
            raise TypeError(
                f"'column_widths' argument does not support type: "
                f"{type(column_widths).__name__}."
                )

        # Set autofilter
        if autofilter:
            self.autofilter(
                start_row,
                start_col,
                start_row + len(df) + (0 if total_row is None else 1) - 1,
                start_col + len(df.columns) - 1
                )


    def fill_formula(
        self,
        start_cell,
        formula,
        limit,
        headers=None,
        formatting=None,
        down=True,
        outer_border=False
        ):
        '''
        Description
        ------------
        fills a formula down or to the right

        Parameters
        ------------
        start_cell : str | tuple | list
            Individual cell range to be used as the starting position for the
            fill. May be a standard excel range (e.g. 'A2') or a tuple of
            length two (column, row) (e.g. (1, 2) | ('A', 2)). Similar to the
            formula argument, column names may include placeholders (e.g.
            '{Price}2' | ('{Price}', 2)).
        formula : str
            Excel formula to be written to the start cell and used as a fill
            template. Column names may include placeholders with header names
            if 'headers' argument is passed (e.g. '=A1+B1-{Price}1'.
            Placeholders make the formula robust to changes in header
            positioning.
        limit : int
            Number of rows or columns to fill.
        headers : list
            List of column header names. Required argument when placeholders
            are used in formula or start_cell.
        formatting : str
            see write() argument of the same name.
        down : bool
            If True, formula is filled down otherwise it is filled right.
        outer_border : bool
            see write() argument of the same name.

        Returns
        ------------
        None
        '''

        if isinstance(start_cell, (list, tuple)):
            col, row = start_cell
            if isinstance(col, int):
                col = self.get_column_letter(col - 1)
            start_cell = f'{col}{row}'

        if headers:
            header_to_column_map = {
                header: self.get_column_letter(index)
                for index, header in enumerate(headers)
                }
            start_cell = start_cell.format(**header_to_column_map)
            formula = formula.format(**header_to_column_map)

        components = list(set(re.findall('([a-zA-Z]+)(\d+)', formula)))
        cols, rows = zip(*components)
        cols = [self.get_column_index(x) + 1 for x in cols]
        rows = [int(x) for x in rows]

        build_counter = lambda x: \
            OrderedDict((i, k) for i, k in enumerate(sorted(list(set(x)))))

        counter = build_counter(rows) if down else build_counter(cols)
        inverse_counter = {v: k for k, v in counter.items()}

        for c, r in zip(cols, rows):
            x, y = self.get_column_letter(c - 1), str(r)
            repl = x + ('{%d}' % inverse_counter[r]) if down else \
                       ('{%d}' % inverse_counter[c]) + y
            formula = formula.replace(x + y, repl)

        data = []
        for x in range(limit):
            format_args = list(counter.values())
            if not down:
                format_args = [
                    self.get_column_letter(c - 1)
                    for c in format_args
                    ]
            data.append(formula.format(*format_args))
            for k in counter:
                counter[k] += 1

        self.write(
            start_cell=start_cell,
            data=data,
            formatting=formatting,
            inverse=down,
            outer_border=outer_border
            )


    def add_format(self, name, fmt=None):
        '''
        Description
        ------------
        adds a dictionary entry into self.formats where the key is the desired
        name of the format and the value is a dictionary representing format
        parameters

        Parameters
        ------------
        name : str | tuple | list
            • str ➜ name of new format.
                If 'fmt' is None, if the name is already in self.formats then
                    no action is taken. If the name does not already exist then
                    the name's components (delimited by underscores ('_')) will
                    be combined into a new format (e.g.'bold_commas').
                if 'fmt' is not None, if the name conflicts with an existing
                    format name, the existing format will be overwritten.
            • tuple | list ➜ If 'fmt' is None, components will be combined
                into a new format. For example, self.add_format(name=
                ['bold','commas'], fmt=None) would add the following entry to
                self.formats: {'bold_commas': {'bold': True, 'num_format':
                '#,##0'}
        fmt : dict | None
            • dict ➜ see https://xlsxwriter.readthedocs.io/format.html
            • None ➜ format will be constructed based on the components

        Returns
        ------------
        out : str
            Name of format (in other words, the self.formats dictionary key
            value)
        '''
        if fmt is None:
            if isinstance(name, str):
                if name in self.formats:
                    return name
                name = name.split('_')

            odd.validate_value(
                value=name,
                name=name,
                types=(tuple, list)
                )

            fmt = dict()
            for k in name: fmt.update(self.formats[k])
            name = '_'.join(odd.natural_sort(name))
        else:
            if not isinstance(name, str):
                raise TypeError(
                    "'name' argument must be a string "
                    "if 'fmt' is not None, not: "
                    f"{type(fmt).__name__}"
                    )

            if not isinstance(fmt, dict):
                raise TypeError(
                    "'fmt' argument must be a dictionary, not: "
                    f"{type(fmt).__name__}"
                    )

        self.formats[name] = fmt
        return name


    def get_format(self, arg):
        '''
        Description
        ------------
        Takes an argument and returns the corresponding format object.
        The format will be added to the workbook and cached if it does not
        already exist.

        Parameters
        ------------
        arg : str | tuple | list | dict | None
            • str | tuple | list ➜ passed to add_format as 'name' argument.
            • dict ➜ Format parameters. The resulting format will not be
                      available in self.formats but it will still be cached and
                      can be accessed by passing the same dictionary to this
                      function.
            • None ➜ None is returned

        Returns
        ------------
        out : xlsxwriter.format.Format | None
            format object
        '''
        if not arg: return

        if not isinstance(arg, dict):
            name = self.add_format(name=arg)
            arg = self.formats[name]

        key = sha256(str(sorted(list(arg.items()))))
        if key not in self.format_cache:
            self.format_cache[key] = self.workbook.add_format(arg)

        return self.format_cache[key]