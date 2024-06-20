from collections import OrderedDict
from copy import deepcopy
import hashlib
import sys
import re
import pandas as pd
import numpy as np
from iterlab import natural_sort, to_iter

from ._file import FileBase
from .utils import purge_whitespace, get_size_label



class ExcelFile(FileBase):
    '''
    Description
    --------------------
    File object that also interfaces with ExcelWriter to facilitate working with Excel files.
    This documentation is for the polymorphism only. See the base class for more information.

    Class Attributes
    --------------------
    ...

    Instance Attributes
    --------------------
    writer : ExcelWriter
        pd.ExcelWriter instance
    formats : dict
        keys are strings representing names of formats and values are dictionaries
        containing format parameters. For Example, {{'bold': {'bold': True}}. These
        are intended to be frequently used formats that can be both used individually
        and as building blocks for more complex formats (see self.add_format for more
        information.)
    active_worksheet : xlsxwriter.worksheet.Worksheet
        The active worksheet
    format_cache : dict
        keys are hashes of sorted dictionaries containing format parameters and
        values are xlsxwriter.format.Format. This is done so that only one
        format instance will be created per unique format used.
    sheet_cache : dict
        keys are worksheet names and values are the actual worksheet names as
        they appear in the spreadsheet. The reasoning behind this is that the
        desired name may be truncated by the 31 character limit or have a page
        prefix added so this allows the user to access the sheet using the
        original name.
    number_tabs : bool
        if True, tabs in the workbook will have a number prefix. For example,
        'MyTab' would be appear as '1 - MyTab' in the workbook.
    page : int
        the current tab number
    verbose : bool
        if True, information is printed when a worksheet is created or written to.
    troubleshoot : bool
        if True, additional information is printed for troubleshooting purposes.

    Notes:
    --------------------
    This URL contains instructions on how to get LibreOffice to calculate formulas (0 by default)
    https://stackoverflow.com/questions/32205927/xlsxwriter-and-libreoffice-not-showing-formulas-result

    '''

    #+---------------------------------------------------------------------------+
    # Initialize Instance
    #+---------------------------------------------------------------------------+

    def __init__(self, f, number_tabs=False, verbose=True, troubleshoot=False, **kwargs):
        super().__init__(f, **kwargs)
        self.format_cache = dict()
        self.sheet_cache = dict()
        self.number_tabs = number_tabs
        self.page = 0
        self.verbose = verbose
        self.troubleshoot = troubleshoot


    #+---------------------------------------------------------------------------+
    # Static Methods
    #+---------------------------------------------------------------------------+

    @staticmethod
    def get_column_letter(index):
        ''' return the Excel column letter sequence for a given index (e.g. index=25 returns 'Z') '''
        letters = []
        while index >= 0:
            letters.append(chr(index % 26 + ord('A')))
            index = index // 26 - 1
        return ''.join(reversed(letters))


    @staticmethod
    def get_column_index(column):
        ''' return the Excel column index for a given letter sequence (e.g. column='Z' returns 25) '''
        index = 0
        for i, char in enumerate(reversed(column.upper())):
            index += (ord(char) - ord('A') + 1) * (26 ** i)
        return index - 1


    @staticmethod
    def parse_start_cell(start_cell):
        '''
        Description
        ------------
        Converts conventional representations of start cells into the zero-based numbering format
        required by xlsxwriter. For example, 'A1' would be converted to (0, 0).

        Parameters
        ------------
        start_cell : str | tuple | list
            Individual cell to be used as the starting position for writing data to an Excel
            worksheet. May be a standard excel range (e.g. 'A2') or a tuple of length two
            (column, row) (e.g. (1, 2) | ('A', 2)).

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
            col = ExcelFile.get_column_index(re.findall(r'[a-zA-Z]+', start_cell)[0])
            row = re.findall(r'\d+', start_cell)[0]
        else:
            raise TypeError(f"start_cell must be of type 'str' or 'tuple' not {type(start_cell)}")

        row = int(row) - 1

        if row < 0 or col < 0:
            raise ValueError(f"invalid start_cell argument {start_cell}. Zero-based indices are not allowed.")

        return col, row


    @staticmethod
    def rgb_to_hex(red, green, blue):
        ''' convert RGB color to its hex equivalent '''
        return '#%02x%02x%02x' % (red, green, blue)


    #+---------------------------------------------------------------------------+
    # Properties
    #+---------------------------------------------------------------------------+

    @property
    def workbook(self):
        return self.writer.book


    @property
    def worksheet_names(self):
        return [worksheet.name for worksheet in self]


    #+---------------------------------------------------------------------------+
    # Magic Methods
    #+---------------------------------------------------------------------------+

    def __getattr__(self, name):
        '''
        Description
        ------------
        Used to access an object's attributes that do not exist or cannot be accessed through normal means.
        It is called when an attribute is accessed using the dot notation (.). In this case, attributes that
        do not exist will be gotten from the currently active worksheet object (e.g. '.hide_gridlines').

        Parameters
        ------------
        name : str
            xlsxwriter.worksheet.Worksheet attribute name

        Returns
        ------------
        out : ...
            xlsxwriter.worksheet.Worksheet attribute
        '''

        if self.troubleshoot:
            print(f"self.__getattr__(name='{name}')")

        if name == 'writer':
            self.writer = pd.ExcelWriter(self.path)
            if self.verbose: print(f'\nCreating {self.nameext}')
            return self.writer
        elif name == 'formats':
            self.formats = self._get_preset_formats()
            return self.formats
        elif name == 'active_worksheet':
            self.create_worksheet('Sheet1')
            return self.active_worksheet
        else:
            return getattr(self.active_worksheet, name)


    def __getitem__(self, key):
        '''
        Description
        ------------
        Used to access an object's items using the square bracket notation. It is called when an object is indexed
        or sliced with square brackets ([]). In this case, indexing and slicing is applied to writer.book.worksheets_objs
        which will return one or more xlsxwriter.worksheet.Worksheet objects.

        Parameters
        ------------
        key : int | slice
            key

        Returns
        ------------
        out : xlsxwriter.worksheet.Worksheet | list
             one or more worksheet objects
        '''
        if isinstance(key, (int, slice)):
            return self.workbook.worksheets_objs[key]
        elif isinstance(key, str):
            for worksheet in self:
                if worksheet.name in (key, self.sheet_cache.get(key)):
                    return worksheet
            raise KeyError(key)
        else:
            raise TypeError(f"Invalid argument type: '{type(key)}'")


    def __iter__(self):
        ''' iterate through worksheet objects '''
        for worksheet in self.workbook.worksheets_objs:
            yield worksheet


    def __contains__(self, key):
        ''' returns True if worksheet exists in workbook '''
        try:
            self[key]
            return True
        except:
            return False



    #+---------------------------------------------------------------------------+
    # Instance Methods
    #+---------------------------------------------------------------------------+

    @purge_whitespace
    def read(self, **kwargs):
        df = pd.read_excel(
            io=self.path,
            keep_default_na=False,
            **kwargs
            )
        return df


    def _save(self, args=None, sheets=None, **kwargs):
        
        if isinstance(args, dict):
            if sheets is not None: raise ValueError
            sheets = list(args.keys())
            args = list(args.values())

        if args is not None:
            for i, arg in enumerate(to_iter(args)):
                self.write_df(
                    df=arg,
                    sheet=sheets[i] if sheets else f'Sheet{i + 1}',
                    **kwargs
                    )

        self.writer.close()


    def set_active_worksheet(self, key):
        self.active_worksheet = self[key]


    def create_worksheet(self, name):
        ''' create a new worksheet and set it as the active worksheet '''
        if self.number_tabs:
            self.page += 1
            out = f'{self.page} - {name}'
        else:
            out = name[:]
        out = out[:31] # Excel has a 31 character limit
        self.active_worksheet = self.workbook.add_worksheet(out)
        self.sheet_cache[name] = out
        return out


    def write(self, start_cell, data, formatting=None, inverse=False, repeat=None, sheet=None, outer_border=False):
        '''
        Description
        ------------
        writes data to a worksheet

        Parameters
        ------------
        start_cell : str | tuple
            Individual cell range to be used as the starting position for the fill. May be
            a standard excel range (e.g. 'A2') or a tuple of length two (column, row)
            (e.g. (1, 2) | ('A', 2)).
        data : 1D list | 2D list | single data element | pd.DataFrame | pd.Series
            Data to be written to excel worksheet starting in start_cell. Each sub-list is
            treated as row data and written in sequential order. For example, if start_cell
            = 'A1' and data = [[1, 2, 3], ['a', 'b', 'c']], then row 1 and 2 will be filled
            with 1, 2, 3 and 'a', 'b', 'c', respectively.
        formatting : str | dict | one-dimensional list | one-dimensional tuple | None
            Excel formatting to be applied to data. Supported types include:
                • str -> Strings are interpreted as self.formats dictionary keys. Passing a
                         single string (e.g. formatting='bold') causes the corresponding format
                         to be be universally applied to all data cells.
                • dict -> Dictionaries are interpreted as format parameters. Passing a single
                          dictionary (e.g. formatting={'bold': True, 'num_format': '#,##0'})
                          causes the corresponding format to be be universally applied to all
                          data cells.
                • list | tuple -> formats included in a list/tuuple are applied to columns in
                          sequential order. If the length of the list/tuple is shorter than the
                           number of columns then no format is applied to the remaining columns.
                • None -> no formatting is applied
        inverse : bool
            If True, the 2D data is inverted such that each sub-list is treated as column data as
            opposed to row data under the default behavior.
            (e.g. [[1, 2, 3], ['a', 'b', 'c']] -> [[1, 'a'], [2, 'b'], [3, 'c']]
        repeat : int
            If 'data' argument is a single data element then it will be repeated or duplicated
            this number of times.
        sheet : str | int
            If None, self.active_worksheet is utilized. If self.active_worksheet has not yet been
            assigned then it is assigned to a newly created blank worksheet named 'Sheet1'.
            If not None, self.active_worksheet will be assigned via self.set_active_worksheet.
            If the worksheet does not already exist and sheet is of type 'str', it is created as
            a new blank worksheet.
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
                raise TypeError(f"format must be of type 'str' or 'dict', not {type(x)}")


        def format_builder(col, row):
            fmt = deepcopy(formatting[col]) or dict()
            updates = dict()

            if outer_border:
                if col == 0: updates['left'] = 1
                if col == n_cols - 1: updates['right'] = 1
                if row == 0: updates['top'] = 1
                if row == n_rows - 1: updates['bottom'] = 1
                fmt.update(updates)

            return self.get_format(fmt)



        if sheet is not None:
            if sheet in self:
                self.set_active_worksheet(sheet)
            else:
                if isinstance(sheet, str):
                    self.create_worksheet(sheet)
                else:
                    raise TypeError(f"'sheet' argument must be a string if worksheet has not been created yet, not {type(sheet)}")


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
             print(f"\twriting {size_label} to '{self.active_worksheet.name}' tab", end='... ')

        if not data:
            if self.verbose: print('SKIPPED')
            return

        start_col, start_row = self.parse_start_cell(start_cell)

        is_iter = lambda x: isinstance(x, (list, tuple))

        if not is_iter(data):
            # data is non-iterable
            data = [[data] * (repeat or 1)]
        else:
            if repeat: raise NotImplementedError
            # data is 1D iterable
            if not is_iter(data[0]):
                data = [data]

        if inverse: data = list(zip(*data))

        n_rows, n_cols = np.shape(data)

        formatting = list(map(process_format, formatting)) + ([None] * (n_cols - len(formatting))) \
                     if is_iter(formatting) else [process_format(formatting)] * n_cols

        for row_idx, row in enumerate(data):
            for col_idx, cell in enumerate(row):
                self.active_worksheet.write(
                    start_row + row_idx,
                    start_col + col_idx,
                    cell if pd.notnull(cell) else None,
                    format_builder(col_idx, row_idx)
                    )

        if self.verbose: print('DONE')


    def write_df(
        self,
        df,
        sheet=None,
        start_cell='A1',
        header_format='pandas_header',
        data_format='auto',
        column_widths='auto',
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
        Writes a DataFrame to an Excel worksheet. This function is a superior alternative
        to df.to_excel() because it does not share the same limitations such as not being
        able format cells that already have a format including the index, headers, and
        cells that contain dates or datetimes.

        Parameters
        ------------
        df : pd.DataFrame | pd.Series
            DataFrame to be written to worksheet. If a Series is passed it will be
            converted to a DataFrame. Note: a copy is created so the original object
            will be unchanged.
        sheet : ^
            See self.write documentation.
        start_cell : ^
            See self.write documentation.
        header_format : ^
            See self.write 'formatting' argument documentation.
        data_format : ^
            Special cases:
            • 'auto' -> formatting is automatically applied to numeric, percent, and date fields.
            • dict -> if the dictionary keys are only comprised of 'df' index or column names then
                      the values are treated like format parameters and formatting is only applied
                      to those columns included in the keys. If not all the key values are column
                      names then the dictionary receives the default treatment outlined in the self.write
                      documentation. (e.g. {'Price': 'commas', 'Total': {'bold': True}})
            Other cases:
                see self.write 'formatting' argument documentation.
        column_widths : 'auto' | list | tuple | dict
            • 'auto' -> xlsxwriter does not support auto-fitting column widths so
                           this attempts replicate it by setting the column width
                           according to the length of the values in each column
                           (up to a certain limit).
            • list | tuple -> widths are applied to columns in sequential order.
                           If the length of the list/tuple is shorter than the
                           number of columns then the width is not set on the
                           remaining columns.
            • dict -> dictionary where keys are DataFrame column names and values
                           are column widths. Any column names excluded from the
                           dictionary will not have their widths set.
        normalize : bool
            if True, any date columns where the hours, minutes, seconds, microseconds are
            all set to zero (midnight) will be converted from a datetime to date.
        autofilter : bool
            if True, a filter will be applied to the column headers.
        raise_on_empty : bool
            if True and the 'df' argument is empty, an exception will be raised.
            if False, the 'df' columns will be written to an otherwise empty worksheet.
        total_row : bool
            if True, a row is added at the bottom reflecting the sum of each numeric column.
        total_row_format : ^
            format applied if 'total_row' is True. If None and data_format='auto', then
            the same formatting will be applied to the total row plus bold and a top border.
        total_column : bool
            if True, a column is added at the end reflecting the sum of all numeric values in
             each row.
        total_column_format : ^
            format applied if 'total_column' is True. If None and data_format='auto', then
            the same formatting will be applied to the total column plus bold and a left border.

        Returns
        ------------
        None
        '''

        # Type housekeeping
        if isinstance(df, pd.DataFrame):
            df = df.copy(deep=True)
        elif isinstance(df, pd.Series):
            df = df.to_frame()
        else:
            raise TypeError(f"'df' argument type {type(df)} not supported.")

        # kwargs housekeeping
        if kwargs.get('inverse'): raise NotImplementedError

        # Reset index
        if list(filter(None, list(df.index.names))): df.reset_index(inplace=True)

        # Check if empty
        if df.empty:
            if raise_on_empty: raise ValueError("'df' argument cannot be empty.")
            if not df.columns.tolist(): raise ValueError("'df' argument must have an index or columns")
            total_row, total_column = False, False

        # Check for duplicate column names
        s = df.columns.value_counts()
        dupes = s[ s > 1 ].to_frame()
        if len(dupes) > 0: raise ValueError(f"'df' argument cannot have duplicate column names: \n\n{dupes}\n")

        # Add a total column to dataframe
        if total_column:
            total_column_name = 'Total'
            if total_column_name in df.columns: raise ValueError(f"'df' already has a column named '{total_column_name}'")
            df[total_column_name] = df.sum(axis=1, numeric_only=True)

        # Categorize columns
        numeric_columns = set(df._get_numeric_data().columns.tolist())
        percent_columns = set([k for k in numeric_columns if any(z in k.lower() for z in ['%','percent'])])
        numeric_columns -= percent_columns

        datelike_columns = set(df.select_dtypes(include=[np.datetime64]).columns.tolist())
        for k in df.columns:
            if isinstance(k, str) and \
               (any(x in k.lower() for x in ('date','time')) or k.lower()[-2:] == 'dt') and \
               str(df[k].dtype) != 'timedelta64[ns]':
                datelike_columns.add(k)

        date_columns, datetime_columns = [], []
        for k in list(datelike_columns):
            if not np.issubdtype(df[k].dtype, np.datetime64):
                try:
                    df[k] = pd.to_datetime(df[k])
                except:
                    datelike_columns.remove(k)

        for k in datelike_columns:
            if normalize and (df[k].dropna() == df[k].dropna().dt.normalize()).all():
                df[k] = df[k].dt.date
                date_columns.append(k)
            else:
                datetime_columns.append(k)

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
            if not all(k in df.columns for k in data_format):
                if any(isinstance(v, (list, tuple, dict)) for v in data_format.values()):
                    raise ValueError
                data_format = {k: data_format for k in df.columns}

        # Automatically determine the best formatting options for each dataframe column
        if data_format == 'auto':
            data_format = dict()

            # cascade auto formatting
            if total_row_format is None: total_row_format = 'auto'
            if total_column_format is None: total_column_format = 'auto'

            for k in numeric_columns:
                if not df[k].isna().all():
                    s = df[k].dropna().abs()
                    if s.max() >= 1000:
                        data_format[k] = 'commas' if s.sum() - s.round().sum() == 0 else 'commas_two_decimals'

            for k in percent_columns: data_format[k] = 'percent_two_decimals'
            for k in datetime_columns: data_format[k] = 'datetime'
            for k in date_columns: data_format[k] = 'date'

        if isinstance(data_format, str):
            data_format = {k: data_format for k in df.columns}

        if isinstance(data_format, (list, tuple)):
            data_format = {k: v for k,v in zip(df.columns, data_format)}

        if data_format is not None and not isinstance(data_format, dict):
            raise TypeError(f"'data_format' argument does not support type {type(data_format)}.")

        # Change total column to formulas
        if total_column:
            cols = []
            for k in numeric_columns:
                if k != total_column_name:
                    col = self.get_column_letter(start_col + df.columns.get_loc(k))
                    cols.append(col)

            cell_blocks = []
            for i in range(len(df)):
                row = start_row + i + 2
                cell_blocks.append([f'{col}{row}' for col in cols])

            df[total_column_name] = [f"=SUM({','.join(cells)}" for cells in cell_blocks]

            if total_column_format == 'auto':
                data_format[total_column_name] = \
                    self.add_format(to_iter(data_format.pop(total_column_name, [])) + ['bold','left'])
            else:
                if total_column_format:
                    if data_format:
                        data_format[total_column_name] = total_column_format
                    else:
                        data_format = {total_column_name: total_column_format}

        # Write data
        self.write(
            start_cell=(start_col + 1, start_row + 2),
            data=df.replace([np.inf, -np.inf], np.nan).where(df.notnull(), None).values.tolist(),
            formatting=[data_format.get(k) for k in df.columns] if data_format is not None else None,
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
                    col = self.get_column_letter(start_col + df.columns.get_loc(k))
                    total_row.append(f'=SUM({col}{first_row}:{col}{last_row})')
                else:
                    total_row.append(None)

            if total_row_format == 'auto':
                total_row_format = []
                for k in df.columns:
                    fmt = ['bold','top']
                    if total_column and k == total_column_name: fmt.append('left')
                    if k in data_format: fmt.append(data_format[k])
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
        set_column_width = lambda i, w: self.set_column('{0}:{0}'.format(self.get_column_letter(start_col + i)), w)

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
            raise TypeError(f"'column_widths' argument does not support type {type(column_widths)}.")

        # Set autofilter
        if autofilter:
            self.autofilter(
                start_row,
                start_col,
                start_row + len(df) + (0 if total_row is None else 1) - 1,
                start_col + len(df.columns) - 1
                )


    def fill_formula(self, start_cell, formula, limit, headers=None, formatting=None, down=True, outer_border=False):
        '''
        Description
        ------------
        fills a formula down or to the right

        Parameters
        ------------
        start_cell : str | tuple | list
            Individual cell range to be used as the starting position for the fill. May be
            a standard excel range (e.g. 'A2') or a tuple of length two (column, row)
            (e.g. (1, 2) | ('A', 2)). Similar to the formula argument, column names may include
            placeholders (e.g. '{Price}2' | ('{Price}', 2)).
        formula : str
            Excel formula to be written to the start cell and used as a fill template.
            Column names may include placeholders with header names if 'headers'
            argument is passed (e.g. '=A1+B1-{Price}1'. Placeholders make the formula
            robust to changes in header positioning.
        limit : int
            Number of rows or columns to fill.
        headers : list
            List of column header names. Required argument when placeholders are used in formula or
            start_cell.
        formatting : str
            see ExcelFile.write() argument of the same name.
        down : bool
            If True, formula is filled down otherwise it is filled right.
        outer_border : bool
            see ExcelFile.write() argument of the same name.

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
            header_to_column_map = {header: self.get_column_letter(index) for index, header in enumerate(headers)}
            start_cell = start_cell.format(**header_to_column_map)
            formula = formula.format(**header_to_column_map)

        components = list(set(re.findall('([a-zA-Z]+)(\d+)', formula)))
        cols, rows = zip(*components)
        cols, rows = [self.get_column_index(x) + 1 for x in cols], [int(x) for x in rows]

        build_counter = lambda x: OrderedDict((i, k) for i,k in enumerate(sorted(list(set(x)))))
        counter = build_counter(rows) if down else build_counter(cols)
        inverse_counter = {v: k for k,v in counter.items()}

        for c, r in zip(cols, rows):
            x, y = self.get_column_letter(c - 1), str(r)
            repl = x + ('{%d}' % inverse_counter[r]) if down else ('{%d}' % inverse_counter[c]) + y
            formula = formula.replace(x + y, repl)

        data = []
        for x in range(limit):
            format_args = list(counter.values())
            if not down: format_args = [self.get_column_letter(c - 1) for c in format_args]
            data.append(formula.format(*format_args))
            for k in counter: counter[k] += 1

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
        adds a dictionary entry into self.formats where the key is the desired name
        of the format and the value is a dictionary representing format parameters

        Parameters
        ------------
        name : str | tuple | list
            • str -> name of new format.
                     If 'fmt' is None, if the name is already in self.formats then
                     no action is taken. If the name does not already exist then
                     the name's components (delimited by underscores ('_')) will be
                     combined into a new format (e.g. 'bold_commas').
                     if 'fmt' is not None, if the name conflicts with an existing format
                     name, the existing format will be overwritten.
            • tuple | list -> If 'fmt' is None, components will be combined into a new
                     format. For example, self.add_format(name=['bold','commas'], fmt=None)
                     would add the following entry to self.formats:
                     {'bold_commas': {'bold': True, 'num_format': '#,##0'}

        fmt : dict | None
            • dict -> see https://xlsxwriter.readthedocs.io/format.html
            • None -> format will be constructed based on the components

        Returns
        ------------
        out : str
            name of format (in other words, the self.formats dictionary key value)
        '''
        if fmt is None:
            if isinstance(name, str):
                if name in self.formats: return name
                name = name.split('_')
            if not isinstance(name, (tuple, list)):
                raise TypeError(f"'name' arguments of type {type(name)} not supported.")

            fmt = dict()
            for k in name: fmt.update(self.formats[k])
            name = '_'.join(natural_sort(name))
        else:
            if not isinstance(name, str):
                raise TypeError(f"'name' argument must be a string if 'fmt' is not None, not {type(fmt)}")
            if not isinstance(fmt, dict):
                raise TypeError(f"'fmt' argument must be a dictionary, not {type(fmt)}")

        self.formats[name] = fmt
        return name


    def get_format(self, arg):
        '''
        Description
        ------------
        Takes an argument and returns the corresponding format object.
        The format will be added to the workbook and cached if it does not already exist.

        Parameters
        ------------
        arg : str | tuple | list | dict | None
            • str | tuple | list -> passed to add_format as 'name' argument.
            • dict -> Format parameters. The resulting format will not be available in self.formats
                      but it will still be cached and can be accessed by passing the same dictionary
                      to this function.
            • None -> None is returned

        Returns
        ------------
        out : xlsxwriter.format.Format | None
            format object
        '''
        if not arg: return

        if not isinstance(arg, dict):
            name = self.add_format(name=arg)
            arg = self.formats[name]

        sha256 = hashlib.sha256()
        sha256.update(bytes(str(sorted(list(arg.items()))), encoding='utf-8'))
        key = sha256.hexdigest()
        if key not in self.format_cache:
            self.format_cache[key] = self.workbook.add_format(arg)

        return self.format_cache[key]


    def _get_preset_formats(self):
        ''' frequently used formats '''

        header_body_presets = {
            'text_wrap': True,
            'border': True,
            'align': 'left',
            'valign': 'top'
            }

        formats = {
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

            # Header
            'pandas_header': {'border': True, 'align': 'center', 'valign': 'vcenter', 'bold': True},
            'black_header': {'text_wrap': True, 'fg_color': 'black', 'border': True, 'align': 'center',
                             'valign': 'vcenter', 'bold': True, 'font_color': 'white'},
            'white_body': header_body_presets,
            'gold_body': {**header_body_presets, 'fg_color': '#FFE265'},
            'turqoise_body': {**header_body_presets, 'fg_color': '#CCFFFF'},

            # Miscellaneous
            'default_merge': {'border': True, 'align': 'top','text_wrap': True},
            'gold_wrap': {'border': True, 'align': 'top','text_wrap': True, 'fg_color': '#FFE265'},
            'bold_total': {'align': 'right', 'bold': True},
            'header': {'bold': True,'bottom': True, 'text_wrap': True},

            # Conditional
            'conditional_red': {'bg_color': '#FFC7CE', 'font_color': '#9C0006'},
            'conditional_green': {'bg_color': '#C6EFCE', 'font_color': '#006100'},
            'conditional_yellow': {'bg_color': '#FFEB9C', 'font_color': '#9C6500'},
            }

        return formats