import sqlite3
import os
import pandas as pd
import numpy as np
from iterlab import to_iter, lower_iter

from ._file import FileBase



class SQLiteFile(FileBase):

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, f, **kwargs):
        super().__init__(f, **kwargs)


    #╭-------------------------------------------------------------------------╮
    #| Static Methods                                                          |
    #╰-------------------------------------------------------------------------╯

    @staticmethod
    def insert_query(tbl_name, col_names):
        sql = 'INSERT OR IGNORE INTO {0} ({1}) VALUES ({2})'.format(
            tbl_name,
            ','.join('[%s]' % x for x in col_names),
            ','.join('?' for x in range(len(col_names)))
            )
        return sql


    @staticmethod
    def update_query(tbl_name, update_cols, where_cols, where_logic=''):
        sql = 'UPDATE {0} SET {1} WHERE {2} {3}'.format(
            tbl_name,
            ', '.join('[%s] = ?' % x for x in to_iter(update_cols)),
            ' AND '.join('[%s] = ?' % x for x in to_iter(where_cols)),
            where_logic
            )
        return sql


    @staticmethod
    def select_query(tbl_name, select_cols, where_cols, where_logic=''):
        sql = 'SELECT {1} FROM {0} WHERE {2} {3}'.format(
            tbl_name,
            ', '.join('[%s]' % x for x in to_iter(select_cols)),
            ' AND '.join('[%s] = ?' % x for x in to_iter(where_cols)),
            where_logic
            )
        return sql


    @staticmethod
    def delete_query(tbl_name, where_cols, where_logic=''):
        sql = 'DELETE FROM {0} WHERE {1} {2}'.format(
            tbl_name,
            ' AND '.join('[%s] = ?' % x for x in to_iter(where_cols)),
            where_logic
            )
        return sql


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def delete(self, reconnect=False):
        if self.exists: os.remove(self.path)
        if reconnect: self.connect()


    def connect(self):
        ''' connect to the database '''
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.c = self.conn.cursor()


    def disconnect(self):
        ''' disconnect from the database '''
        self.conn.close()
        del self.conn
        del self.c


    def enable_foreign_keys(self):
        self.c.execute('PRAGMA foreign_keys = ON;')
        self.conn.commit()


    def tables(self):
        self.c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return list(x[0] for x in self.c.fetchall())


    def columns(self, tbl_name):
        return self.table_info(tbl_name).name.tolist()


    def table_info(self, tbl_name):
        return self.read_sql('PRAGMA table_info(%s)' % tbl_name.split('.')[-1])


    def data_types(self, tbl_name):
        mapping = {'NULL': None, 'INTEGER': int, 'REAL': float, 'TEXT': str, 'BLOB': object, 'INT': int}
        return {k.lower(): mapping[v] for k,v in self.table_info(tbl_name)[['name','type']].values}


    def drop(self, tbl_name):
        self.c.execute('DROP TABLE %s' % tbl_name)
        self.conn.commit()


    def clear(self, tbl_name):
        self.c.execute('DELETE FROM %s' % tbl_name)
        self.conn.commit()


    def vacuum(self):
        self.c.execute('VACUUM')
        self.conn.commit()


    def read_sql(self, sql, params=None, **kwargs):
        if len(sql.split()) == 1: sql = "SELECT * FROM %s" % sql
        return pd.read_sql(sql, con=self.conn, params=params)


    def column_to_set(self, tbl_name, col_name):
        return set(x[0] for x in self.c.execute("SELECT DISTINCT {1} FROM {0}".format(tbl_name,col_name)).fetchall())


    def execute(self, sql, params=()):
        self.c.execute(sql, params)
        self.conn.commit()


    def format_payload(self, tbl_name, col_names, payload):
        if isinstance(payload, tuple): payload = [payload]
        mapping = self.data_types(tbl_name)
        payload = [tuple([mapping[col](cell) if pd.notnull(cell) else None
                          for col,cell in zip(col_names, row)]) for row in payload]
        return payload


    def insert(self, tbl_name, payload, col_names=None, clear=False):
        if col_names is None: col_names = self.columns(tbl_name)
        sql = self.insert_query(tbl_name, col_names)
        payload = self.format_payload(tbl_name, col_names, payload)
        if clear: self.clear(tbl_name)
        self.c.executemany(sql,payload)
        self.conn.commit()


    def df_to_table(self, tbl_name, df, chunksize=0, clear_tbl=False, where_cols=None, where_logic=''):
        ''' performs bulk insert of dataframe; bulk update occurs if where_cols argument is passed '''

        def to_table(df):
            payload = self.format_payload(tbl_name, col_names, df.values)
            self.c.executemany(sql, payload)
            self.conn.commit()

        if clear_tbl: self.clear(tbl_name)

        if list(filter(None, list(df.index.names))): df.reset_index(inplace=True)
        df.rename(columns={x: x.lower() for x in df.columns}, inplace=True)
        col_names = [x for x in lower_iter(self.columns(tbl_name)) if x in set(list(df.columns))]
        df = df[col_names]
        if df.empty: raise RuntimeError('Dataframe and %s table have no columns in common' % tbl_name)

        if where_cols:
            where_cols = lower_iter(to_iter(where_cols))
            col_names = [x for x in col_names if x not in where_cols]
            sql = self.update_query(tbl_name, col_names, where_cols, where_logic)
            col_names.extend(where_cols)
            df = df[col_names]
        else:
            sql = self.insert_query(tbl_name,col_names)

        if chunksize:
            df = df.groupby(np.arange(len(df)) // chunksize)
            for key,chunk in df: to_table(chunk)
        else:
            to_table(df)


    def copy_as_temp(self, target_name, temp_name=None, index=None):
        ''' creates temporary table based on permanent table '''
        temp_name = temp_name or target_name
        sql = 'CREATE TEMP TABLE {0} AS SELECT * FROM {1} LIMIT 0;'.format(temp_name,target_name)
        self.c.execute(sql)
        self.conn.commit()
        if index:
            sql = 'CREATE INDEX temp.idx ON {0}({1})'.format(temp_name,','.join(to_iter(index)))
            self.c.execute(sql)
            self.conn.commit()


    def clear_tables(self, warn=True):
        response = input(f'Clear all {self.name}.sqlite tables (y/n)? this action cannot be undone: ') if warn else 'y'
        if response.lower() == 'y':
            for x in self.tables():
                self.clear(x)