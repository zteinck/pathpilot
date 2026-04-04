# pathpilot

<div>

[![Package version](https://img.shields.io/pypi/v/pathpilot?color=%2334D058&label=pypi)](https://pypi.org/project/pathpilot/)
[![License](https://img.shields.io/github/license/zteinck/pathpilot)](https://github.com/zteinck/pathpilot/blob/master/LICENSE)

</div>

`pathpilot` is a Python package that makes file and folder manipulation simple and intuitive.


## Installation
```sh
pip install pathpilot
```


## Main Features
- `file_factory` → Function that routes new file instances to the appropriate child class. Some file types are supported natively such as: *.xlsx*, *.csv*, *.txt*, *.parquet*, etc. The mapping of file extensions to their child class counterparts is managed using the `config.extension_mapping` dictionary. Unmapped extensions are routed to the `File` base class by default.
- `Folder` → Class that handles folder operations. It is important to be mindful of the `read_only` parameter which, if `True`, allows folders to be created or deleted.


## Example Usage
Please note the examples below represent a small fraction of the functionality offered by `pathpilot`. Please refer to the intra-code documentation more information.

### Imports
```python
from pathpilot import Folder, file_factory
```

### Folders
Create a `Folder` instance. Passing `read_only=False` will create the folder if it does not already exist.
```python
# initiate a folder instance
folder = Folder(r'C:\Users\MyID\Documents\MyFolder', read_only=False)
```

The `join` method is used to access subfolders. If `read_only=False`, the subfolders are created automatically.
```python
# create subfolders (i.e. C:\Users\MyID\Documents\MyFolder\Year\2025\Month\)
month_folder = folder.join('Year', '2025', 'Month')
```

Alternatively, you can access subfolders by attribute.
```python
# create a new subfolder called "January" by accessing it via attribute
january_folder = month_folder.january
```

Joining to a file name will return a file object instead.
```python
new_years_file = january_folder.join('Happy New Year.txt')
```

### Files
Create an instance of the `ExcelFile` class using the `file_factory` function. This occurs automatically by virtue of the `.xlsx` file extension.
```python
# create ExcelFile instance
file = file_factory(r'C:\Users\MyID\Documents\MyFolder\MyFile.xlsx')
```

Next, let's check if the file exists. If not, let's save a `pandas` `DataFrame` as an Excel file.
```python
# export a pd.DataFrame to the file, if it does not already exist
if not file.exists:
  df = pd.DataFrame({'id': [1, 2, 3], 'data': ['a', 'b', 'c']})
  file.save(df)
```
<pre>
MyFile.xlsx:
    • Wrote 72.00 B to sheet 'Sheet1' in 0.0 seconds.
    • Wrote 80.00 B to sheet 'Sheet1' in 0.0 seconds.
</pre>

Now let's read the file we created as a `DataFrame`.
```python
# read the file we created as a pd.DataFrame
df = file.read()
```

On second thought, let's delete the file.
```python
# delete the file we created
file.delete()
```