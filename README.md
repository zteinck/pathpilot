# pathfinder
`pathfinder` is a library that facilitates file and folder manipulation in Python. It was designed with an emphasis on `pandas` compatibility to ensure smooth workflows.

### Core Utilities
`pathfinder` has two core utilities:
- `File`: Function that assigns new file instances to the correct child class. Many file types are supported natively including: `.xlsx`, `.csv`, `.txt`, `.pickle`, etc. The mapping of file extensions to their respective classes is managed using the `extension_mapping` global dictionary. Unmapped extensions are assigned to the `FileBase` class.
- `Folder`: Class for interacting with folders. It is important to be mindful of the `read_only` parameter which, if set to `True`, allows folders to be created or deleted programically.

## Example Usage
Please note the examples below represent a small fraction of the functionality offered by `pathfinder`. Please refer to the documentation within the code for more information.

### Imports
```python
from pathfinder import Folder, File
```

### Folders
First, we create an instance of the `Folder` class. Passing `read_only=False` causes the folder to be created if it does not already exist.
```python
# initiate a folder instance
folder = Folder(r'C:\Users\MyID\Documents\MyFolder', read_only=False)
```

Moreover, any subfolders that are referenced while interacting with the folder instance will also be created automatically. Let's use the `join` method to create a couple subfolders.
```python
# create subfolders (i.e. C:\Users\MyID\Documents\MyFolder\Year\2025\Month\)
month_folder = folder.join('Year', '2025', 'Month')
```

Alternatively, you can access subfolders by referencing attributes that may or may not already exist.
```python
# create a new subfolder called "January" by accessing it via attribute
january_folder = month_folder.january
```

Joining to a file will return a file object instead.
```python
new_years_file = january_folder.join('Happy New Year.txt')
```

### Files
First, we create an instance of the `ExcelFile` class using the `File` function. This occurs automatically by virtue of the `.xlsx` file extension.
```python
# create ExcelFile instance
file = File(r'C:\Users\MyID\Documents\MyFolder\MyFile.xlsx')
```

Next, let's check if the file exists. If not, let's save a `pandas` `DataFrame` as an Excel file.
```python
# export a pd.DataFrame to the file, if it does not already exist
if not file.exists:
  df = pd.DataFrame({'id': [1, 2, 3], 'data': ['a', 'b', 'c']})
  file.save(df)
```
<pre>
Creating MyFile.xlsx
        writing 72.00 B to 'Sheet1' tab... DONE
        writing 80.00 B to 'Sheet1' tab... DONE
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
