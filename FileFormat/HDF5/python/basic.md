# HDF5 基础

- [HDF5 基础](#hdf5-基础)
  - [Create dataset](#create-dataset)
  - [Read and write to a dataset](#read-and-write-to-a-dataset)
  - [Create an attribute](#create-an-attribute)
  - [Create a group](#create-a-group)

## Create dataset

创建 HDF5 文件 dset.h5，包含一个空的 datasets `/dset`。

```python
#
# This example creates an HDF5 file dset.h5 and an empty datasets /dset in it.
#
import h5py

# Create a new file using default properties.
file = h5py.File('dset.h5', 'w')

# Create a dataset under the Root group.
dataset = file.create_dataset("dset", (4, 6), h5py.h5t.STD_I32BE)
print("Dataset dataspace is", dataset.shape)
print("Dataset Numpy datatype is", dataset.dtype)
print("Dataset name is", dataset.name)
print("Dataset is a member of the group", dataset.parent)
print("Dataset was created in the file", dataset.file)

# Close the file before exiting
file.close()
```

```txt
Dataset dataspace is (4, 6)
Dataset Numpy datatype is >i4
Dataset name is /dset
Dataset is a member of the group <HDF5 group "/" (1 members)>
Dataset was created in the file <HDF5 file "dset.h5" (mode r+)>
```

## Read and write to a dataset

```python
#
# This example writes data to the existing empty dataset created by h5_crtdat.py and then reads it back.
#
import h5py
import numpy as np

# Open an existing file using default properties.
file = h5py.File('dset.h5', 'r+')

# Open "dset" dataset under the root group.
dataset = file['/dset']

# Initialize data object with 0.
data = np.zeros((4, 6))

# Assign new values
for i in range(4):
    for j in range(6):
        data[i][j] = i * 6 + j + 1

# Write data
print("Writing data...")
dataset[...] = data

# Read data back and print it.
print("Reading data back...")
data_read = dataset[...]
print("Printing data...")
print(data_read)

# Close the file before exiting
file.close()
```

```txt
Writing data...
Reading data back...
Printing data...
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]
 [13 14 15 16 17 18]
 [19 20 21 22 23 24]]
```

## Create an attribute

## Create a group

```python
#
# This example creates an HDF5 file group.h5 and a group MyGroup in it
# using H5Py interfaces to the HDF5 library.
#
import h5py

# Use 'w' to remove existing file and create a new one; use 'w-' if
# create operation should fail when the file already exists.
print("Creating an HDF5 file with the name group.h5...")
file = h5py.File('group.h5', 'w')

# Show the Root group which is created when the file is created.
print("When an HDF5 file is created, it has a Root group with the name '", file.name, "'.")

# Create a group with the name "MyGroup"
print("Creating a group MyGroup in the file...")
group = file.create_group("MyGroup")

# Print the content of the Root group
print("An HDF5 group is a container for other objects; a group is similar to Python dictionary with the keys being the "
      "links to the group members.")
print("Show the members of the Root group using dictionary key method:", list(file.keys()))

# Another way to show the content of the Root group.
print("Show the members of the Root group using the list function:", list(file))

# Close the file before exiting; H5Py will close the group.
file.close()
```

```txt
Creating an HDF5 file with the name group.h5...
When an HDF5 file is created, it has a Root group with the name ' / '.
Creating a group MyGroup in the file...
An HDF5 group is a container for other objects; a group is similar to Python dictionary with the keys being the links to the group members.
Show the members of the Root group using dictionary key method: ['MyGroup']
Show the members of the Root group using the list function: ['MyGroup']
```
