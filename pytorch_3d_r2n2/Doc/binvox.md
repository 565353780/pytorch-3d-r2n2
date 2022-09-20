#Binvox to Numpy and back.

```bash
>>> import numpy as np
>>> import pytorch_3d_r2n2.Method.binvox as binvox_rw
```

```bash
>>> with open('chair.binvox', 'rb') as f:
...     m1 = binvox_rw.read_as_3d_array(f)
```

```bash
>>> m1.dims
[32, 32, 32]
```

```bash
>>> m1.scale
41.133000000000003
```

```bash
>>> m1.translate
[0.0, 0.0, 0.0]
```

```bash
>>> with open('chair_out.binvox', 'wb') as f:
...     m1.write(f)
```

```bash
>>> with open('chair_out.binvox', 'rb') as f:
...     m2 = binvox_rw.read_as_3d_array(f)
```

```bash
>>> m1.dims==m2.dims
True
```

```bash
>>> m1.scale==m2.scale
True
```

```bash
>>> m1.translate==m2.translate
True
```

```bash
>>> np.all(m1.data==m2.data)
True
```

```bash
>>> with open('chair.binvox', 'rb') as f:
...     md = binvox_rw.read_as_3d_array(f)
```

```bash
>>> with open('chair.binvox', 'rb') as f:
...     ms = binvox_rw.read_as_coord_array(f)
```

```bash
>>> data_ds = binvox_rw.dense_to_sparse(md.data)
>>> data_sd = binvox_rw.sparse_to_dense(ms.data, 32)
>>> np.all(data_sd==md.data)
True
```

```bash
>>> # the ordering of elements returned by numpy.nonzero changes with axis
>>> # ordering, so to compare for equality we first lexically sort the voxels.
>>> np.all(ms.data[:, np.lexsort(ms.data)] == data_ds[:, np.lexsort(data_ds)])
True
```

