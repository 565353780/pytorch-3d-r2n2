#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Voxels(object):
    """ Holds a binvox model.
    data is either a three-dimensional numpy boolean array (dense representation)
    or a two-dimensional numpy float array (coordinate representation).

    dims, translate and scale are the model metadata.

    dims are the voxel dimensions, e.g. [32, 32, 32] for a 32x32x32 model.

    scale and translate relate the voxels to the original model coordinates.

    To translate voxel coordinates i, j, k to original coordinates x, y, z:

    x_n = (i+.5)/dims[0]
    y_n = (j+.5)/dims[1]
    z_n = (k+.5)/dims[2]
    x = scale*x_n + translate[0]
    y = scale*y_n + translate[1]
    z = scale*z_n + translate[2]

    """

    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)

    def write(self, fp):
        """ Write binary binvox format.

        Note that when saving a model in sparse (coordinate) format, it is first
        converted to dense format.

        Doesn't check if the model is 'sane'.

        """
        if self.data.ndim == 2:
            # TODO avoid conversion to dense
            dense_voxel_data = sparse_to_dense(self.data, self.dims)
        else:
            dense_voxel_data = self.data

        fp.write('#binvox 1\n')
        fp.write('dim ' + ' '.join(map(str, self.dims)) + '\n')
        fp.write('translate ' + ' '.join(map(str, self.translate)) + '\n')
        fp.write('scale ' + str(self.scale) + '\n')
        fp.write('data\n')
        if self.axis_order not in ('xzy', 'xyz'):
            raise ValueError('Unsupported voxel model axis order')

        if self.axis_order == 'xzy':
            voxels_flat = dense_voxel_data.flatten()
        elif self.axis_order == 'xyz':
            voxels_flat = np.transpose(dense_voxel_data, (0, 2, 1)).flatten()

        # keep a sort of state machine for writing run length encoding
        state = voxels_flat[0]
        ctr = 0
        for c in voxels_flat:
            if c == state:
                ctr += 1
                # if ctr hits max, dump
                if ctr == 255:
                    fp.write(chr(state))
                    fp.write(chr(ctr))
                    ctr = 0
            else:
                # if switch state, dump
                fp.write(chr(state))
                fp.write(chr(ctr))
                state = c
                ctr = 1
        # flush out remainders
        if ctr > 0:
            fp.write(chr(state))
            fp.write(chr(ctr))
        return True
