#!/usr/bin/env python

"""
remap ocean output variables (2D or 3D)
"""

from __future__ import print_function
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
from scipy.sparse import csr_matrix

class ocean_remap(object):
    """
    class for remapping ocean output variable (2D or 3D)
    """

    def __init__(self, fname):
        """
        Initialize ocean_remap object by loading sparse remapping matrix.
        """

        fptr = nc.Dataset(fname, 'r') # pylint: disable=E1101

        self.src_grid = ocean_remap_grid(fptr, lsrc_grid=True)
        self.dst_grid = ocean_remap_grid(fptr, lsrc_grid=False)

        # SCRIP matrices use 1-based indexing, so subtract 1 from col and row indices
        row = fptr.variables['row'][:] - 1
        col = fptr.variables['col'][:] - 1
        S = fptr.variables['S'][:] # pylint: disable=C0103

        fptr.close()

        src_grid_size = self.src_grid.dims.prod()
        dst_grid_size = self.dst_grid.dims.prod()

        self.matrix = csr_matrix((S, (row, col)), shape=(dst_grid_size, src_grid_size))

    def remap_var(self, src_var, fill_value=1.e36):
        """
        Remap src_var, using matrix vector product, returning dst_var.
        """

        # flatten spatial dimensions in new view (shallow copy) of src_var,
        # preserving leading non-spatial dimensions
        # this is done to enable regridding via matrix multiply
        src_var_loc = src_var.view()
        src_grid_ndim = self.src_grid.dims.size
        if src_var.ndim < src_grid_ndim:
            raise TypeError('src_var ndim too small')
        if not all(np.array(src_var.shape[-src_grid_ndim:]) == self.src_grid.dims):
            raise TypeError('right-most dimension sizes of src_var do not match matrix')
        src_var_loc.shape = src_var.shape[0:-src_grid_ndim] + (self.src_grid.dims.prod(),)

        ones_thres = 1.0e-6

        # perform regridding via matrix multiply (dot),
        # iterating over leading non-spatial dimensions
        # set mask_b!=1 vals to fill_value
        if src_var_loc.ndim == 1:
            if isinstance(src_var_loc, ma.MaskedArray):
                src_ones = np.where(ma.getmaskarray(src_var_loc), 0.0, 1.0)
            else:
                src_ones = np.where(src_var_loc == fill_value, 0.0, 1.0)
            dst_ones = self.matrix.dot(src_ones)
            # only perform regridding with normaliztion if array of remapped 1.0's
            # differs significantly from 1.0
            if max(np.abs(np.where(dst_ones > ones_thres, dst_ones-1.0, 0.0))) > ones_thres:
                dst_var = self.matrix.dot(np.where(src_ones == 1.0, src_var_loc, 0.0))
                dst_var = np.divide(dst_var, dst_ones, where=(dst_ones > ones_thres))
                dst_var = np.where(
                    (self.dst_grid.mask == 1) & (dst_ones > ones_thres), dst_var, fill_value)
            else:
                dst_var = self.matrix.dot(src_var_loc)
                dst_var = np.where(self.dst_grid.mask == 1, dst_var, fill_value)
        elif src_var_loc.ndim == 2:
            # handle 1 extra leading dimension (e.g., time, ensemble member)
            dst_var = np.empty((src_var_loc.shape[0], self.dst_grid.dims.prod()))
            for dim0 in range(0, src_var_loc.shape[0]):
                if isinstance(src_var_loc, ma.MaskedArray):
                    src_ones = np.where(ma.getmaskarray(src_var_loc[dim0, :]), 0.0, 1.0)
                else:
                    src_ones = np.where(src_var_loc[dim0, :] == fill_value, 0.0, 1.0)
                dst_ones = self.matrix.dot(src_ones)
                # only perform regridding with normaliztion if array of remapped 1.0's
                # differs significantly from 1.0
                if max(np.abs(np.where(dst_ones > ones_thres, dst_ones-1.0, 0.0))) > ones_thres:
                    dst_var[dim0, :] = self.matrix.dot(
                        np.where(src_ones == 1.0, src_var_loc[dim0, :], 0.0))
                    dst_var[dim0, :] = np.divide(
                        dst_var[dim0, :], dst_ones, where=(dst_ones > ones_thres))
                    dst_var[dim0, :] = np.where(
                        (self.dst_grid.mask == 1) & (dst_ones > ones_thres), dst_var[dim0, :],
                        fill_value)
                else:
                    dst_var[dim0, :] = self.matrix.dot(src_var_loc[dim0, :])
                    dst_var[dim0, :] = np.where(
                        self.dst_grid.mask == 1, dst_var[dim0, :], fill_value)
        elif src_var_loc.ndim == 3:
            # handle 2 extra leading dimensions (e.g., time, ensemble member)
            dst_var = np.empty((src_var_loc.shape[0], src_var_loc.shape[1],
                                self.dst_grid.dims.prod()))
            for dim0 in range(0, src_var_loc.shape[0]):
                for dim1 in range(0, src_var_loc.shape[1]):
                    if isinstance(src_var_loc, ma.MaskedArray):
                        src_ones = np.where(ma.getmaskarray(src_var_loc[dim0, dim1, :]), 0.0, 1.0)
                    else:
                        src_ones = np.where(src_var_loc[dim0, dim1, :] == fill_value, 0.0, 1.0)
                    dst_ones = self.matrix.dot(src_ones)
                    # only perform regridding with normaliztion if array of remapped 1.0's
                    # differs significantly from 1.0
                    if max(np.abs(np.where(dst_ones > ones_thres, dst_ones-1.0, 0.0))) > ones_thres:
                        dst_var[dim0, dim1, :] = self.matrix.dot(
                            np.where(src_ones == 1.0, src_var_loc[dim0, dim1, :], 0.0))
                        dst_var[dim0, dim1, :] = np.divide(
                            dst_var[dim0, dim1, :], dst_ones, where=(dst_ones > ones_thres))
                        dst_var[dim0, dim1, :] = np.where(
                            (self.dst_grid.mask == 1) & (dst_ones > ones_thres),
                            dst_var[dim0, dim1, :], fill_value)
                    else:
                        dst_var[dim0, dim1, :] = self.matrix.dot(src_var_loc[dim0, dim1, :])
                        dst_var[dim0, dim1, :] = np.where(
                            self.dst_grid.mask == 1, dst_var[dim0, dim1, :], fill_value)
        else:
            raise TypeError('too many extra dimensions in src_var')

        # unflatten dst_var
        dst_var.shape = dst_var.shape[0:-1] + tuple(self.dst_grid.dims)

        return dst_var

class ocean_remap_grid(object):
    """
    class for storing grid info used by ocean_remap
    """

    def __init__(self, fptr, lsrc_grid):
        """
        Initialize ocean_remap_grid object from open SCRIP file
        """

        # SCRIP dimensions are reversed relative to python, so reverse dimensions
        if lsrc_grid:
            self.dims = fptr.variables['src_grid_dims'][::-1]
        else:
            self.dims = fptr.variables['dst_grid_dims'][::-1]
            self.__read_vars(fptr, var_name_suffix='_b')

    def __read_vars(self, fptr, var_name_suffix):

        lon = fptr.variables['xc'+var_name_suffix][:]
        lon.shape = self.dims[-2:] # reshape to 2D
        # store full 2d lon if it varies from column to column
        lon0 = lon[0, :]
        if np.any(lon-lon0[np.newaxis, :]):
            self.lon = lon
        else:
            self.lon = lon0

        lat = fptr.variables['yc'+var_name_suffix][:]
        lat.shape = self.dims[-2:] # reshape to 2D
        # store full 2d lat if it varies from row to row
        lat0 = lat[:, 0]
        if np.any(lat-lat0[:, np.newaxis]):
            self.lat = lat
        else:
            self.lat = lat0

        # depth related vars are only available for mapping files for 3D fields
        if self.dims.size == 3:
            self.depth = fptr.variables['zc'+var_name_suffix][:]
            self.depth_bnds = fptr.variables['zc_bnds'+var_name_suffix][:]

        self.area = fptr.variables['area'+var_name_suffix][:]
        self.mask = fptr.variables['mask'+var_name_suffix][:]

    def def_dims_common(self, fptr_out, dim_names):
        """
        Define common dimensions in an open netCDF4 file
        """

        dims_dict = {}
        if self.dims.size == 3:
            dims_dict[dim_names['depth']] = self.dims[0]
            dims_dict['d2'] = 2
        dims_dict[dim_names['lat']] = self.dims[-2]
        dims_dict[dim_names['lon']] = self.dims[-1]

        for dimname, dimlen in dims_dict.items():
            if not dimname in fptr_out.dimensions.keys():
                fptr_out.createDimension(dimname, dimlen)

    def write_vars_common(self, fptr_out, dim_names):
        """
        Define and write common vars to an open netCDF4 file
        """

        lat_name = dim_names['lat']
        lon_name = dim_names['lon']

        # depth and depth_bnds
        if self.dims.size == 3:
            depth_name = dim_names['depth']
            if not depth_name in fptr_out.variables.keys():
                varid = fptr_out.createVariable(depth_name, 'f8', (depth_name,))
                varid.axis = 'Z'
                varid.bounds = depth_name+'_bnds'
                varid.long_name = 'ocean model level'
                varid.positive = 'down'
                varid.standard_name = 'olevel'
                varid.units = 'm'
                varid[:] = self.depth

            if not depth_name+'_bnds' in fptr_out.variables.keys():
                varid = fptr_out.createVariable(depth_name+'_bnds', 'f8', (depth_name, 'd2'))
                varid.long_name = 'depth bounds'
                varid.units = 'm'
                varid[:] = self.depth_bnds

        # latitude
        if not lat_name in fptr_out.variables.keys():
            if self.lat.ndim == 1:
                varid = fptr_out.createVariable(lat_name, 'f8', (lat_name,))
            else:
                varid = fptr_out.createVariable(lat_name, 'f8', (lat_name, lon_name))
            if self.lat.ndim == 1:
                varid.axis = 'Y'
                varid.bounds = lat_name+'_bnds'
            varid.long_name = 'latitude'
            varid.standard_name = 'latitude'
            varid.units = 'degrees_north'
            varid[:] = self.lat

        # latitude bounds, currently only supported when latitude is 1d
        if self.lat.ndim == 1 and not lat_name+'_bnds' in fptr_out.variables.keys():
            varid = fptr_out.createVariable(lat_name+'_bnds', 'f8', (lat_name, 'd2'))
            varid.long_name = 'latitude bounds'
            varid.units = 'degrees_north'
            varid[:] = _var_bnds_1d(self.lat, lextrap=True)

        # longitude
        if not lon_name in fptr_out.variables.keys():
            if self.lon.ndim == 1:
                varid = fptr_out.createVariable(lon_name, 'f8', (lon_name,))
            else:
                varid = fptr_out.createVariable(lon_name, 'f8', (lat_name, lon_name))
            if self.lon.ndim == 1:
                varid.axis = 'X'
                varid.bounds = lon_name+'_bnds'
            varid.long_name = 'longitude'
            varid.standard_name = 'longitude'
            varid.units = 'degrees_east'
            varid[:] = self.lon

        # longitude bounds, currently only supported when longitude is 1d
        if self.lon.ndim == 1 and not lon_name+'_bnds' in fptr_out.variables.keys():
            varid = fptr_out.createVariable(lon_name+'_bnds', 'f8', (lon_name, 'd2'))
            varid.long_name = 'longitude bounds'
            varid.units = 'degrees_east'
            varid[:] = _var_bnds_1d(self.lon, lextrap=True)

    def write_var_CMIP_Ofx(self, fptr_out, dim_names, var_name):
        """
        Define and write CMIP Ofx var to an open netCDF4 file
        Also set global attribute external_variables from cell_meassures, if appropriate
        """

        area = self.area.copy()
        area.shape = tuple(self.dims[-2:])
        # convert from radian^2 to m^2
        rearth = 6.37122e6
        area = rearth**2 * area

        if var_name == 'areacello':
            dims_out = (dim_names['lat'], dim_names['lon'])
            varid = fptr_out.createVariable(var_name, 'f8', dims_out)
            varid.cell_methods = 'area: sum'
            varid.long_name = 'Grid-Cell Area'
            varid.standard_name = 'cell_area'
            varid.units = 'm2'
            varid[:] = area
            return

        if var_name == 'deptho':
            dims_out = (dim_names['lat'], dim_names['lon'])
            varid = fptr_out.createVariable(var_name, 'f8', dims_out)
            varid.cell_measures = 'area: areacello'
            varid.cell_methods = 'area: mean where sea'
            varid.long_name = 'Sea Floor Depth Below Geoid'
            varid.standard_name = 'sea_floor_depth_below_geoid'
            varid.units = 'm'
            mask3d = self.mask.view()
            mask3d.shape = tuple(self.dims)
            num_active_layers = np.count_nonzero(mask3d, axis=0)
            varid[:] = np.where(num_active_layers > 0, self.depth_bnds[num_active_layers-1, 1], 0.0)
            fptr_out.external_variables = 'areacello'
            return

        thkcello_1d = self.depth_bnds[:, 1] - self.depth_bnds[:, 0]

        if var_name == 'thkcello':
            dims_out = (dim_names['depth'], dim_names['lat'], dim_names['lon'])
            varid = fptr_out.createVariable(var_name, 'f8', dims_out)
            varid.cell_measures = 'area: areacello volume: volcello'
            varid.cell_methods = 'area: mean'
            varid.long_name = 'Ocean Model Cell Thickness'
            varid.standard_name = 'cell_thickness'
            varid.units = 'm'
            varid[:] = thkcello_1d[:, np.newaxis, np.newaxis] * np.ones(self.dims)
            fptr_out.external_variables = 'areacello volcello'
            return

        if var_name == 'volcello':
            dims_out = (dim_names['depth'], dim_names['lat'], dim_names['lon'])
            varid = fptr_out.createVariable(var_name, 'f8', dims_out)
            varid.cell_measures = 'area: areacello volume: volcello'
            varid.cell_methods = 'area: mean'
            varid.long_name = 'Ocean Grid-Cell Volume'
            varid.standard_name = 'ocean_volume'
            varid.units = 'm3'
            varid[:] = thkcello_1d[:, np.newaxis, np.newaxis] * area
            fptr_out.external_variables = 'areacello volcello'
            return

def _var_bnds_1d(var, lextrap):
    """
    Compute bounds for a 1d variable.
    Assumes bounds are midpoints of provided values.
    Extrapolates at endpoints if lextrap==True
    """
    var_bnds = np.empty((var.size, 2))
    dvar = np.diff(var)
    if lextrap:
        var_bnds[0, 0] = var[0] - 0.5 * dvar[0]
    else:
        var_bnds[0, 0] = var[0]
    var_bnds[1:, 0] = var[:-1] + 0.5 * dvar
    var_bnds[:-1, 1] = var_bnds[1:, 0]
    if lextrap:
        var_bnds[-1, 1] = var[-1] + 0.5 * dvar[-1]
    else:
        var_bnds[-1, 1] = var[-1]

    return var_bnds

def copy_time(fptr_in, fptr_out):
    """
    copy time dimension and var from one netCDF4 file to another
    """

    fptr_out.createDimension('time', None)
    fptr_out.createDimension('d2', 2)

    varid_in = fptr_in.variables['time']
    varid_out = fptr_out.createVariable('time', 'f8', ('time'))
    for att_name in ('bounds', 'calendar', 'long_name', 'units'):
        setattr(varid_out, att_name, getattr(varid_in, att_name))
    varid_out[:] = varid_in[:]

    varid_in = fptr_in.variables['time_bound']
    varid_out = fptr_out.createVariable('time_bound', 'f8', ('time', 'd2'))
    for att_name in ('long_name', 'units'):
        setattr(varid_out, att_name, getattr(varid_in, att_name))
    varid_out[:] = varid_in[:]

def def_var(field_name, fptr_in, fptr_out, dim_names_partial, dim_names_full):
    """
    define var in output file, based on varid_in, copying over particular attributes
    """

    varid_in = fptr_in.variables[field_name]

    # construct tuple of dimensions for output variable
    dims_out = ()
    if varid_in.dimensions[0] == 'time':
        dims_out = dims_out + ('time',)
    if 'z_t_150m' in varid_in.dimensions:
        dims_out = dims_out + (dim_names_partial['depth'],)
    if 'z_t' in varid_in.dimensions:
        dims_out = dims_out + (dim_names_full['depth'],)
    dims_out = dims_out + (dim_names_full['lat'], dim_names_full['lon'])

    # create output variable, using _FillValue of input variable
    varid_out = fptr_out.createVariable(field_name, varid_in.datatype, dims_out,
                                        fill_value=getattr(varid_in, '_FillValue'))

    # copy particular attributes, if present
    for att_name in ('long_name', 'units', 'cell_methods', 'scale_factor', 'missing_value'):
        try:
            setattr(varid_out, att_name, getattr(varid_in, att_name))
            print('    '+att_name+' copied')
        except AttributeError:
            pass

    return varid_out

def main():
    """
    example usage of ocean_remap class
    """

    matrix_2d_fname = 'POP_gx1v7_to_latlon_1x1_0E_mask_conserve_20181015.nc'
    matrix_2d = ocean_remap(matrix_2d_fname)

    matrix_3d_partial_fname = 'POP_gx1v7_to_latlon_1x1_0E_partialdepth_conserve_20190121.nc'
    matrix_3d_partial = ocean_remap(matrix_3d_partial_fname)

    matrix_3d_full_fname = 'POP_gx1v7_to_latlon_1x1_0E_fulldepth_conserve_20181015.nc'
    matrix_3d_full = ocean_remap(matrix_3d_full_fname)

    # names of coordinate dimensions in output files
    dim_names_partial = {'depth': 'lev_partial', 'lat': 'lat', 'lon': 'lon'}
    dim_names_full = {'depth': 'lev', 'lat': 'lat', 'lon': 'lon'}

    outdir = './'

    # create CMIP Ofx files
    for var_name in ('areacello', 'deptho', 'thkcello', 'volcello'):
        print('creating Ofx file for '+var_name)
        fptr_out = nc.Dataset(outdir+'outfile_'+var_name+'.nc', 'w') # pylint: disable=E1101
        matrix_3d_full.dst_grid.def_dims_common(fptr_out, dim_names_full)
        matrix_3d_full.dst_grid.write_vars_common(fptr_out, dim_names_full)
        matrix_3d_full.dst_grid.write_var_CMIP_Ofx(fptr_out, dim_names_full, var_name)

    testfile_in_fname = 'infile.nc'
    testfile_out_fname = outdir+'outfile.nc'
    field_names = ('HT', 'REGION_MASK', 'SSH', 'SHF', 'POC_FLUX_100m', 'diatChl', 'SALT')
    # field_names = ('HT', 'REGION_MASK')

    fptr_in = nc.Dataset(testfile_in_fname, 'r') # pylint: disable=E1101
    fptr_in.set_auto_scale(False)
    fptr_out = nc.Dataset(testfile_out_fname, 'w') # pylint: disable=E1101

    copy_time(fptr_in, fptr_out)

    matrix_3d_partial.dst_grid.def_dims_common(fptr_out, dim_names_partial)
    matrix_3d_partial.dst_grid.write_vars_common(fptr_out, dim_names_partial)

    matrix_3d_full.dst_grid.def_dims_common(fptr_out, dim_names_full)
    matrix_3d_full.dst_grid.write_vars_common(fptr_out, dim_names_full)

    for field_name in field_names:
        print(field_name)

        varid_out = def_var(field_name, fptr_in, fptr_out, dim_names_partial, dim_names_full)
        varid_out.set_auto_scale(False)

        # use appropriate matrix for regridding
        if dim_names_full['depth'] in varid_out.dimensions:
            varid_out[:] = matrix_3d_full.remap_var(fptr_in.variables[field_name][:],
                                                    fill_value=getattr(varid_out, '_FillValue'))
        elif dim_names_partial['depth'] in varid_out.dimensions:
            varid_out[:] = matrix_3d_partial.remap_var(fptr_in.variables[field_name][:],
                                                       fill_value=getattr(varid_out, '_FillValue'))
        else:
            varid_out[:] = matrix_2d.remap_var(fptr_in.variables[field_name][:],
                                               fill_value=getattr(varid_out, '_FillValue'))

    fptr_in.close()
    fptr_out.close()

if __name__ == '__main__':
    main()
