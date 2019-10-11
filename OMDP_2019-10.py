#!/usr/bin/env python

"""
remap ocean output variables for OMDP
"""

from __future__ import print_function
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
from scipy.sparse import csr_matrix

from ocean_remap import ocean_remap

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

def def_var(field_name, fptr_in, fptr_out, dim_names_surf):
    """
    define var in output file, based on varid_in, copying over particular attributes
    """

    varid_in = fptr_in.variables[field_name]

    # construct tuple of dimensions for output variable
    dims_out = ()
    if varid_in.dimensions[0] == 'time':
        dims_out = dims_out + ('time',)
    if 'z_t_150m' in varid_in.dimensions:
        raise TypeError('z_t_150m not allowed in OMDP_2019-10.py')
    if 'z_t' in varid_in.dimensions:
        raise TypeError('z_t not allowed in OMDP_2019-10.py')
    dims_out = dims_out + (dim_names_surf['lat'], dim_names_surf['lon'])

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
    remap ocean output variables for OMDP
    """

    matrix_fnames_per_griddir = {
        'NCAR_POP_HR': 'POP_tx0.1v2_to_latlon_0.25x0.25_0E_mask_conserve_20191009.nc',
        'NCAR_POP_LR': 'POP_gx1v7_to_latlon_1x1_0E_mask_conserve_20181015.nc'
        }

    field_names_per_fname = {
        'ssh_metrics.nc': ('mssh', 'var'),
        'eke_mean_1993-2018.nc': ('eke_mean', ),
        'eke700_mean_1993-2018.nc': ('eke_mean', ),
        'eke1000_mean_1993-2018.nc': ('eke_mean', ),
        'sst_metrics.nc': ('msst', 'var'),
        'sss_metrics.nc': ('msss', 'var'),
        'sit_metrics.nc': ('mean', 'var'),
        'sic_metrics.nc': ('mean', 'var'),
        'thetaot700_metrics.nc': ('mean', 'var'),
        'sot700_metrics.nc': ('mean', 'var'),
        }

    # names of coordinate dimensions in output files
    dim_names_surf = {'lat': 'lat', 'lon': 'lon'}

    for griddir, matrix_2d_fname in matrix_fnames_per_griddir.items():

        matrix_2d = ocean_remap(matrix_2d_fname)

        indir = '/glade/scratch/fredc/OMDP/OMDP_ChassignetEA/' + griddir + '/'
        outdir = '/glade/scratch/klindsay/OMDP/OMDP_ChassignetEA/' + griddir + '/'

        # create CMIP Ofx files
        for var_name in ('areacello', ):
            print('creating Ofx file for '+var_name)
            fptr_out = nc.Dataset(outdir+'outfile_'+var_name+'.nc', 'w') # pylint: disable=E1101
            matrix_2d.dst_grid.def_dims_common(fptr_out, dim_names_surf)
            matrix_2d.dst_grid.write_vars_common(fptr_out, dim_names_surf)
            matrix_2d.dst_grid.write_var_CMIP_Ofx(fptr_out, dim_names_surf, var_name)

        for fname, field_names in field_names_per_fname.items():
            in_fname = indir + fname
            out_fname = outdir + fname

            fptr_in = nc.Dataset(in_fname, 'r') # pylint: disable=E1101
            fptr_out = nc.Dataset(out_fname, 'w') # pylint: disable=E1101

            # copy_time(fptr_in, fptr_out)

            matrix_2d.dst_grid.def_dims_common(fptr_out, dim_names_surf)
            matrix_2d.dst_grid.write_vars_common(fptr_out, dim_names_surf)

            for field_name in field_names:
                print(field_name)

                varid_out = def_var(field_name, fptr_in, fptr_out, dim_names_surf)

                varid_out[:] = matrix_2d.remap_var(fptr_in.variables[field_name][:],
                                                   fill_value=getattr(varid_out, '_FillValue'))

            fptr_in.close()
            fptr_out.close()

if __name__ == '__main__':
    main()
