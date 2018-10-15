;
; gen_surf_weights.ncl
;
; remove dst point whose coverage is less than frac_thres
;

begin
  datestr = systemfunc("date +%Y%m%d")

  fname_in  = "../weights_files/POP_gx1v7_to_latlon_1x1_0E_conserve_20180914.nc"
  fname_out = "POP_gx1v7_to_latlon_1x1_0E_mask_conserve_"+datestr+".nc"

  frac_thres = 0.525
  frac_thres!0 = "ncl_scalar"

  f_in  = addfile(fname_in, "r")

  setfileoption("nc", "Format", "NetCDF4")
  setfileoption("nc", "CompressionLevel", 1)

  print("writing "+fname_out)
  system("rm -f "+fname_out)
  f_out = addfile(fname_out, "c")

; cp_var_names = (/ "src_grid_dims", "dst_grid_dims", \
;                   "xc_a", "yc_a", "xv_a", "yv_a", "mask_a", "area_a", "frac_a", \
;                   "xc_b", "yc_b", "xv_b", "yv_b", "area_b" /)

  cp_var_names = (/ "src_grid_dims", "dst_grid_dims", \
                    "xc_b", "yc_b", "area_b" /)

  do var_ind = 0, dimsizes(cp_var_names)-1
    var_name = cp_var_names(var_ind)
    print("copying "+var_name)
    f_out->$var_name$ = f_in->$var_name$
  end do

  frac_b = f_in->frac_b
  mask_b = f_in->mask_b

  ; remove dst point whose coverage is less than frac_thres

  land_b_ind       = ind(frac_b .le. frac_thres)
  ocean_matrix_ind = ind(frac_b(f_in->row-1) .gt. frac_thres)

  frac_b(land_b_ind) = 0
  mask_b(land_b_ind) = 0
; f_out->frac_b = frac_b
  f_out->mask_b = mask_b

  col = f_in->col(ocean_matrix_ind)
  row = f_in->row(ocean_matrix_ind)
  S   = f_in->S(ocean_matrix_ind)

  f_out->col = col
  f_out->row = row
  f_out->S   = S

  copy_VarAtts(f_in, f_out)
  f_out@history = "matrix entries corresponding to frac_b .le. frac_thres removed, "+datestr
  f_out->frac_thres = frac_thres
end
