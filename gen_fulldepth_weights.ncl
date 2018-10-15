;
; gen_fulldepth_weights.ncl
;
; remove dst point whose coverage is less than frac_thres
;

begin
  datestr = systemfunc("date +%Y%m%d")

  fname_out = "POP_gx1v7_to_latlon_1x1_0E_fulldepth_conserve_"+datestr+".nc"

  frac_thres = 0.525
  frac_thres!0 = "ncl_scalar"

  fname_vert_grid_in = "../grid_files/POP_gx1v7_vert_20161220.nc"
  f_vert_grid_in = addfile(fname_vert_grid_in, "r")
  depth_in = todouble(f_vert_grid_in->depth)
  depth_in@units = "m"
  nlev_in = dimsizes(depth_in)
  ; construct depth_bnds_in as cummulative sum of dz_in
  dz_in = todouble(f_vert_grid_in->dz)
  depth_bnds_in = new((/ nlev_in, 2 /), double)
  depth_bnds_in(:,1) = cumsum(dz_in, 0)
  depth_bnds_in(0,0) = 0.0
  depth_bnds_in(1:nlev_in-1,0) = depth_bnds_in(0:nlev_in-2,1)
  depth_bnds_in@units = "m"

  fname_topo_in = "../grid_files/POP_w_overflows_gx1v7_SCRIP_20180920.nc"
  f_topo_in = addfile(fname_topo_in, "r")
  KMT_flat = ndtooned(f_topo_in->KMT)

  depth_out = todouble((/ \
    0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, \
    700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 2500, \
    3000, 3500, 4000, 4500, 5000, 5500 /))
  depth_out@units = "m"
  nlev_out = dimsizes(depth_out)
  ; construct depth_bnds_out as midpoints of depth, compute dz_out from depth_bnds_out
  depth_bnds_out = new((/ nlev_out, 2 /), double)
  depth_bnds_out(0,0) = 0.0
  depth_bnds_out(1:nlev_out-1,0) = 0.5*(depth_out(0:nlev_out-2)+depth_out(1:nlev_out-1))
  depth_bnds_out(0:nlev_out-2,1) = depth_bnds_out(1:nlev_out-1,0)
  depth_bnds_out(nlev_out-1,1) = depth_out(nlev_out-1)
  depth_bnds_out@units = "m"
  dz_out = depth_bnds_out(:,1) - depth_bnds_out(:,0)

  fname_in  = "../weights_files/POP_gx1v7_to_latlon_1x1_0E_conserve_20180914.nc"
  f_in  = addfile(fname_in, "r")

  src_grid_dims_in = f_in->src_grid_dims
  dst_grid_dims_in = f_in->dst_grid_dims

  n_a_in = product(src_grid_dims_in)
  n_b_in = product(dst_grid_dims_in)

  xc_a_in = f_in->xc_a
  yc_a_in = f_in->yc_a
  area_a_in = f_in->area_a
  mask_a_in = f_in->mask_a

  xc_b_in = f_in->xc_b
  yc_b_in = f_in->yc_b
  area_b_in = f_in->area_b
  frac_b_in = f_in->frac_b

  ; SCRIP matrices use 1-based indexing, so subtract 1 from col and row indices
  row_in = f_in->row-1
  col_in = f_in->col-1
  S_in   = f_in->S

  ; construct fulldepth src grid

  src_grid_dims = (/ src_grid_dims_in(0), src_grid_dims_in(1), nlev_in /)
  src_grid_dims!0 = "src_grid_rank"

  n_a = product(src_grid_dims)

  xc_a = new(n_a_in, double)
  xc_a!0 = "n_a_lateral"
  xc_a@units = xc_a_in@units
  xc_a = (/ xc_a_in /)

  yc_a = new(n_a_in, double)
  yc_a!0 = "n_a_lateral"
  yc_a@units = yc_a_in@units
  yc_a = (/ yc_a_in /)

  area_a = new(n_a_in, double)
  area_a!0 = "n_a_lateral"
  area_a@units = area_a_in@units
  area_a = (/ area_a_in /)

  zc_a = new(nlev_in, double)
  zc_a!0 = "n_a_depth"
  zc_a@units = depth_in@units
  zc_a@bounds = "zc_bnds_a"
  zc_a = (/ depth_in /)

  zc_bnds_a = new((/ nlev_in, 2 /), double)
  zc_bnds_a!0 = "n_a_depth"
  zc_bnds_a!1 = "d2"
  zc_bnds_a@units = depth_in@units
  zc_bnds_a = (/ depth_bnds_in /)

  frac_a = new(n_a, double)
  frac_a!0 = "n_a"
  frac_a@units = "unitless"
  do k = 0, nlev_in-1
    frac_a(k*n_a_in:(k+1)*n_a_in-1) = where(k .lt. KMT_flat, 1.0, 0.0)
  end do

  mask_a = new(n_a, integer)
  mask_a!0 = "n_a"
  mask_a@units = "unitless"
  do k = 0, nlev_in-1
    mask_a(k*n_a_in:(k+1)*n_a_in-1) = where(k .lt. KMT_flat, 1, 0)
  end do

  ; construct fulldepth dst grid

  dst_grid_dims = (/ dst_grid_dims_in(0), dst_grid_dims_in(1), nlev_out /)
  dst_grid_dims!0 = "dst_grid_rank"

  n_b = product(dst_grid_dims)

  xc_b = new(n_b_in, double)
  xc_b!0 = "n_b_lateral"
  xc_b@units = xc_b_in@units
  xc_b = (/ xc_b_in /)

  yc_b = new(n_b_in, double)
  yc_b!0 = "n_b_lateral"
  yc_b@units = yc_b_in@units
  yc_b = (/ yc_b_in /)

  area_b = new(n_b_in, double)
  area_b!0 = "n_b_lateral"
  area_b@units = area_b_in@units
  area_b = (/ area_b_in /)

  zc_b = new(nlev_out, double)
  zc_b!0 = "n_b_depth"
  zc_b@units = depth_out@units
  zc_b@bounds = "zc_bnds_b"
  zc_b = (/ depth_out /)

  zc_bnds_b = new((/ nlev_out, 2 /), double)
  zc_bnds_b!0 = "n_b_depth"
  zc_bnds_b!1 = "d2"
  zc_bnds_b@units = depth_out@units
  zc_bnds_b = (/ depth_bnds_out /)

  frac_b = new(n_b, double)
  frac_b!0 = "n_b"
  frac_b@units = "unitless"
  frac_b = 0.0

  mask_b = new(n_b, integer)
  mask_b!0 = "n_b"
  mask_b@units = "unitless"
  mask_b = 0

  ; construct fulldepth matrix

  ; compute fraction of depth_out layers covered by depth_in layers
  depth_frac = new((/ nlev_out, nlev_in /), double)
  do k_in = 0, nlev_in-1
    do k_out = 0, nlev_out-1
      ; in   |---|
      ; out        |---|
      if (depth_bnds_in(k_in,1) .le. depth_bnds_out(k_out,0)) then
        depth_frac(k_out, k_in) = 0.0
      else
        ; in       |---|
        ; out        |---|
        if (depth_bnds_in(k_in,1) .le. depth_bnds_out(k_out,1)) then
          depth_frac(k_out, k_in) = (depth_bnds_in(k_in,1) - (depth_bnds_in(k_in,0) > depth_bnds_out(k_out,0))) / dz_out(k_out)
        else
          ; in           |---|
          ; out    |---|
          if (depth_bnds_in(k_in,0) .gt. depth_bnds_out(k_out,1)) then
            depth_frac(k_out, k_in) = 0.0
          else
            ; in       |---|
            ; out    |---|
            depth_frac(k_out, k_in) = (depth_bnds_out(k_out,1) - (depth_bnds_in(k_in,0) > depth_bnds_out(k_out,0))) / dz_out(k_out)
          end if
        end if
      end if
    end do
  end do

  ; for each value of KMT-1, compute number of layer intersections in that column
  ; this is used to compute an upper bound on the number of fulldepth matrix entries
  column_entries_per_layer = dim_num_n(depth_frac .gt. 0.0, (/ 0 /))
  column_entries_cumsum = cumsum(column_entries_per_layer, 0)

  ; for each value of KMT-1, compute number of layer intersections in that column
  nlev_out_per_KMT = new((/ nlev_in /), integer)
  do k_in = 0, nlev_in-1
    nlev_out_per_KMT(k_in) = num(dim_sum(depth_frac(:,0:k_in)) .gt. 0.0)
  end do

  ; for output layer, compute indices of input layer that intersect it
  k_in_lo = new((/ nlev_out /), integer)
  k_in_hi = new((/ nlev_out /), integer)
  do k_out = 0, nlev_out-1
    k_out_ind := ind(depth_frac(k_out,:) .gt. 0.0)
    k_in_lo(k_out) = min(k_out_ind)
    k_in_hi(k_out) = max(k_out_ind)
  end do

  ; compute upper bound on n_s
  ; every entrie in S_in corresponds to a lateral intersection of grid
  ; this intersection potentially leads to column_entries_cumsum(KMT(col_in)-1) 3d intersections
  ; it is an upper bound, instead of exact, because 3d intersections will be clipped
  ; if coverage of dst_cell is less than frac_thres
  n_s_upper_bnd = sum(column_entries_cumsum(KMT_flat(col_in)-1))
  print(n_s_upper_bnd)

  n_s_upper_bnd = n_s_upper_bnd

  row = new(n_s_upper_bnd, integer)
  row!0 = "n_s"
  col = new(n_s_upper_bnd, integer)
  col!0 = "n_s"
  S   = new(n_s_upper_bnd, double)
  S!0 = "n_s"

  ; construct 3d matrix, one dst column at a time
  ; only add entries corresponding to dst layer if coverage exceeds frac_thres

  prnt_flag_ind_thres = toint(floor(0.05*n_b_in)) ; track progress in 5% increments
  prnt_flag_ind = 0
  print("constructing 3d matrix")

  n_s = 0
  do dst_in_ind = 0, n_b_in-1
    ; skip this dst cell column if it does not have sufficient lateral coverage
    if (frac_b_in(dst_in_ind) .gt. frac_thres) then
      mat_in_subset := ind(row_in .eq. dst_in_ind)
      col_in_subset := col_in(mat_in_subset)
      subset_len = dimsizes(mat_in_subset)
      ; fraction of dst_cell(dst_in_ind) covered by src_cell(col_in_subset)
      area_intersect_frac_subset := frac_b_in(dst_in_ind) * S_in(mat_in_subset)
      max_k_in = max(KMT_flat(col_in_subset)-1)
      do k_out = 0, nlev_out_per_KMT(max_k_in)-1
        dst_out_ind = k_out*n_b_in + dst_in_ind
        n_s_tmp = n_s
        vol_intersect_frac_out = 0.0d0
        do k_in = k_in_lo(k_out), k_in_hi(k_out)
          do i = 0, subset_len-1
            if (k_in .lt. KMT_flat(col_in_subset(i))) then
              col(n_s) = k_in*n_a_in + col_in_subset(i) + 1
              row(n_s) = dst_out_ind + 1
              S(n_s)   = depth_frac(k_out,k_in) * area_intersect_frac_subset(i)
              vol_intersect_frac_out = vol_intersect_frac_out + S(n_s)
              n_s = n_s + 1
            end if
          end do
        end do
        ; if there is sufficient volume coverage then
        ;   set frac_b and mask_b
        ;   normalize volume coverage into remapping weights
        ; otherwise undo additions of matrix entries,
        ;   set frac_b and mask_b to 0, and skip remaining output layers
        if (vol_intersect_frac_out .gt. frac_thres) then
          frac_b(dst_out_ind) = vol_intersect_frac_out
          mask_b(dst_out_ind) = 1
          S(n_s_tmp:n_s-1) = S(n_s_tmp:n_s-1) / vol_intersect_frac_out
        else
          frac_b(dst_out_ind) = 0.0
          mask_b(dst_out_ind) = 0
          n_s = n_s_tmp
          break
        end if
      end do
    end if

    prnt_flag_ind = prnt_flag_ind + 1
    if (prnt_flag_ind .eq. prnt_flag_ind_thres) then
      print(100.0*(dst_in_ind+1)/n_b_in + " % complete, n_s="+n_s)
      prnt_flag_ind = 0
    end if
  end do

  ; write matrix

  setfileoption("nc", "Format", "NetCDF4")
  setfileoption("nc", "CompressionLevel", 1)

  print("writing "+fname_out)
  system("rm -f "+fname_out)
  f_out = addfile(fname_out, "c")

; filedimdef(f_out, (/ "n_a", "n_a_lateral", "n_a_depth", "n_b", "n_b_lateral", "n_b_depth", "n_s" /), \
;                   (/   n_a,        n_a_in,     nlev_in,   n_b,        n_b_in,    nlev_out,   n_s /), \
;                   (/ False,         False,       False, False,         False,       False, False /))

  filedimdef(f_out, (/ "n_b", "n_b_lateral", "n_b_depth", "n_s" /), \
                    (/   n_b,        n_b_in,    nlev_out,   n_s /), \
                    (/ False,         False,       False, False /))

  f_out->src_grid_dims = src_grid_dims
  f_out->dst_grid_dims = dst_grid_dims

; f_out->xc_a      = xc_a
; f_out->yc_a      = yc_a
; f_out->area_a    = area_a
; f_out->zc_a      = zc_a
; f_out->zc_bnds_a = zc_bnds_a
; f_out->frac_a    = frac_a
; f_out->mask_a    = mask_a

  f_out->xc_b      = xc_b
  f_out->yc_b      = yc_b
  f_out->area_b    = area_b
  f_out->zc_b      = zc_b
  f_out->zc_bnds_b = zc_bnds_b
; f_out->frac_b    = frac_b
  f_out->mask_b    = mask_b

  f_out->row = row(0:n_s-1)
  f_out->col = col(0:n_s-1)
  f_out->S = S(0:n_s-1)

end
