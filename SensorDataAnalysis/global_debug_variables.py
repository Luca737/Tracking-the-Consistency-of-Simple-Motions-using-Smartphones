# All global variable in this file and only in this file should start with GDB (global debug variable)
# All of these are to be set to Null and should be set in some testing function.
# Those variable that are Null will be ignored and the standard value or the value given to the function will be used instead.

DEBUG_ENABLED = False

gdv_smooth_points_smoothing_radius = None
gdv_median_filter_kernel_size = None
gdv_pass_filter_high_cutoff = None
gdv_pass_filter_low_cutoff = None
gdv_pass_filter_order = None
gdv_ewma_weight = None

gdv_analysis_minima_baseline_factor = None
gdv_analysis_maxima_baseline_factor = None
gdv_zero_plato_val_radius = None
gdv_zero_plato_slope_radius = None
gdv_zero_plato_min_length = None
gdv_zero_plato_allowed_skip_length = None

# Useless variables for classification analysis. Better use the default global variable by hand instead
gdv_get_motion_data_interval_offset = None
