"""Author: Nicola Vidovic

This files includes all developed analysis methods of all motion data types.

"""

from typing import List, Tuple

import global_debug_variables as gdv
from FileIO import *
from filters import *
from functions import *

folder_name = ""

OUT_FILE_NAME = "auto_annotated.json"
ZIP_FILE_PATH = f"/Users/nicola/code/Bachelor/TrainerHub/testFolder/{folder_name}/{folder_name}.zip"

# TODO: Include zero_plato directly after interval start in search. ...
# ... ~ if there is no significance or to low maxima or similar then remove current
# start and use that plato instead. Deals with overlapping Intervals with only
# different starts (essentially the same) and is probably a better guess of the actual interval.

# TODO: Allow multiple endings to each interval. Only stop if there is a condition, which ...
# ... invalidates the interval. Later have a routine, which selects from the detected intervals.


def analyzeV1(data: list, time_stamps: list, *_) -> List[Tuple[int, int]]:
    """
    Uses nearest zero point or end of nearest zero plato for the start point.
    It then requires exactly two zero points next and finally uses the nearest
    zero point or start of the nearest zero plato for the end of the interval.
    Then it checks if the calculated interval is within a specified length
    range, where those that passed are counted and returned as intervals/repetitions.
    """
    # _, zero_points, zero_platos = calc_key_features(data, time_stamps)
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.01, 0.1, 0, 0)
    detected_intervals = []
    plato_starts = [plato[0] for plato in zero_platos]
    plato_stops = [plato[1] for plato in zero_platos]
    possible_start_points = zero_points + plato_stops
    possible_start_points.sort()
    last_start = None
    for start_point in possible_start_points:
        if last_start == start_point:
            continue
        # Get next 0 element.
        mid_point = next((i for i in zero_points if i > start_point), None)
        if mid_point is None:
            # No full motion cycle possible -> Done.
            break
        # Get next 0 element.
        mid_point = next((i for i in zero_points if i > mid_point), None)
        if mid_point is None:
            # No full motion cycle possible -> Done.
            break
        # Possible End point can be start of zero plato or 0 point.
        end_point = None
        end_point0 = next((i for i in zero_points if i > mid_point), None)
        end_point1 = next((i for i in plato_starts if i > mid_point), None)
        if end_point0 is None and end_point1 is None:
            break
        if end_point0 is None:
            end_point = end_point1
        elif end_point1 is None:
            end_point = end_point0
        else:
            end_point = min(end_point0, end_point1)
        # Calculate time for the possible interval.
        delta_time = time_stamps[end_point] - time_stamps[start_point]
        if 0.5 <= delta_time:
            detected_intervals.append((start_point, end_point))

        # detected_intervals.append((start_point, end_point))

    return detected_intervals


def analyzeV1_1(data: list, time_stamps: list, *_) -> List[Tuple[int, int]]:
    """
    Goal is to achieve high recall and to reduce the enormous amount of false
    positives from the last method to later filter the guessed intervals and
    achieve high precision.

    No non significant sections between significant sections are allowed. In other words
    sections where there is no acceleration between the key section, which indicate
    the acceleration at the start and the stopping motion or deceleration at the end.
    (For the first half and second half of the motion interval)
    Now also the pause section (phase1) is included by allowing two significant
    section in the middle of the motion (stop motion for the first half and start
    motion of the second half.)
    """

    SIGNIFICANCE_VALUE = 0.01
    # _, zero_points, zero_platos = calc_key_features(data, time_stamps)
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.01, 0.1, 0, 0)
    detected_intervals = []
    plato_starts = [plato[0] for plato in zero_platos]
    plato_stops = [plato[1] for plato in zero_platos]
    possible_start_points = zero_points + plato_stops
    possible_start_points.sort()
    last_start = None
    for start_point in possible_start_points:
        if last_start == start_point:
            continue
        last_start = start_point

        zp_index, mid_point = next(
            ((i, val) for i, val in enumerate(zero_points) if val > start_point), (None, None))
        if zp_index is None:
            break
        # Calculate if start_point to mid_point has significant acceleration.
        if not abs(avg(data, start_point, mid_point)) > SIGNIFICANCE_VALUE:
            # Note: Start point might still be viable.
            continue

        # Find significant acceleration/deceleration (stopping motion).
        # TODO: Check for length of current interval and stop if too long.
        is_valid_motion = None
        while True:
            # Significant acceleration means there was likely a stop motion.
            zp_index += 1
            if zp_index >= len(zero_points):
                is_valid_motion = False
                break
            cord0, cord1 = zero_points[zp_index-1], zero_points[zp_index]
            if abs(avg(data, cord0, cord1)) > SIGNIFICANCE_VALUE:
                is_valid_motion = True
                break
            # Since there was no significant acceleration the next section must
            # not have significant acceleration.
            # Note: Can theoretically be part of a motion but shouldn't be.
            zp_index += 1
            if zp_index >= len(zero_points):
                is_valid_motion = False
                break
            cord0, cord1 = zero_points[zp_index-1], zero_points[zp_index]
            if abs(avg(data, cord0, cord1)) > SIGNIFICANCE_VALUE:
                is_valid_motion = False
                break

        assert is_valid_motion is not None
        if not is_valid_motion:
            continue

        mid_point = zero_points[zp_index]

        # Calculate endpoint.
        is_zero_point = None
        allow_further_significant_curved = True
        section_start = mid_point
        last_potential_section_end = mid_point
        is_valid_motion = None
        while True:
            # Possible End point can be start of zero plato or 0 point.
            end_point = None
            end_point0 = next((i for i in zero_points if i > last_potential_section_end), None)
            end_point1 = next((i for i in plato_starts if i > last_potential_section_end), None)
            if end_point0 is None and end_point1 is None:
                is_valid_motion = False
                break
            if end_point0 is None:
                end_point = end_point1
                is_zero_point = False
            elif end_point1 is None:
                end_point = end_point0
                is_zero_point = True
            elif end_point0 <= end_point1:
                end_point = end_point0
                is_zero_point = True
            else:
                end_point = end_point1
                is_zero_point = False
            if abs(avg(data, section_start, end_point)) > SIGNIFICANCE_VALUE:
                is_valid_motion = True
                break
            last_potential_section_end = end_point
            if not is_zero_point:
                # Non significant platos are ignored and since the sign of the
                # current curve has not changed we try again.
                # TODO: Potential Problem: If there is a long zero plato the average is stronger held near zero and a higher/longer bump/interval is needed for significance.
                # This should not be an issue in most cases though.
                # There is an easy fix, by counting the number of zero point in the zero plato and
                # differentiating between odd and even number but at this point is not worth
                # implementing anymore.
                continue
            # The sign of the curve has changed and it has to change again (search for 0-point).
            # In this portion only one significant curve is allowed indicating a pause
            # in the executed motion between the first and second part of that motion.
            next_section_start = next(
                (i for i in zero_points if i > last_potential_section_end), None)
            if next_section_start is None:
                is_valid_motion = False
                break
            acc = abs(avg(data, section_start, next_section_start))
            section_start = next_section_start
            last_potential_section_end = next_section_start
            if not acc > SIGNIFICANCE_VALUE:
                continue
            # Significance is only allowed once in this section.
            if allow_further_significant_curved:
                allow_further_significant_curved = False
                continue
            is_valid_motion = False
            break

        assert is_valid_motion is not None
        if not is_valid_motion:
            continue

        # Calculate if the interval has reasonable length.
        delta_time = time_stamps[end_point] - time_stamps[start_point]
        if 1 <= delta_time:
            detected_intervals.append((start_point, end_point))

    return detected_intervals


def analyzeV1_1b(data: list, time_stamps: list, *_) -> List[Tuple[int, int]]:
    """
    Copy of V1_1 but with simple BASELINE for significance testing during the writing
    of the Thesis. This could work if implemented better for both positive and negative
    values (maxima and minima) but as v2 does this with a better approach this is not
    further explored.

    Goal is to achieve high recall and to reduce the enormous amount of false
    positives from the last method to later filter the guessed intervals and
    achieve high precision.

    No non significant sections between significant sections are allowed. In other words
    sections where there is no acceleration between the key section, which indicate
    the acceleration at the start and the stopping motion or deceleration at the end.
    (For the first half and second half of the motion interval)
    Now also the pause section (phase1) is included by allowing two significant
    section in the middle of the motion (stop motion for the first half and start
    motion of the second half.)
    """

    # SIGNIFICANCE_VALUE = 0.02
    BASELINE = max(np.abs(data)) / 2.1
    # _, zero_points, zero_platos = calc_key_features(data, time_stamps)
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.01, 0.1, 0, 0)
    detected_intervals = []
    plato_starts = [plato[0] for plato in zero_platos]
    plato_stops = [plato[1] for plato in zero_platos]
    possible_start_points = zero_points + plato_stops
    possible_start_points.sort()
    last_start = None
    for start_point in possible_start_points:
        if last_start == start_point:
            continue
        last_start = start_point

        zp_index, mid_point = next(
            ((i, val) for i, val in enumerate(zero_points) if val > start_point), (None, None))
        if zp_index is None:
            break
        # Calculate if start_point to mid_point has significant acceleration.
        if not max(np.abs(data[start_point:mid_point])) > BASELINE:
            # Note: Start point might still be viable.
            continue

        # Find significant acceleration/deceleration (stopping motion).
        # TODO: Check for length of current interval and stop if too long.
        is_valid_motion = None
        while True:
            # Significant acceleration means there was likely a stop motion.
            zp_index += 1
            if zp_index >= len(zero_points):
                is_valid_motion = False
                break
            cord0, cord1 = zero_points[zp_index-1], zero_points[zp_index]
            if max(np.abs(data[cord0:cord1])) > BASELINE:
                is_valid_motion = True
                break
            # Since there was no significant acceleration the next section must
            # not have significant acceleration.
            # Note: Can theoretically be part of a motion but shouldn't be.
            zp_index += 1
            if zp_index >= len(zero_points):
                is_valid_motion = False
                break
            cord0, cord1 = zero_points[zp_index-1], zero_points[zp_index]
            if max(np.abs(data[cord0:cord1])) > BASELINE:
                is_valid_motion = False
                break

        assert is_valid_motion is not None
        if not is_valid_motion:
            continue

        mid_point = zero_points[zp_index]

        # Calculate endpoint.
        is_zero_point = None
        allow_further_significant_curved = True
        section_start = mid_point
        last_potential_section_end = mid_point
        is_valid_motion = None
        while True:
            # Possible End point can be start of zero plato or 0 point.
            end_point = None
            end_point0 = next((i for i in zero_points if i > last_potential_section_end), None)
            end_point1 = next((i for i in plato_starts if i > last_potential_section_end), None)
            if end_point0 is None and end_point1 is None:
                is_valid_motion = False
                break
            if end_point0 is None:
                end_point = end_point1
                is_zero_point = False
            elif end_point1 is None:
                end_point = end_point0
                is_zero_point = True
            elif end_point0 <= end_point1:
                end_point = end_point0
                is_zero_point = True
            else:
                end_point = end_point1
                is_zero_point = False
            if max(np.abs(data[section_start:end_point])) > BASELINE:
                is_valid_motion = True
                break
            last_potential_section_end = end_point
            if not is_zero_point:
                # Non significant platos are ignored and since the sign of the
                # current curve has not changed we try again.
                # TODO: Potential Problem: If there is a long zero plato the average is stronger held near zero and a higher/longer bump/interval is needed for significance.
                # This should not be an issue in most cases though.
                # There is an easy fix, by counting the number of zero point in the zero plato and
                # differentiating between odd and even number but at this point is not worth
                # implementing anymore.
                continue
            # The sign of the curve has changed and it has to change again (search for 0-point).
            # In this portion only one significant curve is allowed indicating a pause
            # in the executed motion between the first and second part of that motion.
            next_section_start = next(
                (i for i in zero_points if i > last_potential_section_end), None)
            if next_section_start is None:
                is_valid_motion = False
                break
            max_val = max(np.abs(data[section_start:next_section_start]))
            section_start = next_section_start
            last_potential_section_end = next_section_start
            if not max_val > BASELINE:
                continue
            # Significance is only allowed once in this section.
            if allow_further_significant_curved:
                allow_further_significant_curved = False
                continue
            is_valid_motion = False
            break

        assert is_valid_motion is not None
        if not is_valid_motion:
            continue

        # Calculate if the interval has reasonable length.
        # delta_time = time_stamps[end_point] - time_stamps[start_point]
        # if 0.5 <= delta_time <= 5.0:
        detected_intervals.append((start_point, end_point))

    return detected_intervals


def analyzeV1_2(data: list, time_stamps: list, *_) -> List[Tuple[int, int]]:
    """
    Small change: When the start of a zero plato is used as an end point of the
    repetition then it is saved as a repetition but additionally further end point
    to the interval are searched for.
    """

    SIGNIFICANCE_VALUE = 0.01
    # _, zero_points, zero_platos = calc_key_features(data, time_stamps)
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.01, 0.1, 0, 0)
    detected_intervals = []
    plato_starts = [plato[0] for plato in zero_platos]
    plato_stops = [plato[1] for plato in zero_platos]
    possible_start_points = zero_points + plato_stops
    possible_start_points.sort()
    last_start = None
    for start_point in possible_start_points:
        if last_start == start_point:
            continue
        last_start = start_point

        zp_index, mid_point = next(
            ((i, val) for i, val in enumerate(zero_points) if val > start_point), (None, None))
        if zp_index is None:
            break
        # Calculate if start_point to mid_point has significant acceleration.
        if not abs(avg(data, start_point, mid_point)) > SIGNIFICANCE_VALUE:
            # Note: Start point might still be viable.
            continue

        # Check for significant acceleration (Stopping motion) for next section.
        # TODO: Check for length of current interval and stop if too long.
        is_valid_motion = None
        while True:
            # Significant acceleration means there was likely a stop motion.
            zp_index += 1
            if zp_index >= len(zero_points):
                is_valid_motion = False
                break
            cord0, cord1 = zero_points[zp_index-1], zero_points[zp_index]
            if abs(avg(data, cord0, cord1)) > SIGNIFICANCE_VALUE:
                is_valid_motion = True
                break
            # Next section must not have significant acceleration.
            # Note: Can theoretically be part of a motion.
            zp_index += 1
            if zp_index >= len(zero_points):
                is_valid_motion = False
                break
            cord0, cord1 = zero_points[zp_index-1], zero_points[zp_index]
            if abs(avg(data, cord0, cord1)) > SIGNIFICANCE_VALUE:
                is_valid_motion = False
                break

        assert is_valid_motion is not None
        if not is_valid_motion:
            continue

        mid_point = zero_points[zp_index]

        # Calculate endpoint.
        is_zero_point = None
        allow_further_significant_curved = True
        section_start = mid_point
        last_potential_section_end = mid_point
        is_valid_motion = None
        while True:
            # Possible End point can be start of zero plato or 0 point.
            end_point = None
            end_point0 = next((i for i in zero_points if i > last_potential_section_end), None)
            end_point1 = next((i for i in plato_starts if i > last_potential_section_end), None)
            if end_point0 is None and end_point1 is None:
                is_valid_motion = False
                break
            if end_point0 is None:
                end_point = end_point1
                is_zero_point = False
            elif end_point1 is None:
                end_point = end_point0
                is_zero_point = True
            elif end_point0 <= end_point1:
                end_point = end_point0
                is_zero_point = True
            else:
                end_point = end_point1
                is_zero_point = False
            last_potential_section_end = end_point
            if abs(avg(data, section_start, end_point)) > SIGNIFICANCE_VALUE:
                detected_intervals.append((start_point, end_point))
                # Try further if it is a zero plato; Interval might be longer.
                if is_zero_point:
                    is_valid_motion = True
                    break
                else:
                    continue
            if not is_zero_point:
                # Non significant platos are ignored and since the sign of the
                # current curve has not changed we try again.
                # TODO: Potential Problem: If there is a long zero plato the average is stronger held near zero and a higher/longer bump/interval is needed for significance.
                # There is an easy fix, by counting the number of zero point in the zero plato and
                # differentiating between odd and even number but at this point is not worth
                # implementing anymore.
                continue
            # The sign of the curve has changed and it has to change again (search for 0-point).
            # In this portion only one significant curve is allowed indicating a pause
            # in the executed motion between the first and second part of that motion.
            next_section_start = next(
                (i for i in zero_points if i > last_potential_section_end), None)
            if next_section_start is None:
                is_valid_motion = False
                break
            acc = abs(avg(data, section_start, next_section_start))
            section_start = next_section_start
            last_potential_section_end = next_section_start
            if not acc > SIGNIFICANCE_VALUE:
                continue
            # Significance is only allowed once in this section.
            if allow_further_significant_curved:
                allow_further_significant_curved = False
                continue
            is_valid_motion = False
            break

        assert is_valid_motion is not None
        if not is_valid_motion:
            continue

        # Calculate if the interval has reasonable length.
        # delta_time = time_stamps[end_point] - time_stamps[start_point]
        # if 0.5 <= delta_time <= 5.0:
        # detected_intervals.append((start_point, end_point))

    return detected_intervals


def analyze_v2(data: list, time_stamps: list, *_) -> List[Tuple[int, int]]:
    """Approach completely changed to V1. Now similar to analysis of rotation rate."""

    # Best (and same) results (21 files, 17.02.23) from ~2.95 - ~2.9725 for data -> pca + lowPass(2.5, 100)
    min_baseline_factor = 2.965
    max_baseline_factor = 2.965

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline > 0 or maxima_baseline < 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)
    # _, zero_points, zero_platos = calc_key_features(data, time_stamps)
    # zero_platos = calc_zero_platos(data, derivative(data, time_stamps))
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.01, 0.1, 0, 0)

    # Requirement for at least one motion interval to exist / be detected by this method.
    if len(minima) + len(maxima) < 3 or len(minima) == 0 or len(maxima) == 0:
        return []

    # TODO: This is potentially more dangerous for linear acceleration. There are more odd bumps, similar to 'real' bumps.
    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        start_stop_extrema = minima
        middle_extrema = maxima
    else:
        start_stop_extrema = maxima
        middle_extrema = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    middle_extrema_i = 0
    for start_stop_extrema_i in range(len(start_stop_extrema)-1):
        start_extremum = start_stop_extrema_i
        stop_extremum = start_stop_extrema_i+1
        middle_extremum_0 = next((i for i, el in enumerate(middle_extrema[middle_extrema_i:], middle_extrema_i) if el > start_stop_extrema[start_extremum]), None)
        if middle_extremum_0 is None:
            break
        if middle_extrema[middle_extremum_0] > start_stop_extrema[stop_extremum]:
            continue
        middle_extremum_1 = None
        if middle_extremum_0+1 < len(middle_extrema) and middle_extrema[middle_extremum_0+1] < start_stop_extrema[stop_extremum]:
            middle_extremum_1 = middle_extremum_0+1
            if middle_extremum_0+2 < len(middle_extrema) and middle_extrema[middle_extremum_0+2] < start_stop_extrema[stop_extremum]:
                middle_extrema_i += 3
                continue

        # Valid interval found.
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[start_extremum])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +2 since start != stop of same interval and the next zero point is always between the two extrema.
            zero_point_i = closest_left_zero_point_i + 2
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > start_stop_extrema[start_extremum]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[stop_extremum]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > start_stop_extrema[stop_extremum]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        detected_intervals.append((interval_start, interval_stop))

        middle_extrema_i += 1 + int((middle_extremum_1 is not None))

    return split_overlapping_intervals(detected_intervals)
    # return detected_intervals


def analyze_v3(data: list, time_stamps: list, *_) -> List[Tuple[int, int, int, int]]:
    """Now supports phases."""

    # Best (and same) results (21 files, 17.02.23) from ~2.95 - ~2.9725 for data -> pca + lowPass(2.5, 100)
    min_baseline_factor = 2.965
    max_baseline_factor = 2.965

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline > 0 or maxima_baseline < 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)
    # _, zero_points, zero_platos = calc_key_features(data, time_stamps)
    # zero_platos = calc_zero_platos(data, derivative(data, time_stamps))
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.01, 0.1, 0, 0)

    # Requirement for at least one motion interval to exist / be detected by this method.
    if len(minima) + len(maxima) < 3 or len(minima) == 0 or len(maxima) == 0:
        return []

    # TODO: This is potentially more dangerous for linear acceleration. There are more odd bumps, similar to 'real' bumps.
    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        start_stop_extrema = minima
        middle_extrema = maxima
    else:
        start_stop_extrema = maxima
        middle_extrema = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    middle_extrema_i = 0
    for start_stop_extrema_i in range(len(start_stop_extrema)-1):
        start_extremum = start_stop_extrema_i
        stop_extremum = start_stop_extrema_i+1
        middle_extremum_0 = next((i for i, el in enumerate(middle_extrema[middle_extrema_i:], middle_extrema_i) if el > start_stop_extrema[start_extremum]), None)
        if middle_extremum_0 is None:
            break
        if middle_extrema[middle_extremum_0] > start_stop_extrema[stop_extremum]:
            continue
        middle_extremum_1 = None
        if middle_extremum_0+1 < len(middle_extrema) and middle_extrema[middle_extremum_0+1] < start_stop_extrema[stop_extremum]:
            middle_extremum_1 = middle_extremum_0+1
            if middle_extremum_0+2 < len(middle_extrema) and middle_extrema[middle_extremum_0+2] < start_stop_extrema[stop_extremum]:
                middle_extrema_i += 3
                continue

        # Valid interval found.
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[start_extremum])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +2 since start != stop of same interval and the next zero point is always between the two extrema.
            zero_point_i = closest_left_zero_point_i + 2
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > start_stop_extrema[start_extremum]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find ends of phase0 and phase1 -----------------------------------
        # TODO: Naive approach - ~outer bound. phase0_end must be larger, phase1_end must be smaller.
        phase0_end = middle_extrema[middle_extremum_0]
        if middle_extremum_1 is None:
            phase1_end = phase0_end
        else:
            phase1_end = middle_extrema[middle_extremum_1]

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[stop_extremum]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > start_stop_extrema[stop_extremum]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        detected_intervals.append((interval_start, phase0_end, phase1_end, interval_stop))

        middle_extrema_i += 1 + int((middle_extremum_1 is not None))

    return split_overlapping_intervals(detected_intervals)
    # return detected_intervals


def analyze_v3_1(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """This methods improves the phase 1 start and end detection, by using a baseline
    approach similar to how the extrema are calculated.
    """

    # Best (and same) results (21 files, 17.02.23) from ~2.95 - ~2.9725 for data -> pca + lowPass(2.5, 100)
    min_baseline_factor = 2.965
    max_baseline_factor = 2.965

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline > 0 or maxima_baseline < 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)
    # _, zero_points, zero_platos = calc_key_features(data, time_stamps)
    # zero_platos = calc_zero_platos(data, derivative(data, time_stamps))
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.01, 0.1, 0, 0)

    if len(minima) != len(maxima):
        print_conditional(verbose, f"{YELLOW[0]}#minima ({len(minima)}) != #maxima ({len(maxima)}){YELLOW[1]}")

    # Requirement for at least one motion interval to exist / be detected by this method.
    if len(minima) + len(maxima) < 3 or len(minima) == 0 or len(maxima) == 0:
        return []

    # TODO: This is potentially more dangerous for linear acceleration. There are more odd bumps, similar to 'real' bumps.
    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        start_stop_extrema = minima
        middle_extrema = maxima
    else:
        start_stop_extrema = maxima
        middle_extrema = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    middle_extrema_i = 0
    for start_stop_extrema_i in range(len(start_stop_extrema)-1):
        start_extremum = start_stop_extrema_i
        stop_extremum = start_stop_extrema_i+1
        middle_extremum_0 = next((i for i, el in enumerate(middle_extrema[middle_extrema_i:], middle_extrema_i) if el > start_stop_extrema[start_extremum]), None)
        if middle_extremum_0 is None:
            break
        if middle_extrema[middle_extremum_0] > start_stop_extrema[stop_extremum]:
            continue
        middle_extremum_1 = None
        if middle_extremum_0+1 < len(middle_extrema) and middle_extrema[middle_extremum_0+1] < start_stop_extrema[stop_extremum]:
            middle_extremum_1 = middle_extremum_0+1
            if middle_extremum_0+2 < len(middle_extrema) and middle_extrema[middle_extremum_0+2] < start_stop_extrema[stop_extremum]:
                middle_extrema_i += 3
                continue

        # Valid interval found.
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[start_extremum])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +3 since there are at least two more zero points between the first and last extrema that are of no interest.
            zero_point_i = closest_left_zero_point_i + 3
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > start_stop_extrema[start_extremum]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find ends of phase0 and phase1 -----------------------------------
        # Use the first point below a baseline (calculated to be between the local max and min) from the left and from the right
        # as the start and stop respectively.
        if middle_extremum_1 is None:
            phase0_end = middle_extrema[middle_extremum_0]
            phase1_end = phase0_end
        else:
            if interval_starts_with_minima:
                relative_upper = min(data[middle_extrema[middle_extremum_0]], data[middle_extrema[middle_extremum_1]])
                relative_lower = max(0, min(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]]))
                # baseline = relative_upper - (relative_upper - relative_lower) / 2.3
            else:
                relative_upper = max(data[middle_extrema[middle_extremum_0]], data[middle_extrema[middle_extremum_1]])
                relative_lower = min(0, max(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]]))
                # baseline = relative_upper - (relative_upper - relative_lower) / 2.3
            baseline = (relative_upper + relative_lower) / 2
            factor = 1 - 2 * (1 - int(interval_starts_with_minima))  # Correct for searching below or above 0.
            phase0_end = next((i for i, el in enumerate(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]], middle_extrema[middle_extremum_0])
                               if el*factor < baseline*factor))
            phase1_end = next((i for el, i in zip(data[middle_extrema[middle_extremum_1]:middle_extrema[middle_extremum_0]:-1], range(middle_extrema[middle_extremum_1], 0, -1))
                               if el*factor < baseline*factor))

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[stop_extremum]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > start_stop_extrema[stop_extremum]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        if interval_start > phase0_end or phase0_end > phase1_end or phase1_end > interval_stop:
            print_red("Acceleration: Interval is wrong")
            print_red(interval_start, phase0_end, phase1_end, interval_stop)
            exit()

        detected_intervals.append((interval_start, phase0_end, phase1_end, interval_stop))

        middle_extrema_i += 1 + int((middle_extremum_1 is not None))

    return split_overlapping_intervals(detected_intervals)
    # return detected_intervals


def analyze_v3_1_1(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """This version fixes the bug with the zero platos extending over the extrema."""

    # Best (and same) results (21 files, 17.02.23) from ~2.95 - ~2.9725 for data -> pca + lowPass(2.5, 100)
    min_baseline_factor = 3.05
    max_baseline_factor = 3.05

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline > 0 or maxima_baseline < 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)
    # _, zero_points, zero_platos = calc_key_features(data, time_stamps)
    # zero_platos = calc_zero_platos(data, derivative(data, time_stamps))
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.01, 0.3, 0, 0)
    # zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.01, 0.1, 0, 0)

    if len(minima) != len(maxima):
        print_conditional(verbose, f"{YELLOW[0]}#minima ({len(minima)}) != #maxima ({len(maxima)}){YELLOW[1]}")

    # Requirement for at least one motion interval to exist / be detected by this method.
    if len(minima) + len(maxima) < 3 or len(minima) == 0 or len(maxima) == 0:
        return []

    # TODO: This is potentially more dangerous for linear acceleration. There are more odd bumps, similar to 'real' bumps.
    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        start_stop_extrema = minima
        middle_extrema = maxima
    else:
        start_stop_extrema = maxima
        middle_extrema = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    middle_extrema_i = 0
    for start_stop_extrema_i in range(len(start_stop_extrema)-1):
        start_extremum = start_stop_extrema_i
        stop_extremum = start_stop_extrema_i+1
        middle_extremum_0 = next((i for i, el in enumerate(middle_extrema[middle_extrema_i:], middle_extrema_i) if el > start_stop_extrema[start_extremum]), None)
        if middle_extremum_0 is None:
            break
        if middle_extrema[middle_extremum_0] > start_stop_extrema[stop_extremum]:
            continue
        middle_extremum_1 = None
        if middle_extremum_0+1 < len(middle_extrema) and middle_extrema[middle_extremum_0+1] < start_stop_extrema[stop_extremum]:
            middle_extremum_1 = middle_extremum_0+1
            if middle_extremum_0+2 < len(middle_extrema) and middle_extrema[middle_extremum_0+2] < start_stop_extrema[stop_extremum]:
                middle_extrema_i += 3
                continue

        # Valid interval found.
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[start_extremum])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +3 since there are at least two more zero points between the first and last extrema that are of no interest.
            zero_point_i = closest_left_zero_point_i + 3
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > start_stop_extrema[start_extremum]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                if zero_platos[closest_zero_plato_i][1] < start_stop_extrema[start_extremum]:
                    key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find ends of phase0 and phase1 -----------------------------------
        # Use the first point below a baseline (calculated to be between the local max and min) from the left and from the right
        # as the start and stop respectively.
        if middle_extremum_1 is None:
            phase0_end = middle_extrema[middle_extremum_0]
            phase1_end = phase0_end
        else:
            if interval_starts_with_minima:
                relative_upper = min(data[middle_extrema[middle_extremum_0]], data[middle_extrema[middle_extremum_1]])
                relative_lower = max(0, min(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]]))
                # baseline = relative_upper - (relative_upper - relative_lower) / 2.3
            else:
                relative_upper = max(data[middle_extrema[middle_extremum_0]], data[middle_extrema[middle_extremum_1]])
                relative_lower = min(0, max(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]]))
                # baseline = relative_upper - (relative_upper - relative_lower) / 2.3
            baseline = (relative_upper + relative_lower) / 2
            factor = 1 - 2 * (1 - int(interval_starts_with_minima))  # Correct for searching below or above 0.
            phase0_end = next((i for i, el in enumerate(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]], middle_extrema[middle_extremum_0])
                               if el*factor < baseline*factor))
            phase1_end = next((i for el, i in zip(data[middle_extrema[middle_extremum_1]:middle_extrema[middle_extremum_0]:-1], range(middle_extrema[middle_extremum_1], 0, -1))
                               if el*factor < baseline*factor))

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[stop_extremum]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > start_stop_extrema[stop_extremum]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None and zero_platos[closest_zero_plato_i][0] > start_stop_extrema[stop_extremum]:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        if interval_start > phase0_end or phase0_end > phase1_end or phase1_end > interval_stop:
            print_red("Acceleration: Interval is wrong")
            print_red(interval_start, phase0_end, phase1_end, interval_stop)
            exit()

        detected_intervals.append((interval_start, phase0_end, phase1_end, interval_stop))

        middle_extrema_i += 1 + int((middle_extremum_1 is not None))

    return split_overlapping_intervals(detected_intervals)
    # return detected_intervals


def analyze_v3_1_1_b(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """This methods gets rid of the zero plato bug completely and not just ignores faulty platos like 3.1.1.
    However it is somewhat worse then v3.1.2. Minor improvement to F-Score but overall worse length errors.
    This may need more fine tuning.
    + The temporary fixes of 3.1.1 are not removed as this was just a quick test. In this method they are useless.
    """

    # Best (and same) results (21 files, 17.02.23) from ~2.95 - ~2.9725 for data -> pca + lowPass(2.5, 100)
    min_baseline_factor = 3.05
    max_baseline_factor = 3.05

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline > 0 or maxima_baseline < 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)

    abs_min_extrema = min(*map(lambda x: abs(data[x]), minima), *map(lambda x: abs(data[x]), maxima))
    d = derivative(data, time_stamps)
    max_d = max(np.abs(d))
    max_val_zero_plato = abs_min_extrema / 3
    max_slope_zero_plato = max_d / 6.5

    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), max_val_zero_plato, max_slope_zero_plato, 0, 0)

    if len(minima) != len(maxima):
        print_conditional(verbose, f"{YELLOW[0]}#minima ({len(minima)}) != #maxima ({len(maxima)}){YELLOW[1]}")

    # Requirement for at least one motion interval to exist / be detected by this method.
    if len(minima) + len(maxima) < 3 or len(minima) == 0 or len(maxima) == 0:
        return []

    # TODO: This is potentially more dangerous for linear acceleration. There are more odd bumps, similar to 'real' bumps.
    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        start_stop_extrema = minima
        middle_extrema = maxima
    else:
        start_stop_extrema = maxima
        middle_extrema = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    middle_extrema_i = 0
    for start_stop_extrema_i in range(len(start_stop_extrema)-1):
        start_extremum = start_stop_extrema_i
        stop_extremum = start_stop_extrema_i+1
        middle_extremum_0 = next((i for i, el in enumerate(middle_extrema[middle_extrema_i:], middle_extrema_i) if el > start_stop_extrema[start_extremum]), None)
        if middle_extremum_0 is None:
            break
        if middle_extrema[middle_extremum_0] > start_stop_extrema[stop_extremum]:
            continue
        middle_extremum_1 = None
        if middle_extremum_0+1 < len(middle_extrema) and middle_extrema[middle_extremum_0+1] < start_stop_extrema[stop_extremum]:
            middle_extremum_1 = middle_extremum_0+1
            if middle_extremum_0+2 < len(middle_extrema) and middle_extrema[middle_extremum_0+2] < start_stop_extrema[stop_extremum]:
                middle_extrema_i += 3
                continue

        # Valid interval found.
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[start_extremum])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +3 since there are at least two more zero points between the first and last extrema that are of no interest.
            zero_point_i = closest_left_zero_point_i + 3
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > start_stop_extrema[start_extremum]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                if zero_platos[closest_zero_plato_i][1] < start_stop_extrema[start_extremum]:
                    key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find ends of phase0 and phase1 -----------------------------------
        # Use the first point below a baseline (calculated to be between the local max and min) from the left and from the right
        # as the start and stop respectively.
        if middle_extremum_1 is None:
            phase0_end = middle_extrema[middle_extremum_0]
            phase1_end = phase0_end
        else:
            if interval_starts_with_minima:
                relative_upper = min(data[middle_extrema[middle_extremum_0]], data[middle_extrema[middle_extremum_1]])
                relative_lower = max(0, min(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]]))
                # baseline = relative_upper - (relative_upper - relative_lower) / 2.3
            else:
                relative_upper = max(data[middle_extrema[middle_extremum_0]], data[middle_extrema[middle_extremum_1]])
                relative_lower = min(0, max(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]]))
                # baseline = relative_upper - (relative_upper - relative_lower) / 2.3
            baseline = (relative_upper + relative_lower) / 2
            factor = 1 - 2 * (1 - int(interval_starts_with_minima))  # Correct for searching below or above 0.
            phase0_end = next((i for i, el in enumerate(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]], middle_extrema[middle_extremum_0])
                               if el*factor < baseline*factor))
            phase1_end = next((i for el, i in zip(data[middle_extrema[middle_extremum_1]:middle_extrema[middle_extremum_0]:-1], range(middle_extrema[middle_extremum_1], 0, -1))
                               if el*factor < baseline*factor))

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[stop_extremum]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > start_stop_extrema[stop_extremum]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None and zero_platos[closest_zero_plato_i][0] > start_stop_extrema[stop_extremum]:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        if interval_start > phase0_end or phase0_end > phase1_end or phase1_end > interval_stop:
            print_red("Acceleration: Interval is wrong")
            print_red(interval_start, phase0_end, phase1_end, interval_stop)
            exit()

        detected_intervals.append((interval_start, phase0_end, phase1_end, interval_stop))

        middle_extrema_i += 1 + int((middle_extremum_1 is not None))

    return split_overlapping_intervals(detected_intervals)
    # return detected_intervals


def analyze_v3_1_2(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """More fine tuning, significantly improved v3.1.1."""

    # Best (and same) results (21 files, 17.02.23) from ~2.95 - ~2.9725 for data -> pca + lowPass(2.5, 100)
    min_baseline_factor = 3.05
    max_baseline_factor = 3.05

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline > 0 or maxima_baseline < 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)
    # _, zero_points, zero_platos = calc_key_features(data, time_stamps)
    # zero_platos = calc_zero_platos(data, derivative(data, time_stamps))
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.02, 0.4, 0, 0)

    if len(minima) != len(maxima):
        print_conditional(verbose, f"{YELLOW[0]}#minima ({len(minima)}) != #maxima ({len(maxima)}){YELLOW[1]}")

    # Requirement for at least one motion interval to exist / be detected by this method.
    if len(minima) + len(maxima) < 3 or len(minima) == 0 or len(maxima) == 0:
        return []

    # TODO: This is potentially more dangerous for linear acceleration. There are more odd bumps, similar to 'real' bumps.
    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        start_stop_extrema = minima
        middle_extrema = maxima
    else:
        start_stop_extrema = maxima
        middle_extrema = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    middle_extrema_i = 0
    for start_stop_extrema_i in range(len(start_stop_extrema)-1):
        start_extremum = start_stop_extrema_i
        stop_extremum = start_stop_extrema_i+1
        middle_extremum_0 = next((i for i, el in enumerate(middle_extrema[middle_extrema_i:], middle_extrema_i) if el > start_stop_extrema[start_extremum]), None)
        if middle_extremum_0 is None:
            break
        if middle_extrema[middle_extremum_0] > start_stop_extrema[stop_extremum]:
            continue
        middle_extremum_1 = None
        if middle_extremum_0+1 < len(middle_extrema) and middle_extrema[middle_extremum_0+1] < start_stop_extrema[stop_extremum]:
            middle_extremum_1 = middle_extremum_0+1
            if middle_extremum_0+2 < len(middle_extrema) and middle_extrema[middle_extremum_0+2] < start_stop_extrema[stop_extremum]:
                middle_extrema_i += 3
                continue

        # Valid interval found.
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[start_extremum])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +3 since there are at least two more zero points between the first and last extrema that are of no interest.
            zero_point_i = closest_left_zero_point_i + 3
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > start_stop_extrema[start_extremum]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                if zero_platos[closest_zero_plato_i][1] < start_stop_extrema[start_extremum]:
                    key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find ends of phase0 and phase1 -----------------------------------
        # Use the first point below a baseline (calculated to be between the local max and min) from the left and from the right
        # as the start and stop respectively.
        if middle_extremum_1 is None:
            phase0_end = middle_extrema[middle_extremum_0]
            phase1_end = phase0_end
        else:
            if interval_starts_with_minima:
                relative_upper = min(data[middle_extrema[middle_extremum_0]], data[middle_extrema[middle_extremum_1]])
                relative_lower = max(0, min(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]]))
                # baseline = relative_upper - (relative_upper - relative_lower) / 2.3
            else:
                relative_upper = max(data[middle_extrema[middle_extremum_0]], data[middle_extrema[middle_extremum_1]])
                relative_lower = min(0, max(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]]))
                # baseline = relative_upper - (relative_upper - relative_lower) / 2.3
            baseline = (relative_upper + relative_lower) / 2
            factor = 1 - 2 * (1 - int(interval_starts_with_minima))  # Correct for searching below or above 0.
            phase0_end = next((i for i, el in enumerate(data[middle_extrema[middle_extremum_0]:middle_extrema[middle_extremum_1]], middle_extrema[middle_extremum_0])
                               if el*factor < baseline*factor))
            phase1_end = next((i for el, i in zip(data[middle_extrema[middle_extremum_1]:middle_extrema[middle_extremum_0]:-1], range(middle_extrema[middle_extremum_1], 0, -1))
                               if el*factor < baseline*factor))

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > start_stop_extrema[stop_extremum]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > start_stop_extrema[stop_extremum]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None and zero_platos[closest_zero_plato_i][0] > start_stop_extrema[stop_extremum]:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        if interval_start > phase0_end or phase0_end > phase1_end or phase1_end > interval_stop:
            print_red("Acceleration: Interval is wrong")
            print_red(interval_start, phase0_end, phase1_end, interval_stop)
            exit()

        detected_intervals.append((interval_start, phase0_end, phase1_end, interval_stop))

        middle_extrema_i += 1 + int((middle_extremum_1 is not None))

    return split_overlapping_intervals(detected_intervals)
    # return detected_intervals


def split_overlapping_intervals(intervals: List[Tuple[int, int]] | List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """Works without and with phases"""
    converted_intervals = [list(interval) for interval in intervals]
    for i in range(len(intervals)-1):
        if intervals[i][-1] > intervals[i+1][0]:
            split = int((intervals[i][-1] + intervals[i+1][0]) / 2)
            converted_intervals[i][-1] = converted_intervals[i+1][0] = split
    return [tuple(interval) for interval in converted_intervals]


def analyze_rot_rate_v1(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int]]:
    """
    Maxima and minima are used as start or stop of an initial guess of an interval respectively.
    Between a potential interval (start (eg. maxima), stop (-> minima)) no other maxima or minima
    is allowed.
    The actual start and stop are found by using the first zero point outside of the maxima/minima
    pair. So the first zero point on the left is the final start and the first zero point on the
    right is the stop. If no zero point exists on these sides then the start is time point 0 for the left
    and the end of the recording for the right.

    How Extrema are found:
        Maxima/Minima must be above (-> in absolute value terms) a specified baseline (half the global max/min).
        For each continues cluster of points above the baseline (half max/min) the max/min of these points
        is used as the extremum.

    Potential Problems:
        - If one motion cycle is due to whatever reason double the speed then most or all other
        cycle then this analysis will miss many or all of the other cycles.
        - If there is a short strong movement in the opposite direction of the prior movement eg.
        when stopping after the first half of the movement cycle, then that might be interpreted
        as the stop. -> maybe increase the baseline or use outlier detection (mean+variance) for
        the extrema.
        - If a false maxima or minima is detected first, while actually the opposite is the actual
        minima (when maxima) / maxima (when minima) then every classification will be wrong.
    """

    min_baseline_factor = 2
    max_baseline_factor = 2

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline >= 0 or maxima_baseline <= 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)

    # _, zero_points, _ = calc_key_features(data, time_stamps)
    zero_points, _ = calc_zero_points_platos(data, derivative(data, time_stamps), 0.20, 0.20*2, 1/10, 1/10)

    # No zero points means no intervals for this analysis method.
    # TODO: This check is redundant due to the next check. Remove it in every function.
    if len(zero_points) == 0:
        return []

    if len(minima) == 0 or len(maxima) == 0:
        return []

    if len(minima) != len(maxima):
        print_conditional(verbose, f"#minima ({len(minima)}) != #maxima ({len(maxima)})")

    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        starts = minima
        stops = maxima
    else:
        starts = maxima
        stops = minima

    detected_intervals = []
    zero_point_i = 0
    start_i = stop_i = 0
    while start_i < len(starts) and stop_i < len(stops):
        if starts[start_i] > stops[stop_i]:
            stop_i += 1
            continue
        if start_i + 1 < len(starts) and starts[start_i+1] < stops[stop_i]:
            start_i += 1
            continue
        # Start stop pair found.
        # Find edges of interval.
        # Will always return a value (default (None) is not required) as long as zero_points is not empty
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > starts[start_i])
        )
        if closest_left_zero_point_i >= 0:
            interval_start = zero_points[closest_left_zero_point_i]
            # +2 since start != stop of same interval and the next zero point is always between the two extrema.
            # TODO: +2 comment
            zero_point_i = closest_left_zero_point_i + 1
        else:
            interval_start = 0
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_right_zero_point_i is not None:
            interval_stop = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            interval_stop = len(time_stamps) - 1

        detected_intervals.append((interval_start, interval_stop))

        start_i += 1
        stop_i += 1

    return detected_intervals


def analyze_rot_rate_v2(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """Changes to v1: The interval is now split up into the three phases of an exercise."""

    # Find end of phase 0 and end of phase 1:
    # Method 1: Used
    #  - end phase 0 is most often zero point
    #  - use horizontal line left to the second extremum and lower it until
    #    it touches two points on the graph - Interval [end phase 0, 2.extremum]
    # Method 2: Not Used
    #  - same as above
    #  - use peak of 2. derivative -> if the curve is nice then very useful
    #    if not it might be completely broken. The latter happens frequently.
    # Mixture of both maybe.

    min_baseline_factor = 2
    max_baseline_factor = 2

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline >= 0 or maxima_baseline <= 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)

    # _, zero_points, _ = calc_key_features(data, time_stamps)
    zero_points, _ = calc_zero_points_platos(data, derivative(data, time_stamps), 0.20, 0.20*2, 1/10, 1/10)

    # No zero points means no intervals for this analysis method.
    if len(zero_points) == 0:
        return []

    if len(minima) == 0 or len(maxima) == 0:
        return []

    if len(minima) != len(maxima):
        print_conditional(verbose, f"#minima ({len(minima)}) != #maxima ({len(maxima)})")

    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        starts = minima
        stops = maxima
    else:
        starts = maxima
        stops = minima

    detected_intervals = []
    zero_point_i = 0
    start_i = stop_i = 0
    while start_i < len(starts) and stop_i < len(stops):
        if starts[start_i] > stops[stop_i]:
            stop_i += 1
            continue
        if start_i + 1 < len(starts) and starts[start_i+1] < stops[stop_i]:
            start_i += 1
            continue
        # Start stop pair found.
        # Find edges of interval and end of phase 0 and phase 1:
        # Find start:
        # Will always return a value (default (None) is not required) as long as zero_points is not empty
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > starts[start_i])
        )
        if closest_left_zero_point_i >= 0:
            interval_start = zero_points[closest_left_zero_point_i]
            # +2 since start != stop of same interval and the next zero point is always between the two extrema.
            # TODO: +2 comment
            zero_point_i = closest_left_zero_point_i + 1
        else:
            interval_start = 0

        # Find end of phase 0:
        closest_right_zero_point_i = closest_left_zero_point_i + 1
        end_phase0 = zero_points[closest_right_zero_point_i]

        # Find end of phase 1:
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_left_zero_point_i is not None:
            end_phase1 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side of the second extremum and only look at points on the right of it.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            end_phase1 = zero_points[-1]

        # Find end:
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_right_zero_point_i is not None:
            interval_stop = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            interval_stop = len(time_stamps) - 1

        detected_intervals.append((interval_start, end_phase0, end_phase1, interval_stop))

        start_i += 1
        stop_i += 1

    return detected_intervals


def analyze_rot_rate_v2_1(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """Now additionally uses the edges of Zero platos as start/stop points for all phases."""

    min_baseline_factor = 2
    max_baseline_factor = 2

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline >= 0 or maxima_baseline <= 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)

    # _, zero_points, _ = calc_key_features(data, time_stamps)
    # zero_platos = calc_zero_platos(data, derivative(data, time_stamps))
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.15, 0.79, 0, 0.1)

    # No zero points means no intervals for this analysis method.
    if len(zero_points) == 0:
        return []

    if len(minima) == 0 or len(maxima) == 0:
        return []

    if len(minima) != len(maxima):
        print_conditional(verbose, f"#minima ({len(minima)}) != #maxima ({len(maxima)})")

    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        starts = minima
        stops = maxima
    else:
        starts = maxima
        stops = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    start_i = stop_i = 0
    while start_i < len(starts) and stop_i < len(stops):
        if starts[start_i] > stops[stop_i]:
            stop_i += 1
            continue
        if start_i + 1 < len(starts) and starts[start_i+1] < stops[stop_i]:
            start_i += 1
            continue
        # Start stop pair found.
        # Find edges of interval and end of phase 0 and phase 1:
        # Will always return a value (default (None) is not required) as long as zero_points is not empty
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > starts[start_i])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side and only look at points on the right of the extremum.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > starts[start_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find end of phase 0 ----------------------------------------------
        # Find closest end of phase 0 / start of phase 1 from the right of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = closest_left_zero_point_i + 1
        key_point_candidate0 = zero_points[closest_right_zero_point_i]
        # Find start of closest zero_plato:
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > starts[start_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase0 = min(key_point_candidate0, key_point_candidate1)
        else:
            end_phase0 = key_point_candidate0

        # -- Find end of phase 1 ----------------------------------------------
        # Find closest end of phase 1 / start of phase 2 from the left of the second extremum.
        # TODO: Add comment.
        key_point_candidate0 = None
        key_point_candidate1 = None
        # Find closest zero point to the left of the second exremum.
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_left_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side of the second extremum and only look at points on the right of it.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            key_point_candidate0 = zero_points[-1]
        # Find stop of closest plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > stops[stop_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase1 = max(key_point_candidate0, key_point_candidate1)
        else:
            end_phase1 = key_point_candidate0

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > stops[stop_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        if interval_start > end_phase0 or end_phase0 > end_phase1 or end_phase1 > interval_stop:
            print_red("Rotation: Interval is wrong")
            print_red(interval_start, end_phase0, end_phase1, interval_stop)

        detected_intervals.append((interval_start, end_phase0, end_phase1, interval_stop))

        start_i += 1
        stop_i += 1

    return detected_intervals


def analyze_rot_rate_v2_1_1(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """This version fixes the bug with the zero platos extending over extrema"""

    min_baseline_factor = 2
    max_baseline_factor = 2

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline >= 0 or maxima_baseline <= 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)

    # _, zero_points, _ = calc_key_features(data, time_stamps)
    # zero_platos = calc_zero_platos(data, derivative(data, time_stamps))
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.15, 0.79, 0, 0.1)

    # No zero points means no intervals for this analysis method.
    if len(zero_points) == 0:
        return []

    if len(minima) == 0 or len(maxima) == 0:
        return []

    if len(minima) != len(maxima):
        print_conditional(verbose, f"#minima ({len(minima)}) != #maxima ({len(maxima)})")

    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        starts = minima
        stops = maxima
    else:
        starts = maxima
        stops = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    start_i = stop_i = 0
    while start_i < len(starts) and stop_i < len(stops):
        if starts[start_i] > stops[stop_i]:
            stop_i += 1
            continue
        if start_i + 1 < len(starts) and starts[start_i+1] < stops[stop_i]:
            start_i += 1
            continue
        # Start stop pair found.
        # Find edges of interval and end of phase 0 and phase 1:
        # Will always return a value (default (None) is not required) as long as zero_points is not empty
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > starts[start_i])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side and only look at points on the right of the extremum.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > starts[start_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                if zero_platos[closest_zero_plato_i][1] < starts[start_i]:
                    key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find end of phase 0 ----------------------------------------------
        # Find closest end of phase 0 / start of phase 1 from the right of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = closest_left_zero_point_i + 1
        key_point_candidate0 = zero_points[closest_right_zero_point_i]
        # Find start of closest zero_plato:
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > starts[start_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None and zero_platos[closest_zero_plato_i][0] > starts[start_i]:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase0 = min(key_point_candidate0, key_point_candidate1)
        else:
            end_phase0 = key_point_candidate0

        # -- Find end of phase 1 ----------------------------------------------
        # Find closest end of phase 1 / start of phase 2 from the left of the second extremum.
        # TODO: Add comment.
        key_point_candidate0 = None
        key_point_candidate1 = None
        # Find closest zero point to the left of the second exremum.
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_left_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side of the second extremum and only look at points on the right of it.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            key_point_candidate0 = zero_points[-1]
        # Find stop of closest plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > stops[stop_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                if zero_platos[closest_zero_plato_i][1] < stops[stop_i]:
                    key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase1 = max(key_point_candidate0, key_point_candidate1)
        else:
            end_phase1 = key_point_candidate0

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > stops[stop_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None and zero_platos[closest_zero_plato_i][0] > stops[stop_i]:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        if interval_start > end_phase0 or end_phase0 > end_phase1 or end_phase1 > interval_stop:
            print_red("Rotation: Interval is wrong")
            print_red(interval_start, end_phase0, end_phase1, interval_stop)

        detected_intervals.append((interval_start, end_phase0, end_phase1, interval_stop))

        start_i += 1
        stop_i += 1

    return detected_intervals


def analyze_rot_rate_v2_1_1g(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """Same as v2.1.1 but optimized for gravity"""

    min_baseline_factor = 2
    max_baseline_factor = 2

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline >= 0 or maxima_baseline <= 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)

    # _, zero_points, _ = calc_key_features(data, time_stamps)
    # zero_platos = calc_zero_platos(data, derivative(data, time_stamps))
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.1, 0.75, 0, 0.1)

    # No zero points means no intervals for this analysis method.
    if len(zero_points) == 0:
        return []

    if len(minima) == 0 or len(maxima) == 0:
        return []

    if len(minima) != len(maxima):
        print_conditional(verbose, f"#minima ({len(minima)}) != #maxima ({len(maxima)})")

    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        starts = minima
        stops = maxima
    else:
        starts = maxima
        stops = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    start_i = stop_i = 0
    while start_i < len(starts) and stop_i < len(stops):
        if starts[start_i] > stops[stop_i]:
            stop_i += 1
            continue
        if start_i + 1 < len(starts) and starts[start_i+1] < stops[stop_i]:
            start_i += 1
            continue
        # Start stop pair found.
        # Find edges of interval and end of phase 0 and phase 1:
        # Will always return a value (default (None) is not required) as long as zero_points is not empty
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > starts[start_i])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side and only look at points on the right of the extremum.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > starts[start_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                if zero_platos[closest_zero_plato_i][1] < starts[start_i]:
                    key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find end of phase 0 ----------------------------------------------
        # Find closest end of phase 0 / start of phase 1 from the right of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = closest_left_zero_point_i + 1
        key_point_candidate0 = zero_points[closest_right_zero_point_i]
        # Find start of closest zero_plato:
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > starts[start_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None and zero_platos[closest_zero_plato_i][0] > starts[start_i]:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase0 = min(key_point_candidate0, key_point_candidate1)
        else:
            end_phase0 = key_point_candidate0

        # -- Find end of phase 1 ----------------------------------------------
        # Find closest end of phase 1 / start of phase 2 from the left of the second extremum.
        # TODO: Add comment.
        key_point_candidate0 = None
        key_point_candidate1 = None
        # Find closest zero point to the left of the second exremum.
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_left_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side of the second extremum and only look at points on the right of it.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            key_point_candidate0 = zero_points[-1]
        # Find stop of closest plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > stops[stop_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                if zero_platos[closest_zero_plato_i][1] < stops[stop_i]:
                    key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase1 = max(key_point_candidate0, key_point_candidate1)
        else:
            end_phase1 = key_point_candidate0

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > stops[stop_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None and zero_platos[closest_zero_plato_i][0] > stops[stop_i]:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        if interval_start > end_phase0 or end_phase0 > end_phase1 or end_phase1 > interval_stop:
            print_red("Rotation: Interval is wrong")
            print_red(interval_start, end_phase0, end_phase1, interval_stop)

        detected_intervals.append((interval_start, end_phase0, end_phase1, interval_stop))

        start_i += 1
        stop_i += 1

    return detected_intervals


def analyze_rot_rate_v2_2(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    Based on v2.1
    Change: Additionally uses horizontal line to find end of pase0.
    -> Is worse than v2.1

    """

    min_baseline_factor = 2
    max_baseline_factor = 2

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline >= 0 or maxima_baseline <= 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)

    # _, zero_points, _ = calc_key_features(data, time_stamps)
    # zero_platos = calc_zero_platos(data, derivative(data, time_stamps))
    zero_points, zero_platos = calc_zero_points_platos(data, derivative(data, time_stamps), 0.15, 0.79, 0, 0.1)

    # No zero points means no intervals for this analysis method.
    if len(zero_points) == 0:
        return []

    if len(minima) == 0 or len(maxima) == 0:
        return []

    if len(minima) != len(maxima):
        print_conditional(verbose, f"#minima ({len(minima)}) != #maxima ({len(maxima)})")

    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        starts = minima
        stops = maxima
    else:
        starts = maxima
        stops = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    start_i = stop_i = 0
    while start_i < len(starts) and stop_i < len(stops):
        if starts[start_i] > stops[stop_i]:
            stop_i += 1
            continue
        if start_i + 1 < len(starts) and starts[start_i+1] < stops[stop_i]:
            start_i += 1
            continue
        # Start stop pair found.
        # Find edges of interval and end of phase 0 and phase 1:
        # Will always return a value (default (None) is not required) as long as zero_points is not empty
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > starts[start_i])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side and only look at points on the right of the extremum.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > starts[start_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find end of phase 0 ----------------------------------------------
        # Find closest end of phase 0 / start of phase 1 from the right of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = closest_left_zero_point_i + 1
        key_point_candidate0 = zero_points[closest_right_zero_point_i]
        # Find start of closest zero_plato:
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > starts[start_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase0 = min(key_point_candidate0, key_point_candidate1)
        else:
            end_phase0 = key_point_candidate0

        # -- Find end of phase 1 ----------------------------------------------
        # Find closest end of phase 1 / start of phase 2 from the left of the second extremum.
        # TODO: Add comment.
        key_point_candidate0 = None
        key_point_candidate1 = None
        end_phase0_to_snd_extremum = list(enumerate(data[stops[stop_i]:end_phase0:-1]))
        end_phase0_to_snd_extremum.sort(key=itemgetter(1), reverse=interval_starts_with_minima)
        for i, (j, val) in enumerate(end_phase0_to_snd_extremum):
            if i != j:
                assert i > 0, print(f"i = {i} -> Maxima/Minima calculation must be wrong.")
                key_point_candidate0 = stops[stop_i] - (i - 1)
                break
        else:
            key_point_candidate0 = end_phase0
        # Find stop of closest plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > stops[stop_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase1 = max(key_point_candidate0, key_point_candidate1)
        else:
            end_phase1 = key_point_candidate0

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > stops[stop_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        if interval_start > end_phase0 or end_phase0 > end_phase1 or end_phase1 > interval_stop:
            print_red("Rotation: Interval is wrong")
            print_red(interval_start, end_phase0, end_phase1, interval_stop)
            exit()

        detected_intervals.append((interval_start, end_phase0, end_phase1, interval_stop))

        start_i += 1
        stop_i += 1

    return detected_intervals


def analyze_rot_rate_v2_3(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    Based on roRateV2.1
    Change:
        - zero plato calculation settings dynamically set
    This method overall has a higher performance than v2.1 (and v2.2)
    """

    min_baseline_factor = 2
    max_baseline_factor = 2

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline >= 0 or maxima_baseline <= 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)

    abs_min_extrema = min(*map(lambda x: abs(data[x]), minima), *map(lambda x: abs(data[x]), maxima))
    d = derivative(data, time_stamps)
    max_d = max(np.abs(d))
    # max_val_zero_plato = max_val / 13.3333333333
    # max_slope_zero_plato = max_d / 10.1265822785
    max_val_zero_plato = abs_min_extrema / 5.8
    max_slope_zero_plato = max_d / 7

    # _, zero_points, _ = calc_key_features(data, time_stamps)
    # zero_platos = calc_zero_platos(data, derivative(data, time_stamps))
    zero_points, zero_platos = calc_zero_points_platos(data, d, max_val_zero_plato, max_slope_zero_plato, 0, 0.1)

    # No zero points means no intervals for this analysis method.
    if len(zero_points) == 0:
        return []

    if len(minima) == 0 or len(maxima) == 0:
        return []

    if len(minima) != len(maxima):
        print_conditional(verbose, f"#minima ({len(minima)}) != #maxima ({len(maxima)})")

    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        starts = minima
        stops = maxima
    else:
        starts = maxima
        stops = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    start_i = stop_i = 0
    while start_i < len(starts) and stop_i < len(stops):
        if starts[start_i] > stops[stop_i]:
            stop_i += 1
            continue
        if start_i + 1 < len(starts) and starts[start_i+1] < stops[stop_i]:
            start_i += 1
            continue
        # Start stop pair found.
        # Find edges of interval and end of phase 0 and phase 1:
        # Will always return a value (default (None) is not required) as long as zero_points is not empty
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > starts[start_i])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side and only look at points on the right of the extremum.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > starts[start_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find end of phase 0 ----------------------------------------------
        # Find closest end of phase 0 / start of phase 1 from the right of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = closest_left_zero_point_i + 1
        key_point_candidate0 = zero_points[closest_right_zero_point_i]
        # Find start of closest zero_plato:
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > starts[start_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase0 = min(key_point_candidate0, key_point_candidate1)
        else:
            end_phase0 = key_point_candidate0

        # -- Find end of phase 1 ----------------------------------------------
        # Find closest end of phase 1 / start of phase 2 from the left of the second extremum.
        # TODO: Add comment.
        key_point_candidate0 = None
        key_point_candidate1 = None
        # Find closest zero point to the left of the second exremum.
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_left_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side of the second extremum and only look at points on the right of it.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            key_point_candidate0 = zero_points[-1]
        # Find stop of closest plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > stops[stop_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase1 = max(key_point_candidate0, key_point_candidate1)
        else:
            end_phase1 = key_point_candidate0

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > stops[stop_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        if interval_start > end_phase0 or end_phase0 > end_phase1 or end_phase1 > interval_stop:
            print_red("Rotation: Interval is wrong")
            print_red(interval_start, end_phase0, end_phase1, interval_stop)

        detected_intervals.append((interval_start, end_phase0, end_phase1, interval_stop))

        start_i += 1
        stop_i += 1

    return detected_intervals


def analyze_rot_rate_v2_3g(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """Same as v2.3 but optimized for gravity, but has worse scores than using rotRateV2.1.1."""

    min_baseline_factor = 2.3
    max_baseline_factor = 2.3

    if gdv.DEBUG_ENABLED:
        if gdv.gdv_analysis_minima_baseline_factor is not None:
            min_baseline_factor = gdv.gdv_analysis_minima_baseline_factor
        if gdv.gdv_analysis_maxima_baseline_factor is not None:
            max_baseline_factor = gdv.gdv_analysis_maxima_baseline_factor

    minima_baseline = min(data) / min_baseline_factor
    maxima_baseline = max(data) / max_baseline_factor

    if minima_baseline >= 0 or maxima_baseline <= 0:
        return []

    minima, maxima = calc_extrema_with_baseline(data, minima_baseline, maxima_baseline)

    abs_min_extrema = min(*map(lambda x: abs(data[x]), minima), *map(lambda x: abs(data[x]), maxima))
    d = derivative(data, time_stamps)
    max_d = max(np.abs(d))
    # max_val_zero_plato = max_val / 13.3333333333
    # max_slope_zero_plato = max_d / 10.1265822785
    max_val_zero_plato = abs_min_extrema / 9
    max_slope_zero_plato = max_d / 10

    # _, zero_points, _ = calc_key_features(data, time_stamps)
    # zero_platos = calc_zero_platos(data, derivative(data, time_stamps))
    zero_points, zero_platos = calc_zero_points_platos(data, d, max_val_zero_plato, max_slope_zero_plato, 0, 0.1)

    # No zero points means no intervals for this analysis method.
    if len(zero_points) == 0:
        return []

    if len(minima) == 0 or len(maxima) == 0:
        return []

    if len(minima) != len(maxima):
        print_conditional(verbose, f"#minima ({len(minima)}) != #maxima ({len(maxima)})")

    interval_starts_with_minima = minima[0] < maxima[0]

    if interval_starts_with_minima:
        starts = minima
        stops = maxima
    else:
        starts = maxima
        stops = minima

    detected_intervals = []
    zero_point_i = 0
    zero_plato_i = 0 if len(zero_platos) > 0 else None
    start_i = stop_i = 0
    while start_i < len(starts) and stop_i < len(stops):
        if starts[start_i] > stops[stop_i]:
            stop_i += 1
            continue
        if start_i + 1 < len(starts) and starts[start_i+1] < stops[stop_i]:
            start_i += 1
            continue
        # Start stop pair found.
        # Find edges of interval and end of phase 0 and phase 1:
        # Will always return a value (default (None) is not required) as long as zero_points is not empty
        # -- Find start point -------------------------------------------------
        # Find closest potential starting point on the left of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > starts[start_i])
        )
        if closest_left_zero_point_i >= 0:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side and only look at points on the right of the extremum.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            key_point_candidate0 = 0
        # Find stop of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > starts[start_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_start = max(key_point_candidate0, key_point_candidate1)
        else:
            interval_start = key_point_candidate0

        # -- Find end of phase 0 ----------------------------------------------
        # Find closest end of phase 0 / start of phase 1 from the right of the first extremum.
        # Find closest zero point.
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = closest_left_zero_point_i + 1
        key_point_candidate0 = zero_points[closest_right_zero_point_i]
        # Find start of closest zero_plato:
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > starts[start_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase0 = min(key_point_candidate0, key_point_candidate1)
        else:
            end_phase0 = key_point_candidate0

        # -- Find end of phase 1 ----------------------------------------------
        # Find closest end of phase 1 / start of phase 2 from the left of the second extremum.
        # TODO: Add comment.
        key_point_candidate0 = None
        key_point_candidate1 = None
        # Find closest zero point to the left of the second exremum.
        closest_left_zero_point_i = next(
            (i-1 for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_left_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_left_zero_point_i]
            # +1 since we are now finished with the left side of the second extremum and only look at points on the right of it.
            zero_point_i = closest_left_zero_point_i + 1
        else:
            key_point_candidate0 = zero_points[-1]
        # Find stop of closest plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i-1 for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[1] > stops[stop_i]), None)
            if closest_zero_plato_i is None or closest_zero_plato_i >= 0:
                zero_plato_i = closest_zero_plato_i
                if closest_zero_plato_i is None:
                    closest_zero_plato_i = len(zero_platos) - 1
                key_point_candidate1 = zero_platos[closest_zero_plato_i][1]
        # Select closest:
        if key_point_candidate1 is not None:
            end_phase1 = max(key_point_candidate0, key_point_candidate1)
        else:
            end_phase1 = key_point_candidate0

        # -- Find end point ---------------------------------------------------
        # Find closest end of the motion interval from the right of the second extremum.
        # Find closest zero_point
        key_point_candidate0 = None
        key_point_candidate1 = None
        closest_right_zero_point_i = next(
            (i for i, el in enumerate(zero_points[zero_point_i:], zero_point_i)
             if el > stops[stop_i]),
            None
        )
        if closest_right_zero_point_i is not None:
            key_point_candidate0 = zero_points[closest_right_zero_point_i]
            zero_point_i = closest_right_zero_point_i
        else:
            key_point_candidate0 = len(time_stamps) - 1
        # Find start of closest zero plato.
        if zero_plato_i is not None:
            closest_zero_plato_i = next((i for i, interval in enumerate(zero_platos[zero_plato_i:], zero_plato_i) if interval[0] > stops[stop_i]), None)
            zero_plato_i = closest_zero_plato_i
            if closest_zero_plato_i is not None:
                key_point_candidate1 = zero_platos[closest_zero_plato_i][0]
        # Select closest:
        if key_point_candidate1 is not None:
            interval_stop = min(key_point_candidate0, key_point_candidate1)
        else:
            interval_stop = key_point_candidate0
        # ---------------------------------------------------------------------

        if interval_start > end_phase0 or end_phase0 > end_phase1 or end_phase1 > interval_stop:
            print_red("Rotation: Interval is wrong")
            print_red(interval_start, end_phase0, end_phase1, interval_stop)

        detected_intervals.append((interval_start, end_phase0, end_phase1, interval_stop))

        start_i += 1
        stop_i += 1

    return detected_intervals


def analyze_gravity_v1(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """Converts gravity to rotation rate and uses the rotation rate analysis method.
    Derivative of gravity is roughly rotation rate -> Very similar to the rotation rate calculated
    by Core Motion from Apple in iOS.
    """
    gravity_lowPass_derivative1 = derivative(data, time_stamps)
    detected_intervals = analyze_rot_rate_v2(gravity_lowPass_derivative1, time_stamps, verbose)
    return detected_intervals


def analyze_gravity_v1_1(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    gravity_lowPass_derivative1 = derivative(data, time_stamps)
    detected_intervals = analyze_rot_rate_v2_1(gravity_lowPass_derivative1, time_stamps, verbose)
    return detected_intervals


def analyze_gravity_v1_1_1(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    gravity_lowPass_derivative1 = derivative(data, time_stamps)
    detected_intervals = analyze_rot_rate_v2_1_1g(gravity_lowPass_derivative1, time_stamps, verbose)
    return detected_intervals


def analyze_gravity_v1_2(data: list, time_stamps: list, verbose: bool = True) -> List[Tuple[int, int, int, int]]:
    """Worse than v1.1.1"""
    gravity_lowPass_derivative1 = derivative(data, time_stamps)
    detected_intervals = analyze_rot_rate_v2_3g(gravity_lowPass_derivative1, time_stamps, verbose)
    return detected_intervals


# def analyze_gravity_v2(data: list, time_stamps: list, *_) -> List[Tuple[int, int, int, int]]:
#     platos = calc_platos(derivative(data), time_stamps)
#     avg_heights_of_platos = [avg(data, *plato) for plato in platos]

#     # Get up and down regions of the gravity curve (splitting: half of min+max height)
#     split = (min(data) + max(data)) / 2
#     _, switch_up_down_points, _ = calc_key_features(np.array[data] - split, time_stamps)
#     switch_up_down_points.insert(0, 0)

#     plato_i = 0
#     for switch_points_i in range(len(switch_up_down_points)-1):
#         up_down_region_start = switch_up_down_points[switch_points_i]
#         up_down_region_stop = switch_up_down_points[switch_points_i+1]

#         furthest_plato_from_split = None
#         furthest_val_from_split = 0
#         while plato_i < len(platos):
#             plato_start, plato_stop = platos[plato_i]
#             plato_start = max(plato_start, up_down_region_start)
#             if plato_start >= up_down_region_stop:
#                 break
#             plato_stop = min(plato_stop, up_down_region_stop)
#             if plato_start >= plato_stop:
#                 plato_i += 1
#                 continue
#             if avg_heights_of_platos[plato_i] > furthest_val_from_split:
#                 furthest_val_from_split = avg_heights_of_platos[plato_i]
#                 furthest_plato_from_split = (plato_start, plato_stop)

#         if furthest_plato_from_split is None:
#             pass
