"""Author: Nicola Vidovic

Here all functions that run/test all analysis methods on the provided dataset are included.

Most of the functions here are deprecated, but I have left them in for the history. The most
important one is the last one, capable of running all versions of all analysis methods.

"""

import os
from itertools import chain
from math import sqrt

from analyze import *
from FileIO import *
from functions import *
from global_constants import *
from transform import *


def statistics_of_all_truth():
    for folder_name in os.listdir(RECORDINGS_PATH):
        print(f"----------- {folder_name} -----------")

        data = get_data_from_zip(folder_name)
        annotations = get_annotation_from_folder(folder_name)
        if data is None or annotations is None:
            print("Some file(s) is/are missing")
            continue

        # Read and calculate all needed Data from the recordings JSON.
        frames = data["frames"]
        acceleration_vectors = get_acceleration(frames)
        reduced_data = PCA(acceleration_vectors, 3)
        reduced_data_multi_smooth = multi_smooth(reduced_data)
        reduced_data_multi_smooth_pc0 = reduced_data_multi_smooth[:, 0]

        time_stamps = get_time_stamps(frames)
        _, zero_points, _ = calc_key_features(reduced_data_multi_smooth_pc0, time_stamps)

        # Construct full motion intervals form their parts.
        interval_parts = annotations["intervals"]
        intervals = get_intervals_from_parts(interval_parts, time_stamps)

        # Get Statistics.
        print_statistics_from_intervals(
            intervals, reduced_data_multi_smooth_pc0, time_stamps, zero_points)


def get_statistics_from(folder_name: str):
    print(f"----------- {folder_name} -----------")

    data = get_data_from_zip(folder_name)
    annotations = get_annotation_from_folder(folder_name)
    if data is None or annotations is None:
        print("Some file(s) is/are missing")
        return

    # Read and calculate all needed Data from the recordings JSON.
    frames = data["frames"]
    acceleration_vectors = get_acceleration(frames)
    reduced_data = PCA(acceleration_vectors, 3)
    reduced_data_multi_smooth = multi_smooth(reduced_data)
    reduced_data_multi_smooth_pc0 = reduced_data_multi_smooth[:, 0]

    time_stamps = get_time_stamps(frames)
    _, zero_points, _ = calc_key_features(reduced_data_multi_smooth_pc0, time_stamps)

    # Construct full motion intervals form their parts.
    interval_parts = annotations["intervals"]
    intervals = get_intervals_from_parts(interval_parts, time_stamps)

    for start, stop in intervals:
        print("Interval:", round(time_stamps[start], 3), "-", round(time_stamps[stop], 3),
              f"({round(time_stamps[stop]-time_stamps[start], 3)})")
        print(" SumOfVals (OGI):", sum(acceleration_vectors[start:stop+1, 1]))
        print(" SumOfVals (PCA):", sum(reduced_data[start:stop+1, 0]))
        print(" SumOfVals (PSM):", sum(reduced_data_multi_smooth[start:stop+1, 0]))


def compare_guesses_to_truth_with_phases(time_stamps: list, truth_intervals: list, guessed_intervals: list, verbose: bool = True) -> Tuple[int, int, float, float, float, float, list, list, list, list]:
    """Deprecated
    Assumes that the interval lists are sorted in acceding order of their start times. TODO: Check if this is still true.

    When the guessed interval's start and stop are close to the start and stop of a real interval they are matched.
    The error for the guessed interval is the absolute difference between the length of the guessed interval and
    the length of the true interval for each phase and total length.
    If only the start or only the end are near the true interval then the guessed interval is put into of_interest.
    Else in misses.

    - Worst matched interval is determined by the worst length of interval error -> Phases are not taken into account.
    """

    # Try to match the guessed intervals to truth.
    # If start and stop are near truth's start and stop -> match.
    # If start xor stop near truth's start or stop -> print as of interest.
    # If nothing -> Discard.

    # Allowed error in seconds for matching starts and ends.
    maximum_error = ALLOWED_INTERVAL_MATCHING_ERROR_DEFAULT

    matched = [list() for _ in range(len(truth_intervals))]
    of_interest = []
    misses = []

    # Classify intervals and calculate errors for each matched interval.
    for guessed_interval_i, (start, phase0_stop, phase1_stop, stop) in enumerate(guessed_intervals):
        best_truth_match_i = None
        best_error = None
        is_of_interest = False
        for truth_i, (truth_start, _, _, truth_stop) in enumerate(truth_intervals):
            distance_start = abs(time_stamps[truth_start] - time_stamps[start])
            distance_stop = abs(time_stamps[truth_stop] - time_stamps[stop])
            near_start = distance_start <= maximum_error
            near_end = distance_stop <= maximum_error
            total_error = distance_start + distance_stop
            if near_start and near_end and (best_error is None or total_error < best_error):
                best_truth_match_i = truth_i
                best_error = total_error
            elif near_start or near_end:
                is_of_interest = True

        if best_truth_match_i is not None:
            (truth_start, truth_phase0_stop, truth_phase1_stop, truth_stop) = truth_intervals[best_truth_match_i]
            error_length = abs((time_stamps[truth_stop] - time_stamps[truth_start])
                               - (time_stamps[stop] - time_stamps[start]))
            error_phase0 = abs((time_stamps[truth_phase0_stop] - time_stamps[truth_start])
                               - (time_stamps[phase0_stop] - time_stamps[start]))
            error_phase1 = abs((time_stamps[truth_phase1_stop] - time_stamps[truth_phase0_stop])
                               - (time_stamps[phase1_stop] - time_stamps[phase0_stop]))
            error_phase2 = abs((time_stamps[truth_stop] - time_stamps[truth_phase1_stop])
                               - (time_stamps[stop] - time_stamps[phase1_stop]))
            error_length_percent = error_length / (time_stamps[stop] - time_stamps[start])
            error_phase0_percent = error_phase0 / (time_stamps[phase0_stop] - time_stamps[start])
            try:
                error_phase1_percent = error_phase1 / (time_stamps[phase1_stop] - time_stamps[phase0_stop])
            except ZeroDivisionError:
                error_phase1_percent = 0
            error_phase2_percent = error_phase2 / (time_stamps[stop] - time_stamps[phase1_stop])
            matched[best_truth_match_i].append((
                guessed_interval_i,
                ((error_length, error_phase0, error_phase1, error_phase2), (error_length_percent, error_phase0_percent, error_phase1_percent, error_phase2_percent))
            ))
        elif is_of_interest:
            of_interest.append(guessed_interval_i)
        else:
            misses.append(guessed_interval_i)

    # Sum up worst error of each matched list.
    # error is the difference in interval length. -> Errors due to shifts in annotated truth intervals (mis alignment of mp4 and data)
    total_error_interval_length = 0
    total_error_phase0 = 0
    total_error_phase1 = 0
    total_error_phase2 = 0
    total_error_interval_length_percent = 0
    total_error_phase0_percent = 0
    total_error_phase1_percent = 0
    total_error_phase2_percent = 0
    all_error_interval_length = []
    all_error_phase0 = []
    all_error_phase1 = []
    all_error_phase2 = []
    all_error_interval_length_percent = []
    all_error_phase0_percent = []
    all_error_phase1_percent = []
    all_error_phase2_percent = []
    for matched_intervals in matched:
        worst_error_length = -1
        worst_i = -1
        for i, (_, ((error_length, _, _, _), _)) in enumerate(matched_intervals):
            if error_length > worst_error_length:
                worst_error_length = error_length
                worst_i = i
        if worst_error_length == -1:
            continue
        total_error_interval_length += worst_error_length
        total_error_phase0 += matched_intervals[worst_i][1][0][1]
        total_error_phase1 += matched_intervals[worst_i][1][0][2]
        total_error_phase2 += matched_intervals[worst_i][1][0][3]
        # Maybe do worst case for percent separately.
        total_error_interval_length_percent += matched_intervals[worst_i][1][1][0]
        total_error_phase0_percent += matched_intervals[worst_i][1][1][1]
        total_error_phase1_percent += matched_intervals[worst_i][1][1][2]
        total_error_phase2_percent += matched_intervals[worst_i][1][1][3]

        all_error_interval_length.append(worst_error_length)
        all_error_phase0.append(matched_intervals[worst_i][1][0][1])
        all_error_phase1.append(matched_intervals[worst_i][1][0][2])
        all_error_phase2.append(matched_intervals[worst_i][1][0][3])
        all_error_interval_length_percent.append(matched_intervals[worst_i][1][1][0])
        all_error_phase0_percent.append(matched_intervals[worst_i][1][1][1])
        all_error_phase1_percent.append(matched_intervals[worst_i][1][1][2])
        all_error_phase2_percent.append(matched_intervals[worst_i][1][1][3])

    # Print results.
    # Match statistics:
    truths_matched = sum(map(lambda matches: len(matches) > 0, matched))
    avg_error_length = avg_error_phase0 = avg_error_phase1 = avg_error_phase2 = 0
    avg_error_length_percent = avg_error_phase0_percent = avg_error_phase1_percent = avg_error_phase2_percent = 0
    if truths_matched > 0:
        avg_error_length = round(total_error_interval_length/truths_matched, 3)
        avg_error_phase0 = round(total_error_phase0/truths_matched, 3)
        avg_error_phase1 = round(total_error_phase1/truths_matched, 3)
        avg_error_phase2 = round(total_error_phase2/truths_matched, 3)
        avg_error_length_percent = round(total_error_interval_length_percent/truths_matched, 3)
        avg_error_phase0_percent = round(total_error_phase0_percent/truths_matched, 3)
        avg_error_phase1_percent = round(total_error_phase1_percent/truths_matched, 3)
        avg_error_phase2_percent = round(total_error_phase2_percent/truths_matched, 3)
    false_and_duplicate_guesses = len(guessed_intervals) - truths_matched

    if verbose:
        print(
            f"Intervals Matched: {truths_matched}/{len(truth_intervals)}; False: {false_and_duplicate_guesses}; totalLengthError={round(total_error_interval_length, 3)}; avgLengthError={avg_error_length}; avgPhase0Error={avg_error_phase0}; avgPhase1Error={avg_error_phase1}; avgPhase2Error={avg_error_phase2}\n" +
            f"                   -> totalLengthError%={round(total_error_interval_length_percent, 3)}; avgLengthError%={avg_error_length_percent}; avgPhase0Error%={avg_error_phase0_percent}; avgPhase1Error%={avg_error_phase1_percent}; avgPhase2Error%={avg_error_phase2_percent}"
        )

        for (truth_start, truth_phase0_stop, truth_phase1_stop, truth_stop), matched_intervals in zip(truth_intervals, matched):
            print(
                f"Truth Interval: {round(time_stamps[truth_start],3)}-{round(time_stamps[truth_phase0_stop],3)}-{round(time_stamps[truth_phase1_stop],3)}-{round(time_stamps[truth_stop],3)} ->",
                f"({round(time_stamps[truth_phase0_stop] - time_stamps[truth_start], 3)})",
                f"({round(time_stamps[truth_phase1_stop] - time_stamps[truth_phase0_stop], 3)})",
                f"({round(time_stamps[truth_stop] - time_stamps[truth_phase1_stop], 3)})",
                "| matched:"
            )
            for i, _ in matched_intervals:
                start, phase0_stop, phase1_stop, stop = guessed_intervals[i]
                if start > phase0_stop or phase0_stop > phase1_stop or phase1_stop > stop:
                    print(RED[0], end="")
                print(
                    f"   {round(time_stamps[start], 3)}-{round(time_stamps[phase0_stop], 3)}-{round(time_stamps[phase1_stop], 3)}-{round(time_stamps[stop], 3)} ->",
                    f"({round(time_stamps[phase0_stop] - time_stamps[start], 3)})",
                    f"({round(time_stamps[phase1_stop] - time_stamps[phase0_stop], 3)})",
                    f"({round(time_stamps[stop] - time_stamps[phase1_stop], 3)})"
                )
                if start > phase0_stop or phase0_stop > phase1_stop or phase1_stop > stop:
                    print(RED[1], end="")
            print()

        print("Of Interest (start or stop matched):")
        for guessed_interval_i in of_interest:
            start, phase0_stop, phase1_stop, stop = guessed_intervals[guessed_interval_i]
            if start > phase0_stop or phase0_stop > phase1_stop or phase1_stop > stop:
                print(RED[0], end="")
            print(
                f"   {round(time_stamps[start], 3)}-{round(time_stamps[phase0_stop], 3)}-{round(time_stamps[phase1_stop], 3)}-{round(time_stamps[stop], 3)} ->",
                f"({round(time_stamps[phase0_stop] - time_stamps[start], 3)})",
                f"({round(time_stamps[phase1_stop] - time_stamps[phase0_stop], 3)})",
                f"({round(time_stamps[stop] - time_stamps[phase1_stop], 3)})"
            )
            if start > phase0_stop or phase0_stop > phase1_stop or phase1_stop > stop:
                print(RED[1], end="")

        print(verbose, "Wrong:")
        for guessed_interval_i in misses:
            start, phase0_stop, phase1_stop, stop = guessed_intervals[guessed_interval_i]
            if start > phase0_stop or phase0_stop > phase1_stop or phase1_stop > stop:
                print(RED[0], end="")
            print(
                f"   {round(time_stamps[start], 3)}-{round(time_stamps[phase0_stop], 3)}-{round(time_stamps[phase1_stop], 3)}-{round(time_stamps[stop], 3)} ->",
                f"({round(time_stamps[phase0_stop] - time_stamps[start], 3)})",
                f"({round(time_stamps[phase1_stop] - time_stamps[phase0_stop], 3)})",
                f"({round(time_stamps[stop] - time_stamps[phase1_stop], 3)})"
            )
            if start > phase0_stop or phase0_stop > phase1_stop or phase1_stop > stop:
                print(RED[1], end="")

    return (truths_matched, false_and_duplicate_guesses,
            total_error_interval_length,
            total_error_phase0,
            total_error_phase1,
            total_error_phase2,
            total_error_interval_length_percent,
            total_error_phase0_percent,
            total_error_phase1_percent,
            total_error_phase2_percent,
            all_error_interval_length,
            all_error_phase0,
            all_error_phase1,
            all_error_phase2,
            all_error_interval_length_percent,
            all_error_phase0_percent,
            all_error_phase1_percent,
            all_error_phase2_percent)


def compare_guesses_to_truth_without_phases(time_stamps: list, truth_intervals: List[Tuple[int, int]], guessed_intervals: list, verbose: bool = True) -> Tuple[int, int, float, float, list, list]:
    """Assumes that the interval lists are sorted in acceding order of their start times. TODO: Check if this is still true."""

    # Try to match the guessed intervals to truth.
    # If start and stop are near truth's start and stop -> match.
    # If start xor stop near truth's start or stop -> print as of interest.
    # If nothing -> Discard.

    # Allowed error in seconds for matching starts and ends.
    maximum_error = ALLOWED_INTERVAL_MATCHING_ERROR_DEFAULT

    matched = [list() for _ in range(len(truth_intervals))]
    of_interest = []
    misses = []

    # Classify intervals and calculate errors for each matched interval.
    for guessed_interval_i, (start, stop) in enumerate(guessed_intervals):
        best_truth_match_i = None
        best_error = None
        is_of_interest = False
        for truth_i, (truth_start, truth_stop) in enumerate(truth_intervals):
            distance_start = abs(time_stamps[truth_start] - time_stamps[start])
            distance_stop = abs(time_stamps[truth_stop] - time_stamps[stop])
            near_start = distance_start <= maximum_error
            near_end = distance_stop <= maximum_error
            total_error = distance_start + distance_stop
            if near_start and near_end and (best_error is None or total_error < best_error):
                best_truth_match_i = truth_i
                best_error = total_error
            elif near_start or near_end:
                is_of_interest = True

        if best_truth_match_i is not None:
            (truth_start, truth_stop) = truth_intervals[best_truth_match_i]
            error_length = abs((time_stamps[truth_stop] - time_stamps[truth_start])
                               - (time_stamps[stop] - time_stamps[start]))
            error_length_percent = error_length / (time_stamps[stop] - time_stamps[start])
            matched[best_truth_match_i].append((
                guessed_interval_i,
                error_length,
                error_length_percent
            ))
        elif is_of_interest:
            of_interest.append(guessed_interval_i)
        else:
            misses.append(guessed_interval_i)

    # Sum up worst error of each matched list.
    # error is the difference in interval length. -> Errors due to shifts in annotated truth intervals (mis alignment of mp4 and data)
    # total_error_interval_length = sum([max([error for _, error in matched_intervals], default=0) for matched_intervals in matched])
    total_error_interval_length = 0
    total_error_interval_length_percent = 0
    all_error_interval_length = []
    all_error_interval_length_percent = []
    for matched_intervals in matched:
        worst_i = None
        worst_length_error = -1
        for i, (_, length_error, _) in enumerate(matched_intervals):
            if length_error > worst_length_error:
                worst_length_error = length_error
                worst_i = i
        if worst_i is None:
            continue
        total_error_interval_length += worst_length_error
        total_error_interval_length_percent += matched_intervals[worst_i][2]

        all_error_interval_length.append(worst_length_error)
        all_error_interval_length_percent.append(matched_intervals[worst_i][2])

    # Print results.
    # Match statistics:
    truths_matched = sum(map(lambda matches: len(matches) > 0, matched))
    avg_error_length = 0
    avg_error_length_percent = 0
    if truths_matched > 0:
        avg_error_length = round(total_error_interval_length/truths_matched, 3)
        avg_error_length_percent = round(total_error_interval_length_percent/truths_matched, 3)
    false_and_duplicate_guesses = len(guessed_intervals) - truths_matched

    if verbose:
        print(
            f"Intervals Matched: {truths_matched}/{len(truth_intervals)}; False: {false_and_duplicate_guesses}; totalLengthError={round(total_error_interval_length, 3)}; avgLengthError={avg_error_length}; totalLengthError%={round(total_error_interval_length_percent, 3)}; avgLengthError={avg_error_length_percent}"
        )

        for (truth_start, truth_stop), matched_intervals in zip(truth_intervals, matched):
            print(
                f"Truth Interval: {round(time_stamps[truth_start],3)}-{round(time_stamps[truth_stop],3)}",
                "| matched:"
            )
            for i, *_ in matched_intervals:
                start, stop = guessed_intervals[i]
                print(
                    f"   {round(time_stamps[start], 3)}-{round(time_stamps[stop], 3)}"
                )
            print()

        print("Of Interest (start or stop matched):")
        for guessed_interval_i in of_interest:
            start, stop = guessed_intervals[guessed_interval_i]
            print(
                f"   {round(time_stamps[start], 3)}-{round(time_stamps[stop], 3)}"
            )

        print("Wrong:")
        for guessed_interval_i in misses:
            start, stop = guessed_intervals[guessed_interval_i]
            print(
                f"   {round(time_stamps[start], 3)}-{round(time_stamps[stop], 3)}"
            )

    return (truths_matched, false_and_duplicate_guesses, total_error_interval_length, total_error_interval_length_percent, all_error_interval_length, all_error_interval_length_percent)


def compare_guesses_to_truth(time_stamps: list, truth_intervals: list, guessed_intervals: list):
    """Deprecated
    Assumes that the interval lists are sorted in acceding order of their start times."""

    # Try to match the guessed intervals to truth.
    # If start and stop are near truth's start and stop -> match.
    # If start xor stop near truth's start or stop -> print as of interest.
    # If nothing -> Discard.

    # Allowed error in seconds for matching starts and ends.
    maximum_error = 0.5

    matched = [list() for _ in range(len(truth_intervals))]
    of_interest = []
    misses = []

    # Classify intervals.
    for start, stop in guessed_intervals:
        is_miss = False
        for truth_i, (truth_start, truth_stop) in enumerate(truth_intervals):
            near_start = (time_stamps[truth_start] - maximum_error
                          <= time_stamps[start]
                          <= time_stamps[truth_start] + maximum_error)
            near_end = (time_stamps[truth_stop] - maximum_error
                        <= time_stamps[stop]
                        <= time_stamps[truth_stop] + maximum_error)
            if near_start and near_end:
                matched[truth_i].append((start, stop))
                break
            if near_start or near_end:
                of_interest.append((start, stop))
                break
        else:
            is_miss = True

        if is_miss:
            misses.append((start, stop))

    # Get worst error of each matched list.
    error = 0
    for (truth_start, truth_stop), matched_intervals in zip(truth_intervals, matched):
        worst_error = 0
        for start, stop in matched_intervals:
            current_error = abs(time_stamps[start] - time_stamps[truth_start])
            current_error += abs(time_stamps[stop] - time_stamps[truth_stop])
            worst_error = max(current_error, worst_error)
        error += worst_error

    # Print results.
    # Match statistics:
    truths_matched = sum(map(lambda matches: len(matches) > 0, matched))
    avg_error = round(error/truths_matched, 3) if truths_matched > 0 else 0
    print(
        f"Intervals Matched: {truths_matched}/{len(truth_intervals)}; False: {len(guessed_intervals)-truths_matched}; totalError={round(error, 3)}; avgError={avg_error}"
    )

    for (truth_start, truth_stop), matched_intervals in zip(truth_intervals, matched):
        print(f"Truth Interval: {time_stamps[truth_start]} - {time_stamps[truth_stop]} | matched:")
        for start, stop in matched_intervals:
            print(f"   {round(time_stamps[start], 3)}-{round(time_stamps[stop], 3)}")

    print("Of Interest (start or stop matched):")
    for start, stop in of_interest:
        print(f"   {round(time_stamps[start], 3)}-{round(time_stamps[stop], 3)}")

    print("Wrong:")
    for start, stop in misses:
        print(f"   {round(time_stamps[start], 3)}-{round(time_stamps[stop], 3)}")


def rate_analysis_method(folder_name: str, analyze_method: callable):
    """Deprecated"""
    data = get_data_from_zip(folder_name)
    annotations = get_annotation_from_folder(folder_name)
    if data is None or annotations is None:
        print("Error: Either the annotations are missing or non recording Folder")
        return

    frames = data["frames"]
    time_stamps = get_time_stamps(frames)
    interval_parts = annotations["intervals"]

    # Get the start and stop point of the actual interval where the exercise is executed.
    global_start, global_stop = get_motion_data_interval(interval_parts, time_stamps)

    # Slice all lists to only account for the actual motion data of the exercise.
    time_stamps = time_stamps[global_start:global_stop+1]
    # acceleration_vectors = acceleration_vectors[global_start:global_stop+1]

    truth_intervals = get_intervals_from_parts(interval_parts, time_stamps)

    acceleration_vectors = get_acceleration(frames)
    reduced_data = PCA(acceleration_vectors, 3)
    reduced_data_multi_smooth = multi_smooth(reduced_data)
    reduced_data_multi_smooth_pc0 = reduced_data_multi_smooth[:, 0]
    reduced_data_multi_smooth_pc0 = reduced_data_multi_smooth_pc0[global_start:global_stop+1]

    guessed_intervals = analyze_method(reduced_data_multi_smooth_pc0, time_stamps)

    compare_guesses_to_truth(time_stamps, truth_intervals, guessed_intervals)


def rate_rot_rate_analysis(folder_name: str, analyze_method: callable, with_phases: bool) -> None:
    """Deprecated"""
    data = get_data_from_zip(folder_name)
    annotations = get_annotation_from_folder(folder_name)
    if data is None or annotations is None:
        print("Error: Either the annotations are missing or non recording Folder")
        return

    frames = data["frames"]
    rotation_rate = get_rotation_rate(frames)
    if rotation_rate is None:
        print("No rotation rate data available")
        return
    time_stamps = get_time_stamps(frames)
    interval_parts = annotations["intervals"]

    global_start, global_stop = get_motion_data_interval(interval_parts, time_stamps)

    time_stamps = time_stamps[global_start:global_stop+1]
    rotation_rate = rotation_rate[global_start:global_stop+1]

    rotation_rate_low_pass = low_pass_filter(rotation_rate, 2.5, 100)

    if with_phases:
        truth_intervals = get_interval_phases_from_parts(interval_parts, time_stamps)
    else:
        truth_intervals = get_intervals_from_parts(interval_parts, time_stamps)

    for axis_name, axis_i in zip(["x", "y", "z"], range(rotation_rate_low_pass.shape[1])):
        print(f"**** {axis_name} axis ****")
        axis_data = rotation_rate_low_pass[:, axis_i]
        guessed_intervals = analyze_method(axis_data, time_stamps)
        if with_phases:
            compare_guesses_to_truth_with_phases(time_stamps, truth_intervals, guessed_intervals)
        else:
            compare_guesses_to_truth(time_stamps, truth_intervals, guessed_intervals)
        print()


def rate_gravity_analysis(folder_name: str, analyze_method: callable) -> List[Tuple[int, int, float, float, float, float]]:
    """Deprecated"""
    data = get_data_from_zip(folder_name)
    annotations = get_annotation_from_folder(folder_name)
    if data is None or annotations is None:
        print("Error: Either the annotations are missing or non recording Folder")
        return

    frames = data["frames"]
    gravity = get_gravity(frames)
    time_stamps = get_time_stamps(frames)
    interval_parts = annotations["intervals"]

    global_start, global_stop = get_motion_data_interval(interval_parts, time_stamps)

    time_stamps = time_stamps[global_start:global_stop+1]
    gravity = gravity[global_start:global_stop+1]

    gravity_low_pass = low_pass_filter(gravity, 2.5, 100)

    truth_intervals = get_interval_phases_from_parts(interval_parts, time_stamps)

    stats = []
    for axis_name, axis_i in zip(["x", "y", "z"], range(gravity_low_pass.shape[1])):
        print(f"**** {axis_name} axis ****")
        axis_data = gravity_low_pass[:, axis_i]
        guessed_intervals = analyze_method(axis_data, time_stamps)
        stats.append(compare_guesses_to_truth_with_phases(
            time_stamps, truth_intervals, guessed_intervals))
        print()

    return stats


def rate_analysis(analysis_method: callable, data_type: SensorDataType, with_phases: bool, folder_name: str = None, only_summary: bool = False, verbose: bool = True) -> dict:
    """
    This method can run all versions of all analysis methods. It lets them calculate the repetitions
    and summarizes their performance.
    """

    if not verbose:
        only_summary = True

    if folder_name is not None:
        files = [folder_name]
    else:
        files = os.listdir(RECORDINGS_PATH)

    all_stats = []
    for file_name in files:
        file = os.path.join(RECORDINGS_PATH, file_name, f"{file_name}.zip")
        if not os.path.isfile(file):
            continue

        print_conditional(not only_summary, f"------------- {file_name} -----------------------------")

        raw_data = get_data_from_zip(file_name)  # TODO: Change to JSONOps variant
        annotations = get_annotation_from_folder(file_name)
        if raw_data is None or annotations is None:
            print_conditional(not only_summary, "Error: Either the annotations are missing or non recording Folder")
            continue

        frames = raw_data["frames"]
        time_stamps = get_time_stamps(frames)
        interval_parts = annotations["intervals"]
        global_start, global_stop = get_motion_data_interval(interval_parts, time_stamps)
        time_stamps = time_stamps[global_start:global_stop+1]

        if data_type == SensorDataType.linearAcceleration:
            data = get_acceleration(frames)
            # data = np.sqrt(np.abs(linAccData))
            # data[linAccData < 0] = data[linAccData < 0] * (-1)
            # pca = PCA_(data[global_start:global_stop+1])
            # data = pca.fit(data, 3)
            # data = smooth_points(data, 40)
            # data = median_filter(data, 31)
            # data = IRR_low_pass_filter(data, 2, 100, 2, -12)
            # data = IIR_low_pass_filter_ord2_single(data, 1.2, 100)
            # data = IIR_low_pass_filter_ord3_single(data, 1.2, 100)
            # data = low_pass_filter(data, 0.75, 100)
            data = low_pass_filter(data, 1.2, 100)
        elif data_type == SensorDataType.gravity:
            gravity = get_gravity(frames)
            gravity_low_pass = low_pass_filter(gravity, 2.5, 100)
            # gravity_low_pass = IRR_low_pass_filter(gravity, 2.5, FS, 2)
            # gravity_low_pass = IIR_low_pass_filter_ord2_single(gravity, 2.5, 100)
            # gravity_low_pass = IIR_low_pass_filter_ord3_single(gravity, 2.5, 100)
            data = gravity_low_pass
        elif data_type == SensorDataType.rotationRate:
            rotation_rate = get_rotation_rate(frames)
            if rotation_rate is None:
                print_conditional(not only_summary, "No rotation rate data available")
                continue
            rotation_rate = low_pass_filter(rotation_rate, 2.5, 100)
            # rotation_rate = IRR_low_pass_filter(rotation_rate, 2.5, FS, 2)
            # rotation_rate = IIR_low_pass_filter_ord3_single(rotation_rate, 2.5, 100)
            data = rotation_rate

        data = data[global_start:global_stop+1]
        if with_phases:
            truth_intervals = get_interval_phases_from_parts(interval_parts, time_stamps)
        else:
            truth_intervals = get_intervals_from_parts(interval_parts, time_stamps)
        current_stats = []
        for axis_name, axis_i in zip(["x", "y", "z"], range(data.shape[1])):
            print_conditional(not only_summary, f"**** {axis_name} axis ****")
            axis_data = data[:, axis_i]
            guessed_intervals = analysis_method(axis_data, time_stamps, not only_summary)
            if with_phases:
                current_stats.append(compare_guesses_to_truth_with_phases(
                    time_stamps, truth_intervals, guessed_intervals, verbose=(not only_summary)))
            else:
                current_stats.append(compare_guesses_to_truth_without_phases(
                    time_stamps, truth_intervals, guessed_intervals, verbose=(not only_summary)))
            print_conditional(not only_summary)

        # Auto calc 'best' axis.
        best_axis = auto_choose_axis(data if data_type != SensorDataType.gravity else derivative(data, time_stamps))
        all_stats.append([len(truth_intervals), current_stats, best_axis])

    if len(all_stats) == 0:
        print("No valid files have been given")
        return

    # Remove / Select the axis's which should be evaluated for each exercise / file.
    # Methods -> Best (best=leastTotalError), average of best (best=highestDetectedTruthCount), average (non are removed)
    all_bests = []
    only_best = []
    only_worst_of_bests = []
    auto_best = []
    total_truths = 0
    total_truths_all_best = 0
    for n_truths, stats, best_axis in all_stats:
        total_truths += n_truths
        max_detection_score = max(stat[0]-stat[1] for stat in stats)
        bests = [stat for stat in stats if stat[0]-stat[1] == max_detection_score]
        all_bests.extend(bests)
        total_truths_all_best += len(bests) * n_truths

        # First priority least false positives, second lowest sum of errors.
        sorted_bests = sorted([(i, false, sum(errors[:(4 if with_phases else 2)])) for i, (_, false, *errors) in enumerate(bests)], key=itemgetter(1, 2))
        best = bests[sorted_bests[0][0]]
        worst = bests[sorted_bests[-1][0]]

        only_best.append(best)
        only_worst_of_bests.append(worst)

        auto_best.append(stats[best_axis.value])

    def calc_sums_std_and_medians(l) -> list:
        split = 10 if with_phases else 4
        out = [sum(axis) for axis in zip(*(map(lambda x: x[:split], l)))]
        medians = []
        for i, vals in enumerate([chain(*vals) for vals in zip(*(map(lambda x: x[split:], l)))]):
            vals = sorted(list(vals))
            if out[0] > 0:
                mean = out[2+i] / out[0]
                variance = sum((val-mean)**2 for val in vals) / out[0]
                if len(vals) % 2 == 0:
                    median = (vals[round((len(vals)/2))-1] + vals[round(len(vals)/2)]) / 2
                else:
                    median = vals[int(len(vals)/2)]
            else:
                variance = 0
                median = 0
            out.append(sqrt(variance))
            medians.append(median)
        out.extend(medians)
        return out

    # print(list([chain(*vals) for vals in zip(*(map(lambda x: x[10:], only_best)))][0]))
    only_best_sums = calc_sums_std_and_medians(only_best)
    only_worst_sums = calc_sums_std_and_medians(only_worst_of_bests)
    all_bests_sum = calc_sums_std_and_medians(all_bests)
    all_axis = []
    for _, states, _ in all_stats:
        all_axis.extend(states)
    all_all_sums = calc_sums_std_and_medians(all_axis)
    auto_best_sums = calc_sums_std_and_medians(auto_best)

    if verbose:
        # Print Results.
        # Only Best:
        print()
        print("-----------------------------")
        print("--         Summary         --")
        print("-----------------------------")
        print()
        print(f"Sensor Data Type:            {data_type.name}")
        print(f"Number of exercises (files): {len(all_stats)}")
        print()

        print("Only best of each recording:")
        print_results(only_best_sums, total_truths, len(all_stats), with_phases)

        print("Only worst of bests of each recording:")
        print_results(only_worst_sums, total_truths, len(all_stats), with_phases)

        print("All bests of each recording (highest number of matched intervals):")
        print_results(all_bests_sum, total_truths_all_best, len(all_stats), with_phases)

        print("Automatically chosen 'best' axis")
        print_results(auto_best_sums, total_truths, len(all_stats), with_phases)

        print("All axis:")
        print_results(all_all_sums, total_truths*3, len(all_stats), with_phases)

    out = {
        "only_best": only_best_sums,
        "only_worst": only_worst_sums,
        "all_bests": all_bests_sum,
        "all_axis": all_all_sums,
        "total_truths": total_truths,
        "total_truths_all_bests": total_truths_all_best,
        "num_files": len(all_stats)
    }

    return out


if __name__ == "__main__":
    # gdv.DEBUG_ENABLED = True
    # gdv.gdv_zero_plato_slope_radius = 0.79
    # gdv.gdv_zero_plato_val_radius = 0.15
    # gdv.gdv_zero_plato_min_length = 0.0
    # gdv.gdv_zero_plato_allowed_skip_length = 0.1
    folder_name = input("Folder Name (blank for all): ")
    if len(folder_name) == 0:
        folder_name = None
    rate_analysis(analyze_rot_rate_v2_3, SensorDataType.rotationRate, with_phases=True, folder_name=folder_name, only_summary=True)
