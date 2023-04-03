"""For writing the best axis for each data type into a separate file for further analysis
or for using them in the transformations."""

import os

from analyze_truth import *

files = os.listdir(RECORDINGS_PATH)
files.sort(key=lambda x: x.lower())

# for file in files:
#     curr_file = os.path.join(RECORDINGS_PATH, file, BEST_AXIS_FILE_NAME)
#     if not os.path.isfile(curr_file):
#         continue
#     with open(curr_file) as f:
#         inp = f.read()
#     out = inp.replace("acceleration", SensorDataType.linearAcceleration.name)
#     with open(curr_file, "w") as f:
#         f.write(out)

# for file in files:
#     zip_file = os.path.join(RECORDINGS_PATH, file, f"{file}.zip")
#     if not os.path.isfile(zip_file):
#         continue
#     if os.path.isfile(os.path.join(RECORDINGS_PATH, file, BEST_AXIS_FILE_NAME)):
#         continue
#     print("-------", file, "-------")
#     vals = []
#     for prompt in ["Acceleration: ", "Gravity: ", "Rotation Rate: "]:
#         while True:
#             val = input(prompt)
#             if ((len(val) == 2 and val[0] == "xyz-" and val[1] == "-")
#                     or (len(val) == 1 and val in "xyz")):
#                 break
#         vals.append(val)
#     out_file = os.path.join(RECORDINGS_PATH, file, BEST_AXIS_FILE_NAME)
#     out_text = f"{SensorDataType.linearAcceleration.name}={vals[0]}\n{SensorDataType.gravity.name}={vals[1]}\n{SensorDataType.rotationRate.name}={vals[2]}"
#     with open(out_file, "w") as f:
#         f.write(out_text)


def auto_write_best_axis_using_analysis(lin_acc_analysis: callable, grav_analysis: callable, rot_rate_analysis: callable, with_phases: bool, folder_name: str = None) -> None:
    """Only methods, which calculated the phases of the motions.

    TODO: Work over sorting of the bests -> absolute lengths and percentages.
    """

    if folder_name is not None:
        files = [folder_name]
    else:
        files = os.listdir(RECORDINGS_PATH)

    all_stats = []
    for file_name in files:
        file = os.path.join(RECORDINGS_PATH, file_name, f"{file_name}.zip")
        if not os.path.isfile(file):
            continue

        print(f"------------- {file_name} -----------------------------")

        raw_data = get_data_from_zip(file_name)  # TODO: Change to JSONOps variant
        annotations = get_annotation_from_folder(file_name)
        if raw_data is None or annotations is None:
            print("Error: Either the annotations are missing or non recording Folder")
            continue

        frames = raw_data["frames"]
        time_stamps = get_time_stamps(frames)
        interval_parts = annotations["intervals"]
        global_start, global_stop = get_motion_data_interval(interval_parts, time_stamps)
        time_stamps = time_stamps[global_start:global_stop+1]

        out_parts = []
        for analysis_method, data_type in zip(
            [lin_acc_analysis, grav_analysis, rot_rate_analysis],
                [SensorDataType.linearAcceleration, SensorDataType.gravity, SensorDataType.rotationRate]):

            if data_type == SensorDataType.linearAcceleration:
                data = get_acceleration(frames)
                data = low_pass_filter(data, 2.5, 100)
            elif data_type == SensorDataType.gravity:
                gravity = get_gravity(frames)
                gravity_low_pass = low_pass_filter(gravity, 2.5, 100)
                data = gravity_low_pass
            elif data_type == SensorDataType.rotationRate:
                rotation_rate = get_rotation_rate(frames)
                if rotation_rate is None:
                    out_parts.append(f"{data_type.name}=--")
                    continue
                rotation_rate = low_pass_filter(rotation_rate, 2.5, 100)
                data = rotation_rate

            data = data[global_start:global_stop+1]
            if with_phases:
                truth_intervals = get_interval_phases_from_parts(interval_parts, time_stamps)
            else:
                truth_intervals = get_intervals_from_parts(interval_parts, time_stamps)
            stats = []
            for axis_i in range(data.shape[1]):
                axis_data = data[:, axis_i]
                guessed_intervals = analysis_method(axis_data, time_stamps, False)
                if with_phases:
                    stats.append(compare_guesses_to_truth_with_phases(
                        time_stamps, truth_intervals, guessed_intervals, False))
                else:
                    stats.append(compare_guesses_to_truth_without_phases(
                        time_stamps, truth_intervals, guessed_intervals, False))

            #####
            max_detected_truths = max(stat[0] for stat in stats)
            bests = [(i, stat) for (i, stat) in enumerate(stats) if stat[0] == max_detected_truths]
            sorted_bests = sorted([(i, false, sum(errors[:4])) for i, (_, false, *errors) in bests], key=itemgetter(1, 2))
            best = Axis(sorted_bests[0][0])
            if sorted_bests[0][1] == 0:
                val = best.name
            elif sorted_bests[0][1] <= round(len(truth_intervals) / 2, 1):
                val = f"{best.name}-"
            else:
                val = "--"

            out_parts.append(f"{data_type.name}={val}")

        out_text = "\n".join(out_parts)
        print(out_text)
        print()
        with open(os.path.join(RECORDINGS_PATH, file_name, AUTO_BEST_AXIS_FILE_NAME), "w") as f:
            f.write(out_text)


if __name__ == "__main__":
    folder_name = input("Folder Name (Empty=All): ")
    if folder_name == "":
        folder_name = None
    auto_write_best_axis_using_analysis(analyze_v3_1, analyze_gravity_v1_1, analyze_rot_rate_v2_1, True, folder_name)
