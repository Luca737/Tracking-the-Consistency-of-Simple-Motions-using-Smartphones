"""Author: Nicola Vidovic

This method is used for all transformations done to the recorded data.
The transformed data can then be viewed in VIT.

"""

import os
from time import sleep

# import matplotlib.pyplot as plt
import numpy as np
from analyze import *
from FileIO import *
from filters import *
from functions import *
from PCA_LDA import PCA, PCA_


def transform(folder_name: str) -> None:
    if gdv.DEBUG_ENABLED:
        print_yellow("Warning: Debug mode is enabled")
        sleep(2)

    TRANSFORM_RECORDING_ENDING = "_dataTransform"

    ZIP_FILE_PATH = os.path.join(RECORDINGS_PATH, folder_name, f"{folder_name}.zip")

    data, file_path = readRecordingJSONFromZip(ZIP_FILE_PATH)
    OUT_FILE_PATH_NAME = file_path[:file_path.rfind("/")+1] + TRANSFORM_DATA_NAME

    out = RecordingsStructureCreator(
        {
            "applicationName": data["applicationName"] + TRANSFORM_RECORDING_ENDING,
            "videoStartTime": data.get("videoStartTime", None),
            "startTime": data["startTime"]
        }
    )

    print("Transforming Data")

    frames = data["frames"]

    # Get Time stamps ----------------------------------------
    time_stamps = get_time_stamps(frames)
    # --------------------------------------------------------

    annotations = get_annotation_from_folder(folder_name)
    if annotations is not None:
        global_start, global_stop = get_motion_data_interval(annotations["intervals"], time_stamps)

    best_axes = readBestAxis(os.path.join(RECORDINGS_PATH, folder_name, AUTO_BEST_AXIS_FILE_NAME))

    # -------- Gravity --------------------------------------------------------
    grav_vectors = get_gravity(frames)
    grav_vectors_derivative1 = derivative(grav_vectors, time_stamps)
    grav_vectors_derivative2 = derivative(grav_vectors_derivative1, time_stamps)
    grav_vectors_low_pass = low_pass_filter(grav_vectors, 2.5, 100)
    grav_vectors_low_pass_derivative1 = derivative(grav_vectors_low_pass, time_stamps)
    grav_vectors_low_pass_derivative2 = derivative(grav_vectors_low_pass_derivative1, time_stamps)
    grav_low_pass_platos_x = calc_platos(grav_vectors_low_pass_derivative1[:, 0])
    grav_low_pass_platos_y = calc_platos(grav_vectors_low_pass_derivative1[:, 1])
    grav_low_pass_platos_z = calc_platos(grav_vectors_low_pass_derivative1[:, 2])
    # reduced_grav = PCA(grav_vectors, 3)
    # reduced_grav_smoothed = smooth_points(reduced_grav, 10)
    # reduced_grav_multi_smoothed = multi_smooth(reduced_grav)
    grav_low_pass_snd_order = low_pass_filter(grav_vectors, 2.5, 100, 2)
    grav_low_pass_snd_order_hand_single = IIR_low_pass_filter_ord2_single(grav_vectors, 2.5, 100)
    grav_low_pass_snd_order_hand_double = IIR_low_pass_filter_ord2_double(grav_vectors, 2.5, 100)
    grav_low_pass_snd_order_hand_double_general = IRR_low_pass_filter(grav_vectors, 2.5, 100, 2)
    grav_low_pass_trd_order_hand_single = IIR_low_pass_filter_ord3_single(grav_vectors, 2.5, 100)
    grav_low_pass_trd_order_hand_double = IIR_low_pass_filter_ord3_double(grav_vectors, 2.5, 100)
    grav_low_pass_trd_order_hand_double_general = IRR_low_pass_filter(grav_vectors, 2.5, 100, 3)
    grav_low_pass_six_order_hand_double_general = IRR_low_pass_filter(grav_vectors, 2.5, 100, 6)

    # -------- linAcc ---------------------------------------------------------
    acceleration_vectors = get_acceleration(frames)
    # smoothed_acc_vecs = smooth_points(acceleration_vectors, 10)
    # multi_smoothed_acc_vecs = multi_smooth(acceleration_vectors)
    acceleration_vectors_low_pass = low_pass_filter(acceleration_vectors, 2.5, 100)
    acc_low_pass_snd_order = low_pass_filter(acceleration_vectors, 1.2, 100)
    acc_low_pass_snd_order_hand_single = IIR_low_pass_filter_ord2_single(acceleration_vectors, 1.5, 100)
    acc_low_pass_snd_order_hand_double = IIR_low_pass_filter_ord2_double(acceleration_vectors, 2.5, 100)
    acc_low_pass_snd_order_hand_double_general = IRR_low_pass_filter(acceleration_vectors, 2.5, 100, 2, -6)
    acc_low_pass_trd_order_hand_single = IIR_low_pass_filter_ord3_single(acceleration_vectors, 1.2, 100)
    acc_low_pass_trd_order_hand_double = IIR_low_pass_filter_ord3_double(acceleration_vectors, 1.5, 100)
    acc_low_pass_trd_order_hand_double_general = IRR_low_pass_filter(acceleration_vectors, 1.2, 100, 3, -25)
    acc_low_pass_six_order_hand_double_general = IRR_low_pass_filter(acceleration_vectors, 1.5, 100, 4, -30)
    acc_ewma = ewma(acceleration_vectors, 2.5, 100)

    if best_axes is not None:
        best_axis_acc = get_best_axis_of_type(best_axes, SensorDataType.linearAcceleration, AxisType.all)
        if best_axis_acc is not None:
            _, acc_low_zero_platos = calc_zero_points_platos(acceleration_vectors_low_pass[:, best_axis_acc.value], derivative(
                acceleration_vectors_low_pass[:, best_axis_acc.value], time_stamps),  0.01, 0.1, 0, 0)

    print("Average AccXYZ")
    print(np.mean(acceleration_vectors, axis=0) * 9.81)

    # Plot X & Z: Z on x axis, X on y axis.
    # XY = smoothed_acc_vecs[:, [0, 1]]
    # fig, ax = plt.subplots()
    # ax.scatter(x=XY[:, 0], y=XY[:, 1], s=0.5)
    # plt.show()
    # exit()

    if annotations is not None:
        pca = PCA_(acceleration_vectors[global_start:global_stop+1])
    else:
        pca = PCA_(acceleration_vectors)
    reduced_data = pca.fit(acceleration_vectors, 3)
    # auto_swap(reduced_data)
    reduced_data_smoothed = smooth_points(reduced_data, 40)
    reduced_data_multi_smooth = multi_smooth(reduced_data)
    # auto_swap(reduced_data_multi_smooth)

    highpass_cutoff, lowpass_cutoff = pass_range = [0.25, 2.5]

    reduced_data_med_filtered = median_filter(reduced_data, 31)
    # reduced_data_freq_filtered = freq_filter(reduced_data, 9, 10/100)
    # reduced_data_med_filtered_pass = freq_filter(reduced_data_med_filtered, 9, 10/100)
    reduced_data_low_pass = low_pass_filter(reduced_data, 2.5, 100)
    reduced_data_med_low_pass = low_pass_filter(reduced_data_med_filtered, 2.5, 100)
    reduced_data_low_pass_smoothed = low_pass_filter(reduced_data_smoothed, 2.5, 100)
    # reduced_data_high_pass = high_pass_filter(reduced_data, highpass_cutoff, 100)
    # reduced_data_band_pass = band_pass_filter(reduced_data, pass_range, 100)

    # reduced_data_ewma = ewma(reduced_data, 0.1)

    # Acceleration Vector Calc and Smooth ---------------------------
    # length_acc_XY = np.sqrt(np.sum(np.square(acceleration_vectors[:, [0, 1]]), axis=1))
    # length_acc_XZ = np.sqrt(np.sum(np.square(acceleration_vectors[:, [0, 2]]), axis=1))
    # length_acc_YZ = np.sqrt(np.sum(np.square(acceleration_vectors[:, [1, 2]]), axis=1))
    # length_acc_XYZ = np.sqrt(np.sum(np.square(acceleration_vectors[:, [0, 1, 2]]), axis=1))

    # smooth_factor_acc = 10
    # length_acc_XY_smoothed = smooth_points(length_acc_XY, smooth_factor_acc)
    # length_acc_XZ_smoothed = smooth_points(length_acc_XZ, smooth_factor_acc)
    # length_acc_YZ_smoothed = smooth_points(length_acc_YZ, smooth_factor_acc)
    # length_acc_XYZ_smoothed = smooth_points(length_acc_XYZ, smooth_factor_acc)
    # ---------------------------------------------------------------

    # ------------- Rotation Rate -------------
    rotation_rate = get_rotation_rate(frames)
    if rotation_rate is not None:
        rotation_rate_smoothed = smooth_points(rotation_rate, 10)
        rotation_rate_multi_smoothed = smooth_points(rotation_rate, 4)
        for _ in range(5):
            rotation_rate_multi_smoothed = smooth_points(rotation_rate_multi_smoothed, 4)
        rotation_rate_low_pass = low_pass_filter(rotation_rate, 2.5, 100)
        rotation_rate_low_pass_snd_ord = low_pass_filter(rotation_rate, 2.5, 100, 2)
        rotation_rate_low_pass_by_hand_snd_ord_single = IIR_low_pass_filter_ord2_single(rotation_rate, 2.5, 100)
        rotation_rate_low_pass_by_hand_snd_ord_double = IIR_low_pass_filter_ord2_double(rotation_rate, 2.5, 100)
        rotation_rate_low_pass_by_hand_snd_ord_double_general = IRR_low_pass_filter(rotation_rate, 2.5, 100, 2)
        rotation_rate_low_pass_by_hand_trd_ord_single = IIR_low_pass_filter_ord3_single(rotation_rate, 2.5, 100)
        rotation_rate_low_pass_by_hand_trd_ord_double = IIR_low_pass_filter_ord3_double(rotation_rate, 2.5, 100)
        rotation_rate_low_pass_by_hand_trd_ord_double_general = IRR_low_pass_filter(rotation_rate, 2.5, 100, 3)
        rotation_rate_low_pass_by_hand_six_ord_double_general = IRR_low_pass_filter(rotation_rate, 2.5, 100, 6)
        rotation_rate_ewma = ewma(rotation_rate, 2.5, 100)

        if annotations is not None:
            pca = PCA_(acceleration_vectors[global_start:global_stop+1])
        else:
            pca = PCA_(acceleration_vectors)

        # reduced_rotation_rate = pca.fit(rotation_rate, 3)

        rotation_rate_low_pass_1_derivative = derivative(rotation_rate_low_pass, time_stamps)
        rotation_rate_low_pass_2_derivative = derivative(rotation_rate_low_pass_1_derivative, time_stamps)

        if best_axes is not None:
            best_axis_rot_rate = get_best_axis_of_type(best_axes, SensorDataType.rotationRate, AxisType.all)
            if best_axis_rot_rate is not None:
                rotation_rate_low_pass_zero_platos_new = calc_zero_points_platos(
                    rotation_rate_low_pass[:, best_axis_rot_rate.value], rotation_rate_low_pass_1_derivative[:, best_axis_rot_rate.value], 0.15, 0.79, 0, 0.1)[1]

        rotation_rate_low_pass_1_derivative = rotation_rate_low_pass_1_derivative / 6
        rotation_rate_low_pass_2_derivative = rotation_rate_low_pass_2_derivative / 40
        if annotations is not None:
            print("RotationRate pos average:\n", pos_avg_np(
                rotation_rate_low_pass[global_start:global_stop]))
            print("RotationRate neg average:\n", neg_avg_np(
                rotation_rate_low_pass[global_start:global_stop]))

    # -----------------------------------------

    # Analysis of data ----------------------------------------------
    # Simple Above/Below Zero
    pos_neg = np.where(reduced_data_multi_smooth >= 0, 0.25, -0.25)

    reduced_data_multi_smooth_pc0 = reduced_data_multi_smooth[:, 0]
    pos_neg_zero, switch_sign_points, zero_platos = calc_key_features(
        reduced_data_multi_smooth_pc0, time_stamps)

    # intervals_from_v1 = analyzeV1(time_stamps, switch_sign_points, zero_platos)
    # print("----------- Analysis V1 -----------")
    # for start, stop in intervals_from_v1:
    #     print("Interval:", time_stamps[start], "-", time_stamps[stop])
    # print("Count:", len(intervals_from_v1))

    # intervals_from_v1_1 = analyzeV1_1(
    #     time_stamps, switch_sign_points, zero_platos, reduced_data_multi_smooth_pc0)
    # print("---------- Analysis V1.1 ----------")
    # for start, stop in intervals_from_v1_1:
    #     print("Interval:", round(time_stamps[start], 3), "-", round(time_stamps[stop], 3),
    #           f"({round(time_stamps[stop]-time_stamps[start], 3)})")
    #     interval_statistics = calc_statistics(
    #         time_stamps, start, stop, switch_sign_points, reduced_data_multi_smooth_pc0)
    #     for statistic in interval_statistics:
    #         print(statistic)
    # print("Count:", len(intervals_from_v1_1))

    # exit()

    # End: Analysis of data -----------------------------------------

    starts = [plato[0] for plato in zero_platos]
    stops = [plato[1] for plato in zero_platos]
    grav_low_pass_platos_x_starts = [p[0] for p in grav_low_pass_platos_x]
    grav_low_pass_platos_x_stops = [p[1] for p in grav_low_pass_platos_x]
    grav_low_pass_platos_y_starts = [p[0] for p in grav_low_pass_platos_y]
    grav_low_pass_platos_y_stops = [p[1] for p in grav_low_pass_platos_y]
    grav_low_pass_platos_z_starts = [p[0] for p in grav_low_pass_platos_z]
    grav_low_pass_platos_z_stops = [p[1] for p in grav_low_pass_platos_z]
    if best_axes is not None:
        if best_axis_acc is not None:
            acc_zero_plato_starts = [plato[0] for plato in acc_low_zero_platos]
            acc_zero_plato_stops = [plato[1] for plato in acc_low_zero_platos]
        if best_axis_rot_rate is not None:
            rot_rate_low_plato_starts_new = [plato[0] for plato in rotation_rate_low_pass_zero_platos_new]
            rot_rate_low_plato_stops_new = [plato[1] for plato in rotation_rate_low_pass_zero_platos_new]
    for i, frame in enumerate(frames):
        if i in starts:
            zeroPlato_value = -0.1
        elif i in stops:
            zeroPlato_value = 0.1
        else:
            zeroPlato_value = 0.23
        grav_plato_x = -0.1 if i in grav_low_pass_platos_x_starts else 0.1 if i in grav_low_pass_platos_x_stops else 0.22
        grav_plato_y = -0.1 if i in grav_low_pass_platos_y_starts else 0.1 if i in grav_low_pass_platos_y_stops else 0.22
        grav_plato_z = -0.1 if i in grav_low_pass_platos_z_starts else 0.1 if i in grav_low_pass_platos_z_stops else 0.22
        frame_attr = {
            # "accelerationVectorLength_XYZ_og": length_acc_XYZ[i],
            # "accelerationVectorLength_XYZ_smooth": length_acc_XYZ_smoothed[i],
            # "accelerationVectorLength_XY_og": length_acc_XY[i],
            # "accelerationVectorLength_XY_smooth": length_acc_XY_smoothed[i],
            # "accelerationVectorLength_XZ_og": length_acc_XZ[i],
            # "accelerationVectorLength_XZ_smooth": length_acc_XZ_smoothed[i],
            # "accelerationVectorLength_YZ_og": length_acc_YZ[i],
            # "accelerationVectorLength_YZ_smooth": length_acc_YZ_smoothed[i],
            "PCAacc_og_pc0": reduced_data[i][0],
            "PCAacc_og_pc1": reduced_data[i][1],
            "PCAacc_smooth_pc0": reduced_data_smoothed[i][0],
            "PCAacc_smooth_pc1": reduced_data_smoothed[i][1],
            "PCAacc_multiSmooth_pc0": reduced_data_multi_smooth[i][0],
            "PCAacc_multiSmooth_pc1": reduced_data_multi_smooth[i][1],
            # "PCAgrav_og_pc0": reduced_grav[i][0],
            # "PCAgrav_og_pc1": reduced_grav[i][1],
            # "PCAgrav_smooth_pc0": reduced_grav_smoothed[i][0],
            # "PCAgrav_smooth_pc1": reduced_grav_smoothed[i][1],
            # "PCAgrav_multiSmooth_pc0": reduced_grav_multi_smoothed[i][0],
            # "PCAgrav_multiSmooth_pc1": reduced_grav_multi_smoothed[i][1],
            "gravity_lowPass_x": grav_vectors_low_pass[i][0],
            "gravity_lowPass_y": grav_vectors_low_pass[i][1],
            "gravity_lowPass_z": grav_vectors_low_pass[i][2],

            "gravity_lowPassSndOrd_x": grav_low_pass_snd_order[i][0],
            "gravity_lowPassSndOrd_y": grav_low_pass_snd_order[i][1],
            "gravity_lowPassSndOrd_z": grav_low_pass_snd_order[i][2],

            "gravity_lowPassSndOrdIRRSingle_x": grav_low_pass_snd_order_hand_single[i][0],
            "gravity_lowPassSndOrdIRRSingle_y": grav_low_pass_snd_order_hand_single[i][1],
            "gravity_lowPassSndOrdIRRSingle_z": grav_low_pass_snd_order_hand_single[i][2],

            "gravity_lowPassSndOrdIRRDouble_x": grav_low_pass_snd_order_hand_double[i][0],
            "gravity_lowPassSndOrdIRRDouble_y": grav_low_pass_snd_order_hand_double[i][1],
            "gravity_lowPassSndOrdIRRDouble_z": grav_low_pass_snd_order_hand_double[i][2],

            "gravity_lowPassSndOrdIRRDoubleG_x": grav_low_pass_snd_order_hand_double_general[i][0],
            "gravity_lowPassSndOrdIRRDoubleG_y": grav_low_pass_snd_order_hand_double_general[i][1],
            "gravity_lowPassSndOrdIRRDoubleG_z": grav_low_pass_snd_order_hand_double_general[i][2],

            "gravity_lowPassTrdOrdIRRSingle_x": grav_low_pass_trd_order_hand_single[i][0],
            "gravity_lowPassTrdOrdIRRSingle_y": grav_low_pass_trd_order_hand_single[i][1],
            "gravity_lowPassTrdOrdIRRSingle_z": grav_low_pass_trd_order_hand_single[i][2],

            "gravity_lowPassTrdOrdIRRDouble_x": grav_low_pass_trd_order_hand_double[i][0],
            "gravity_lowPassTrdOrdIRRDouble_y": grav_low_pass_trd_order_hand_double[i][1],
            "gravity_lowPassTrdOrdIRRDouble_z": grav_low_pass_trd_order_hand_double[i][2],

            "gravity_lowPassTrdOrdIRRDoubleG_x": grav_low_pass_trd_order_hand_double_general[i][0],
            "gravity_lowPassTrdOrdIRRDoubleG_y": grav_low_pass_trd_order_hand_double_general[i][1],
            "gravity_lowPassTrdOrdIRRDoubleG_z": grav_low_pass_trd_order_hand_double_general[i][2],

            "gravity_lowPassSixOrdIRRDoubleG_x": grav_low_pass_six_order_hand_double_general[i][0],
            "gravity_lowPassSixOrdIRRDoubleG_y": grav_low_pass_six_order_hand_double_general[i][1],
            "gravity_lowPassSixOrdIRRDoubleG_z": grav_low_pass_six_order_hand_double_general[i][2],

            "gravity_lowPassPlato_x": grav_plato_x,
            "gravity_lowPassPlato_y": grav_plato_y,
            "gravity_lowPassPlato_z": grav_plato_z,

            "gravity_lowPassDerivative1_x": grav_vectors_low_pass_derivative1[i][0],
            "gravity_lowPassDerivative1_y": grav_vectors_low_pass_derivative1[i][1],
            "gravity_lowPassDerivative1_z": grav_vectors_low_pass_derivative1[i][2],
            "gravity_lowPassDerivative2_x": grav_vectors_low_pass_derivative2[i][0],
            "gravity_lowPassDerivative2_y": grav_vectors_low_pass_derivative2[i][1],
            "gravity_lowPassDerivative2_z": grav_vectors_low_pass_derivative2[i][2],
            "gravity_derivative1_x": grav_vectors_derivative1[i][0],
            "gravity_derivative1_y": grav_vectors_derivative1[i][1],
            "gravity_derivative1_z": grav_vectors_derivative1[i][2],
            "gravity_derivative2_x": grav_vectors_derivative2[i][0],
            "gravity_derivative2_y": grav_vectors_derivative2[i][1],
            "gravity_derivative2_z": grav_vectors_derivative2[i][2],

            # "smooth_accX": smoothed_acc_vecs[i][0],
            # "smooth_accY": smoothed_acc_vecs[i][1],
            # "smooth_accZ": smoothed_acc_vecs[i][2],
            # "multiSmooth_accX": multi_smoothed_acc_vecs[i][0],
            # "multiSmooth_accY": multi_smoothed_acc_vecs[i][1],
            # "multiSmooth_accZ": multi_smoothed_acc_vecs[i][2],
            "Analysis_PosNeg_pc0": pos_neg[i][0],
            "Analysis_PosNeg_pc1": pos_neg[i][1],
            "Analysis_PosNeg_pc2": pos_neg[i][2],
            "Analysis_PosNegZero": pos_neg_zero[i],
            "Analysis_zeroPoints": 0.0 if i in switch_sign_points else 0.24,
            "Analysis_zeroPlatos": zeroPlato_value,
            # "MedianFilter_accPC0": reduced_data_med_filtered[i][0],
            # "MedianFilter_accPC1": reduced_data_med_filtered[i][1],
            # "MedianFilter_accPC2": reduced_data_med_filtered[i][2],
            # "FreqFilter_accPC0": reduced_data_freq_filtered[i][0],
            # "FreqFilter_accPC1": reduced_data_freq_filtered[i][1],
            # "FreqFilter_accPC2": reduced_data_freq_filtered[i][2],
            # "MedianFreqFilter_accPC0": reduced_data_med_filtered_pass[i][0],
            # "MedianFreqFilter_accPC1": reduced_data_med_filtered_pass[i][1],
            # "MedianFreqFilter_accPC2": reduced_data_med_filtered_pass[i][2],
            "PCAacc_MedianLowFilter_pc0": reduced_data_med_low_pass[i][0],
            "PCAacc_MedianLowFilter_pc1": reduced_data_med_low_pass[i][1],
            "PCAacc_MedianLowFilter_pc2": reduced_data_med_low_pass[i][2],
            "PCAacc_LowPass_pc0": reduced_data_low_pass[i][0],
            "PCAacc_LowPass_pc1": reduced_data_low_pass[i][1],
            "PCAacc_LowPass_pc2": reduced_data_low_pass[i][2],
            "PCAacc_LowSmooth_pc0": reduced_data_low_pass_smoothed[i][0],
            "PCAacc_LowSmooth_pc1": reduced_data_low_pass_smoothed[i][1],
            "PCAacc_LowSmooth_pc2": reduced_data_low_pass_smoothed[i][2],
            # "HighPass_accPC0": reduced_data_high_pass[i][0],
            # "HighPass_accPC1": reduced_data_high_pass[i][1],
            # "HighPass_accPC2": reduced_data_high_pass[i][2],
            # "BandPass_accPC0": reduced_data_band_pass[i][0],
            # "BandPass_accPC1": reduced_data_band_pass[i][1],
            # "BandPass_accPC2": reduced_data_band_pass[i][2],
            # "ewma_accPC0": reduced_data_ewma[i][0],
            # "ewma_accPC1": reduced_data_ewma[i][1],
            # "ewma_accPC2": reduced_data_ewma[i][2],
            "linAcc_LowPass_x": acceleration_vectors_low_pass[i][0],
            "linAcc_LowPass_y": acceleration_vectors_low_pass[i][1],
            "linAcc_LowPass_z": acceleration_vectors_low_pass[i][2],
            "linAcc_LowPassSndOrd_x": acc_low_pass_snd_order[i][0],
            "linAcc_LowPassSndOrd_y": acc_low_pass_snd_order[i][1],
            "linAcc_LowPassSndOrd_z": acc_low_pass_snd_order[i][2],

            "linAcc_LowPassSndOrdIRRSingle_x": acc_low_pass_snd_order_hand_single[i][0],
            "linAcc_LowPassSndOrdIRRSingle_y": acc_low_pass_snd_order_hand_single[i][1],
            "linAcc_LowPassSndOrdIRRSingle_z": acc_low_pass_snd_order_hand_single[i][2],

            "linAcc_LowPassSndOrdIRRDouble_x": acc_low_pass_snd_order_hand_double[i][0],
            "linAcc_LowPassSndOrdIRRDouble_y": acc_low_pass_snd_order_hand_double[i][1],
            "linAcc_LowPassSndOrdIRRDouble_z": acc_low_pass_snd_order_hand_double[i][2],

            "linAcc_LowPassSndOrdIRRDoubleG_x": acc_low_pass_snd_order_hand_double_general[i][0],
            "linAcc_LowPassSndOrdIRRDoubleG_y": acc_low_pass_snd_order_hand_double_general[i][1],
            "linAcc_LowPassSndOrdIRRDoubleG_z": acc_low_pass_snd_order_hand_double_general[i][2],

            "linAcc_LowPassTrdOrdIRRSingle_x": acc_low_pass_trd_order_hand_single[i][0],
            "linAcc_LowPassTrdOrdIRRSingle_y": acc_low_pass_trd_order_hand_single[i][1],
            "linAcc_LowPassTrdOrdIRRSingle_z": acc_low_pass_trd_order_hand_single[i][2],

            "linAcc_LowPassTrdOrdIRRDouble_x": acc_low_pass_trd_order_hand_double[i][0],
            "linAcc_LowPassTrdOrdIRRDouble_y": acc_low_pass_trd_order_hand_double[i][1],
            "linAcc_LowPassTrdOrdIRRDouble_z": acc_low_pass_trd_order_hand_double[i][2],

            "linAcc_LowPassTrdOrdIRRDoubleG_x": acc_low_pass_trd_order_hand_double_general[i][0],
            "linAcc_LowPassTrdOrdIRRDoubleG_y": acc_low_pass_trd_order_hand_double_general[i][1],
            "linAcc_LowPassTrdOrdIRRDoubleG_z": acc_low_pass_trd_order_hand_double_general[i][2],

            "linAcc_LowPassSixOrdIRRDoubleG_x": acc_low_pass_six_order_hand_double_general[i][0],
            "linAcc_LowPassSixOrdIRRDoubleG_y": acc_low_pass_six_order_hand_double_general[i][1],
            "linAcc_LowPassSixOrdIRRDoubleG_z": acc_low_pass_six_order_hand_double_general[i][2],

            "linAcc_ewma_x": acc_ewma[i][0],
            "linAcc_ewma_y": acc_ewma[i][1],
            "linAcc_ewma_z": acc_ewma[i][2],
        }
        if best_axes is not None and best_axis_acc is not None:
            if i in acc_zero_plato_starts:
                acc_zero_plato_val = -0.12
            elif i in acc_zero_plato_stops:
                acc_zero_plato_val = 0.12
            else:
                acc_zero_plato_val = 0.22
            frame_attr[f"linAcc_LowPassZeroPlato{best_axis_acc.name.upper()}"] = acc_zero_plato_val
        if rotation_rate is not None:
            frame_attr.update({
                "rotationRate_smooth_X": rotation_rate_smoothed[i][0],
                "rotationRate_smooth_Y": rotation_rate_smoothed[i][1],
                "rotationRate_smooth_Z": rotation_rate_smoothed[i][2],
                "rotationRate_multiSmooth_X": rotation_rate_multi_smoothed[i][0],
                "rotationRate_multiSmooth_Y": rotation_rate_multi_smoothed[i][1],
                "rotationRate_multiSmooth_Z": rotation_rate_multi_smoothed[i][2],
                "rotationRate_lowPass_X": rotation_rate_low_pass[i][0],
                "rotationRate_lowPass_Y": rotation_rate_low_pass[i][1],
                "rotationRate_lowPass_Z": rotation_rate_low_pass[i][2],
                "rotationRate_lowPassSndOrd_X": rotation_rate_low_pass_snd_ord[i][0],
                "rotationRate_lowPassSndOrd_Y": rotation_rate_low_pass_snd_ord[i][1],
                "rotationRate_lowPassSndOrd_Z": rotation_rate_low_pass_snd_ord[i][2],

                "rotationRate_lowPassSndOrdIRRSingle_X": rotation_rate_low_pass_by_hand_snd_ord_single[i][0],
                "rotationRate_lowPassSndOrdIRRSingle_Y": rotation_rate_low_pass_by_hand_snd_ord_single[i][1],
                "rotationRate_lowPassSndOrdIRRSingle_Z": rotation_rate_low_pass_by_hand_snd_ord_single[i][2],

                "rotationRate_lowPassSndOrdIRRDouble_X": rotation_rate_low_pass_by_hand_snd_ord_double[i][0],
                "rotationRate_lowPassSndOrdIRRDouble_Y": rotation_rate_low_pass_by_hand_snd_ord_double[i][1],
                "rotationRate_lowPassSndOrdIRRDouble_Z": rotation_rate_low_pass_by_hand_snd_ord_double[i][2],

                "rotationRate_lowPassSndOrdIRRDoubleG_X": rotation_rate_low_pass_by_hand_snd_ord_double_general[i][0],
                "rotationRate_lowPassSndOrdIRRDoubleG_Y": rotation_rate_low_pass_by_hand_snd_ord_double_general[i][1],
                "rotationRate_lowPassSndOrdIRRDoubleG_Z": rotation_rate_low_pass_by_hand_snd_ord_double_general[i][2],

                "rotationRate_lowPassTrdOrdIRRSingle_X": rotation_rate_low_pass_by_hand_trd_ord_single[i][0],
                "rotationRate_lowPassTrdOrdIRRSingle_Y": rotation_rate_low_pass_by_hand_trd_ord_single[i][1],
                "rotationRate_lowPassTrdOrdIRRSingle_Z": rotation_rate_low_pass_by_hand_trd_ord_single[i][2],

                "rotationRate_lowPassTrdOrdIRRDouble_X": rotation_rate_low_pass_by_hand_trd_ord_double[i][0],
                "rotationRate_lowPassTrdOrdIRRDouble_Y": rotation_rate_low_pass_by_hand_trd_ord_double[i][1],
                "rotationRate_lowPassTrdOrdIRRDouble_Z": rotation_rate_low_pass_by_hand_trd_ord_double[i][2],

                "rotationRate_lowPassTrdOrdIRRDoubleG_X": rotation_rate_low_pass_by_hand_trd_ord_double_general[i][0],
                "rotationRate_lowPassTrdOrdIRRDoubleG_Y": rotation_rate_low_pass_by_hand_trd_ord_double_general[i][1],
                "rotationRate_lowPassTrdOrdIRRDoubleG_Z": rotation_rate_low_pass_by_hand_trd_ord_double_general[i][2],

                "rotationRate_lowPassSixOrdIRRDoubleG_X": rotation_rate_low_pass_by_hand_six_ord_double_general[i][0],
                "rotationRate_lowPassSixOrdIRRDoubleG_Y": rotation_rate_low_pass_by_hand_six_ord_double_general[i][1],
                "rotationRate_lowPassSixOrdIRRDoubleG_Z": rotation_rate_low_pass_by_hand_six_ord_double_general[i][2],

                "rotationRate_ewma_X": rotation_rate_ewma[i][0],
                "rotationRate_ewma_Y": rotation_rate_ewma[i][1],

                "rotationRate_lowPass1derivative_X": rotation_rate_low_pass_1_derivative[i][0],
                "rotationRate_lowPass1derivative_Y": rotation_rate_low_pass_1_derivative[i][1],
                "rotationRate_lowPass1derivative_Z": rotation_rate_low_pass_1_derivative[i][2],
                "rotationRate_lowPass2derivative_X": rotation_rate_low_pass_2_derivative[i][0],
                "rotationRate_lowPass2derivative_Y": rotation_rate_low_pass_2_derivative[i][1],
                "rotationRate_lowPass2derivative_Z": rotation_rate_low_pass_2_derivative[i][2],
                # "rotationRate_pca_pc0": reduced_rotation_rate[i][0],
                # "rotationRate_pca_pc1": reduced_rotation_rate[i][1],
                # "rotationRate_pca_pc2": reduced_rotation_rate[i][2],
            })
            if best_axes is not None and best_axis_rot_rate is not None:
                rot_rate_low_plato_starts_new = [plato[0] for plato in rotation_rate_low_pass_zero_platos_new]
                rot_rate_low_plato_stops_new = [plato[1] for plato in rotation_rate_low_pass_zero_platos_new]
                if i in rot_rate_low_plato_starts_new:
                    rotRateLowPassPlatoNew_val = -0.1
                elif i in rot_rate_low_plato_stops_new:
                    rotRateLowPassPlatoNew_val = 0.1
                else:
                    rotRateLowPassPlatoNew_val = 0.23
                frame_attr[f"rotationRate_lowPassZeroPlatoNew{best_axis_rot_rate.name.upper()}"] = rotRateLowPassPlatoNew_val

        out.write_frame(frame_attr, frame["frameStamp"])

    print("Writing Data")
    saveObjectAsJSONIntoZip(
        out.return_data(),
        ZIP_FILE_PATH,
        OUT_FILE_PATH_NAME)
    print("Done")


if __name__ == "__main__":
    # No zip ending.
    folder_name = input("Folder Name: ")
    file = os.path.join(RECORDINGS_PATH, folder_name, f"{folder_name}.zip")
    if not os.path.isfile(file):
        print("Invalid folder name")
        exit()
    transform(folder_name)
