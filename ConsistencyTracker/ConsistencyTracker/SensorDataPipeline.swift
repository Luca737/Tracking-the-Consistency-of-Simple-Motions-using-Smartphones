//
//  MotionPipeline.swift
//  ConsistencyTracker
//
//  Created by Nicola Vidovic on 24.02.23.
//

import Foundation
import Accelerate

protocol SensorDataPipelineDelegate {
    func analysisFinished()
    func recordingHasStarted()
    func recordingHasStopped()
}

func convertAnnotationsToVIT(intervals: Intervals) -> DataAnnotationsVIT {
    var out = DataAnnotationsVIT(analysisMethod: intervals.analysisMethod, axis: intervals.axis)
    for interval in intervals.intervals {
        out.intervals.append(IntervalVIT(start: interval.start, stop: interval.phase0End, phase: "0"))
        out.intervals.append(IntervalVIT(start: interval.phase0End, stop: interval.phase1End, phase: "1"))
        out.intervals.append(IntervalVIT(start: interval.phase1End, stop: interval.stop, phase: "2"))
    }
    return out
}

class SensorDataPipeline {
    /* Manages Motion Capture and the analysis of the collected Motion Data */
    var delegate: SensorDataPipelineDelegate?
    private let motionManager = SensorDataRecorder()
    
    var motionFrames: Frames = Frames(startTime: Date(), frames: [])  // Dummy so that motionFrames must not be an optional.
    var allIntervals: [Intervals] = []
    var analysisResults: AllStatistics?
    
    // For debugging
//    var filteredData: FilteredData = FilteredData()
    
    var isMotionCaptureActive: Bool {
        motionManager.isRecodingActive
    }
    
    init() {
        self.motionManager.sendMotionDataDelegate = self
    }
    
    func startRecording() {
        // Start Motion Capture
        if delegate == nil {
            debugPrint("No Delegate set")
            return
        }
        motionFrames = Frames(startTime: Date(), frames: [])
        motionManager.startMotionCapture()
        delegate?.recordingHasStarted()
    }
    
    func stopRecording() {
        // Stop Motion Capture
        guard isMotionCaptureActive else {
            debugPrint("MotionCapture is not Active")
            return
        }
        motionManager.stopMotionCapture()
        delegate?.recordingHasStopped()
    }
    
    func analyze() {
        debugPrint("Motion Data is sorted: \(isMotionDataSorted(frames: motionFrames.frames))")
        // TODO: Check everything for copy paste errors!!!!!!!!!! I did but who knows...
        // Analyze and calculate all statistics of the motion data.
        if motionFrames.frames.count == 0 {
            debugPrint("No Frames to Analyze")
            return
        }
//        filteredData = FilteredData() // TODO: For Debugging
        allIntervals = []
        analysisResults = AllStatistics()
        // TODO: Get globalStart and globalStop
        let globalStart, globalStop: Int
        globalStart = 0
        globalStop = motionFrames.frames.count
        
        let timeStamps: [Double] = motionFrames.frames[globalStart..<globalStop].map({ frame in return frame.frameStamp })
        var axis0, axis1, axis2: [Double]
        axis0 = Array(repeating: 0, count: globalStop - globalStart)
        axis1 = Array(repeating: 0, count: globalStop - globalStart)
        axis2 = Array(repeating: 0, count: globalStop - globalStart)
        // MARK: Rotation Rate Analysis
        debugPrint("Analyzing Rotation Rate")
        for (i, frame) in motionFrames.frames[globalStart..<globalStop].enumerated() {
            axis0[i] = frame.frameAttributes.rotationRate_X
            axis1[i] = frame.frameAttributes.rotationRate_Y
            axis2[i] = frame.frameAttributes.rotationRate_Z
        }
        var axes: [Axis] = [.x, .y, .z]
        for (axis, data) in zip(axes, [axis0, axis1, axis2]) {
            let data = lowPass3rdOrder_2_5Hz(data)
//            filteredData.rotRate.append(data)  // TODO: For Debugging
            let detectedIntervals: Intervals = Intervals(
                analysisMethod: .rotationRate,
                axis: axis,
                intervals: analyzeRotationRate(data: data, timeStamps: timeStamps))
            allIntervals.append(detectedIntervals)
            let (nRepeats, consistency, timings) = calcStatistics(of: detectedIntervals.intervals)
            analysisResults?.rotationRate.append(
                StatisticsOfIntervals(analysisMethod: .rotationRate, axis: axis, nRepeats: nRepeats, consistency: consistency, intervalTimings: timings)
            )
        }
        analysisResults?.bestRotRateAxis = axes[autoChooseBestAxis(axes: [axis0, axis1, axis2])]
        // MARK: Gravity Analysis
        debugPrint("Analyzing Gravity")
        for (i, frame) in motionFrames.frames[globalStart..<globalStop].enumerated() {
            axis0[i] = frame.frameAttributes.gravity_X
            axis1[i] = frame.frameAttributes.gravity_Y
            axis2[i] = frame.frameAttributes.gravity_Z
        }
        axis0 = derivative(axis0)
        axis1 = derivative(axis1)
        axis2 = derivative(axis2)
        axes = [.x, .y, .z]
        for (axis, data) in zip(axes, [axis0, axis1, axis2]) {
            debugPrint("axis: \(axis)")
            let data = lowPass3rdOrder_2_5Hz(data)
//            filteredData.grav.append(data)  // TODO: For Debugging
            let detectedIntervals: Intervals = Intervals(
                analysisMethod: .gravity,
                axis: axis,
                intervals: analyzeGravity(data: data, timeStamps: timeStamps))
            allIntervals.append(detectedIntervals)
            let (nRepeats, consistency, timings) = calcStatistics(of: detectedIntervals.intervals)
            analysisResults?.gravity.append(
                StatisticsOfIntervals(analysisMethod: .gravity, axis: axis, nRepeats: nRepeats, consistency: consistency, intervalTimings: timings)
            )
        }
        analysisResults?.bestGravityAxis = axes[autoChooseBestAxis(axes: [axis0, axis1, axis2])]
        // MARK: Linear Acceleration Analysis
        debugPrint("Analyzing Linear Acceleration")
        for (i, frame) in motionFrames.frames[globalStart..<globalStop].enumerated() {
            axis0[i] = frame.frameAttributes.acceleration_X
            axis1[i] = frame.frameAttributes.acceleration_Y
            axis2[i] = frame.frameAttributes.acceleration_Z
        }
        axes = [.x, .y, .z]
        for (axis, data) in zip(axes, [axis0, axis1, axis2]) {
            debugPrint("LinAcc: \(axis)")
            let data = lowPass3rdOrder_1_2Hz(data)
//            filteredData.linAcc.append(data)  // TODO: For Debugging
            let detectedIntervals: Intervals = Intervals(
                analysisMethod: .linearAcceleration,
                axis: axis,
                intervals: analyzeLinearAcceleration(data: data, timeStamps: timeStamps))
            allIntervals.append(detectedIntervals)
            let (nRepeats, consistency, timings) = calcStatistics(of: detectedIntervals.intervals)
            analysisResults?.linearAcceleration.append(
                StatisticsOfIntervals(analysisMethod: .linearAcceleration, axis: axis, nRepeats: nRepeats, consistency: consistency, intervalTimings: timings)
            )
        }
        analysisResults?.bestLinearAccelerationAxis = axes[autoChooseBestAxis(axes: [axis0, axis1, axis2])]
        /// Old PCA implementation left here for when the PCA function is reworked.
//        axis = [.pc0, .pc1, .pc2]
//        var accPCA: [[Double]] = [[], [], []]
//        for frame in motionFrames.frames[globalStart..<globalStop] {
//            accPCA[0].append(frame.frameAttributes.acceleration_X)
//            accPCA[1].append(frame.frameAttributes.acceleration_Y)
//            accPCA[2].append(frame.frameAttributes.acceleration_Z)
//        }
//        accPCA = pca(accPCA)
//        for (axis, pc) in zip(axis, accPCA) {
//            data = lowPass(pc)
//            let detectedIntervals: Intervals = Intervals(
//                analysisMethod: .linearAcceleration,
//                axis: axis,
//                intervals: analyzeLinearAcceleration(data: data, timeStamps: timeStamps))
//            allIntervals.append(detectedIntervals)
//            let (nRepeats, consistency, timings) = calcStatistics(of: detectedIntervals.intervals)
//            analysisResults?.linearAcceleration.append(
//                StatisticsOfIntervals(analysisMethod: .linearAcceleration, axis: axis, nRepeats: nRepeats, consistency: consistency, intervalTimings: timings)
//            )
//        }
        // -------------------------------------------------------
        
        
        // -------------------------------------------------------
        delegate?.analysisFinished()
    }
}

extension SensorDataPipeline: SensorDataRecorderDelegate {
    
    func handleMotionData(sensorData: SensorData) {
        let sensorDataPoint = SensorDataPoint(
            acceleration_X: sensorData.accelerationX,
            acceleration_Y: sensorData.accelerationY,
            acceleration_Z: sensorData.accelerationZ,
            gravity_X: sensorData.gravityX,
            gravity_Y: sensorData.gravityY,
            gravity_Z: sensorData.gravityZ,
            rotationRate_X: sensorData.rotationRateX,
            rotationRate_Y: sensorData.rotationRateY,
            rotationRate_Z: sensorData.rotationRateZ
        )
        let timeSinceSessionStart: Double = sensorData.timeStamp.timeIntervalSince(self.motionFrames.startTime)
        let frame = Frame(frameStamp: timeSinceSessionStart, frameAttributes: sensorDataPoint)
        self.motionFrames.frames.append(frame)
    }
    
}
