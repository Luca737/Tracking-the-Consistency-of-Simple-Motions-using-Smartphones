//
//  SensorDataPipelineModel.swift
//  ConsistencyTracker
//
//  Created by Nicola Vidovic on 25.02.23.
//

import Foundation

struct AllStatistics {
    var rotationRate: [StatisticsOfIntervals] = []
    var gravity: [StatisticsOfIntervals] = []
    var linearAcceleration: [StatisticsOfIntervals] = []
    var bestRotRateAxis, bestGravityAxis, bestLinearAccelerationAxis: Axis?
}

struct DataAnnotationsVIT: Encodable {
    let analysisMethod: AnalyzedDataType
    let axis: Axis
    var intervals: [IntervalVIT] = []
    
    enum CodingKeys: String, CodingKey {
        case analysisMethod
        case axis
        case applicationName
        case intervals
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self.analysisMethod.rawValue, forKey: .analysisMethod)
        try container.encode(self.axis.rawValue, forKey: .axis)
        try container.encode("AutomaticAnnotations", forKey: .applicationName)
        try container.encode(self.intervals, forKey: .intervals)
    }
}

struct IntervalVIT: Encodable {
    let start: Double
    let stop: Double
    let phase: String
    
    enum CodingKeys: String, CodingKey {
        case start
        case stop = "end"
        case annotations
    }
    
    enum AnnotationKeys: String, CodingKey {
        case phase
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        let startString: FormattedElapsedTime = formatTimeInterval(interval: start)
        let stopString: FormattedElapsedTime = formatTimeInterval(interval: stop)
        try container.encode(startString, forKey: .start)
        try container.encode(stopString, forKey: .stop)
        var nested_container = container.nestedContainer(keyedBy: AnnotationKeys.self, forKey: .annotations)
        try nested_container.encode(phase, forKey: .phase)
    }
}

struct FilteredData {
    var rotRate: [[Double]] = []
    var linAcc: [[Double]] = []
    var grav: [[Double]] = []
}
