//
//  SampleData.swift
//  ConsistencyTracker
//
//  Created by Nicola Vidovic on 26.02.23.
//

import Foundation

extension AllStatistics {
    static let sampleData: [AllStatistics] = [
        AllStatistics(rotationRate: [StatisticsOfIntervals.sampleData[0]], gravity: [StatisticsOfIntervals.sampleData[1]], linearAcceleration: [StatisticsOfIntervals.sampleData[2]], bestRotRateAxis: Axis.x, bestGravityAxis: Axis.x)
    ]
}

extension StatisticsOfIntervals {
    static let sampleData: [StatisticsOfIntervals] = [
        StatisticsOfIntervals(analysisMethod: .rotationRate, axis: .x, nRepeats: 10, consistency: ConsistencyRating.sampleData[0], intervalTimings: IntervalTimings.sampleData[0]),
        StatisticsOfIntervals(analysisMethod: .gravity, axis: .z, nRepeats: 14, consistency: ConsistencyRating.sampleData[1], intervalTimings: IntervalTimings.sampleData[1]),
        StatisticsOfIntervals(analysisMethod: .linearAcceleration, axis: .pc0, nRepeats: 20, consistency: ConsistencyRating.sampleData[2], intervalTimings: IntervalTimings.sampleData[2])
    ]
}

extension IntervalTimings {
    static let sampleData: [IntervalTimings] = [
        IntervalTimings(exerciseTime: 12.2, activeTime: 10.0, avgIntervalTime: 2.032, avgPhase0Time: 12.3123, avgPhase1Time: 1.231, avgPhase2Time: 341.13241, meanDeviationIntervalTime: 0.123, meanDeviationPhase0Time: 1.23, meanDeviationPhase1Time: 12, meanDeviationPhase2Time: 1.1),
        IntervalTimings(exerciseTime: 15.0293824, activeTime: 10.102192, avgIntervalTime: 23.2032939, avgPhase0Time: 12.123123, avgPhase1Time: 41.13124, avgPhase2Time: 1241.1241, meanDeviationIntervalTime: 0.1236242, meanDeviationPhase0Time: 0.0, meanDeviationPhase1Time: 12.12,meanDeviationPhase2Time: 0.0232312),
        IntervalTimings(exerciseTime: 120.120193, activeTime: 12.1231231, avgIntervalTime: 12312.123123131241, avgPhase0Time: 34.3124124, avgPhase1Time: 33.35235, avgPhase2Time: 0.292324, meanDeviationIntervalTime: 1.85, meanDeviationPhase0Time: 9.99, meanDeviationPhase1Time: 12.12, meanDeviationPhase2Time: 0.32342312564243)
    ]
}

extension ConsistencyRating {
    static let sampleData: [ConsistencyRating] = [
        ConsistencyRating(totalLength: 0.98, phase0: Double.nan, phase1: 0.90, phase2: 0.80, phase0Percentage: 0.99, phase1Percentage: 0.69, phase2Percentage: 0.12),
        ConsistencyRating(totalLength: 0.12, phase0: 0.45, phase1: Double.nan, phase2: 0.9, phase0Percentage: 0.99, phase1Percentage: 0.928, phase2Percentage: 0.3948),
        ConsistencyRating(totalLength: 0.3451336213, phase0: 0.492839532, phase1: 0.38583948, phase2: Double.nan, phase0Percentage: 0.19384723, phase1Percentage: 0.384248028, phase2Percentage: 0.4958392840)
    ]
}
