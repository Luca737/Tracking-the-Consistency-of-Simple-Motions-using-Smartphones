//
//  StatisticsView.swift
//  ConsistencyTracker
//
//  Created by Nicola Vidovic on 26.02.23.
//

import Foundation
import SwiftUI

struct StatisticsView: View {
    private let statistics: StatisticsOfIntervals
    private static let consistencyNameValueTuples: [(String, (StatisticsOfIntervals) -> Double)] = [
        ("Whole Interval", { stats in stats.consistency.totalLength }),
        ("Phase 0", { stats in stats.consistency.phase0 }),
        ("Phase 1", { stats in stats.consistency.phase1 }),
        ("Phase 2", { stats in stats.consistency.phase2 }),
        ("Phase 0 %", { stats in stats.consistency.phase0Percentage }),
        ("Phase 1 %", { stats in stats.consistency.phase1Percentage }),
        ("Phase 2 %", { stats in stats.consistency.phase2Percentage }),
    ]
    private static let timingsNameValueTuples: [(String, (StatisticsOfIntervals) -> Double)] = [
        ("Whole Session", { stats in stats.intervalTimings.exerciseTime }),
        ("Active", { stats in stats.intervalTimings.activeTime })
        ]
    private static let timingPhasesNameValueTuples: [(String, (StatisticsOfIntervals) -> (Double, Double))] = [
        ("AVG Interval", { stats in (stats.intervalTimings.avgIntervalTime, stats.intervalTimings.meanDeviationIntervalTime) }),
        ("AVG Phase 0", { stats in (stats.intervalTimings.avgPhase0Time, stats.intervalTimings.meanDeviationPhase0Time) }),
        ("AVG Phase 1", { stats in (stats.intervalTimings.avgPhase1Time, stats.intervalTimings.meanDeviationPhase1Time) }),
        ("AVG Phase 2", { stats in (stats.intervalTimings.avgPhase2Time, stats.intervalTimings.meanDeviationPhase2Time) }),
    ]
    
    init(statistics: StatisticsOfIntervals) {
        self.statistics = statistics
    }
    
    var body: some View {
        List {
            Section {
                ForEach(StatisticsView.consistencyNameValueTuples, id: \.0) { entryName, valGetter in
                    let val = valGetter(statistics)
                    HStack {
                        Text(entryName)
                            .bold()
                        Spacer()
                        Text("\(!val.isNaN ? String(round(val*10000)/100) : "--")%")
                    }
                }
            } header: {
                Label("Consistency", systemImage: "scope")
            }
            Section {
                ForEach(StatisticsView.timingsNameValueTuples, id: \.0) { entryName, valGetter in
                    let val = valGetter(statistics)
                    HStack {
                        Text(entryName)
                            .bold()
                        Spacer()
                        Text("\(!val.isNaN ? String(round(val*100)/100) : "--")s")
                    }
                }
                ForEach(StatisticsView.timingPhasesNameValueTuples, id: \.0) { entryName, valGetter in
                    let (val, meanDev) = valGetter(statistics)
                    HStack {
                        Text(entryName)
                            .bold()
                        Spacer()
                        Text("(±\(String(round(meanDev*100)/100))s) \(!val.isNaN ? String(round(val*100)/100) : "--")s")
                    }
                }
            } header: {
                Label("Timings", systemImage: "clock")
            }
        }
        .listStyle(.grouped)
    }
}

struct StatisticsView_Previews: PreviewProvider {
    static var previews: some View {
        StatisticsView(statistics: StatisticsOfIntervals.sampleData[2])
    }
}
