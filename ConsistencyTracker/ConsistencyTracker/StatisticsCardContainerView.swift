//
//  StatisticsCardContainerView.swift
//  ConsistencyTracker
//
//  Created by Nicola Vidovic on 27.02.23.
//

import Foundation
import SwiftUI

struct StatisticsCardContainerView: View {
    private let allStatistics: AllStatistics
    
    init(allStatistics: AllStatistics) {
        self.allStatistics = allStatistics
    }
    
    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(allStatistics.rotationRate, id: \.axis) {stats in
                        NavigationLink(destination: {StatisticsView(statistics: stats)},
                                       label: {StatisticsCardView(statistics: stats, isBestAxis: stats.axis == self.allStatistics.bestRotRateAxis)})
                    }
                } header: {
                    Text("Rotation Rate")
                }
                .headerProminence(.increased)
                Section {
                    ForEach(allStatistics.gravity, id: \.axis) {stats in
                        NavigationLink(destination: {StatisticsView(statistics: stats)},
                                       label: {StatisticsCardView(statistics: stats, isBestAxis: stats.axis == self.allStatistics.bestGravityAxis)})
                    }
                } header: {
                    Text("Gravity")
                }
                .headerProminence(.increased)
                Section {
                    ForEach(allStatistics.linearAcceleration, id: \.axis) {stats in
                        NavigationLink(destination: {StatisticsView(statistics: stats)},
                                       label: {StatisticsCardView(statistics: stats, isBestAxis: stats.axis == self.allStatistics.bestLinearAccelerationAxis)})
                    }
                } header: {
                    Text("Linear Acceleration")
                }
                .headerProminence(.increased)
            }
            .listStyle(.grouped)
            .navigationTitle("Analysis Output")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

struct StatisticsCardContainerView_Previews: PreviewProvider {
    static var previews: some View {
        StatisticsCardContainerView(allStatistics: AllStatistics.sampleData[0])
    }
}
