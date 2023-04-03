//
//  SingleStatisticsView.swift
//  ConsistencyTracker
//
//  Created by Nicola Vidovic on 26.02.23.
//

import Foundation
import SwiftUI

struct StatisticsCardView: View {
    private let statistics: StatisticsOfIntervals
    private let isBestAxis: Bool
    
    init(statistics: StatisticsOfIntervals, isBestAxis: Bool = false) {
        self.statistics = statistics
        self.isBestAxis = isBestAxis
    }
    
    var body: some View {
        VStack(alignment: .leading) {
            HStack {
                Text("\(statistics.axis.rawValue.uppercased()) - Axis").font(.headline)
                Spacer()
                if isBestAxis {
                    Image(systemName: "star.fill").foregroundColor(.blue)
                }
            }
            Spacer()
            HStack {
                Image(systemName: "number").foregroundColor(.blue)
                Text(String(statistics.nRepeats))
                Spacer()
                Text("\(String(round(statistics.consistency.totalLength*1000)/10))%")
                Image(systemName: "scope").foregroundColor(.blue)
            }
        }
        .padding()
    }
}

struct StatisticsCardView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(StatisticsOfIntervals.sampleData, id: \.consistency.totalLength) {sample in
                        NavigationLink(destination: {StatisticsView(statistics: sample)}, label: {StatisticsCardView(statistics: sample, isBestAxis: true)})
                    }
                } header: {
                    Text("Rotation Rate")
                }
                .headerProminence(.increased)
            }
            .listStyle(.grouped)
            .navigationTitle("Analysis Output")
            .navigationBarTitleDisplayMode(.large)
        }
    }
//    static var previews: some View {
//        NavigationStack {
//            List {
//                Section {
//                    ForEach(0..<3) {i in
//                        NavigationLink(destination: {StatisticsCardView(statistics: StatisticsOfIntervals.sampleData[0])}, label: {StatisticsCardView(statistics: StatisticsOfIntervals.sampleData[i])})
//                    }
//                } header: {
//                    Text("Rotation Rate")
//                }
//                .headerProminence(.increased)
//            }
//            .listStyle(.grouped)
//            .navigationTitle("Analysis Output")
//            .navigationBarTitleDisplayMode(.large)
//        }
//    }
}
