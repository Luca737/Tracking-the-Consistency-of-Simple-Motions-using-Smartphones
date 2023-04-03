//
//  ConsistencyCalcTests.swift
//  ConsistencyTrackerTests
//
//  Created by Nicola Vidovic on 25.02.23.
//

import XCTest
@testable import ConsistencyTracker

final class ConsistencyCalcTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testExample() throws {
        var intervals: [Interval] = [Interval(start: 0, phase0End: 1, phase1End: 2, stop: 3), Interval(start: 4, phase0End: 5, phase1End: 6, stop: 7)]
        var result = calcConsistency(intervals: intervals)
        XCTAssert(result.totalLength == 0 && result.phase0 == 0 && result.phase1 == 0 && result.phase2 == 0 && result.phase0Percentage == 0 && result.phase1Percentage == 0 && result.phase2Percentage == 0)
    }

    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
