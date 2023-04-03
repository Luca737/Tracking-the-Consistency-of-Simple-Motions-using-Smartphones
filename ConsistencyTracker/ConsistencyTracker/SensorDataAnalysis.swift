//
//  SensorDataAnalysis.swift
//  CollectSensorData
//
//  Created by Nicola Vidovic on 21.02.23.
//

import Foundation

// TODO: Possibility of swapped data points due to scheduling issues. Sensor data my need to be sorted (Insertion sort would be best)

enum AnalyzedDataType: String {
    case rotationRate
    case gravity
    case linearAcceleration
}

enum Axis: String {
    case x, y, z, pc0, pc1, pc2
}

struct Intervals {
    let analysisMethod: AnalyzedDataType
    let axis: Axis
    let intervals: [Interval]
}

struct Interval {
    let start: Double
    let phase0End: Double
    let phase1End: Double
    let stop: Double
}

struct StatisticsOfIntervals {
    let analysisMethod: AnalyzedDataType
    let axis: Axis
    let nRepeats: Int  // Number of detected intervals or repetitions done by the user.
    let consistency: ConsistencyRating
    let intervalTimings: IntervalTimings
}

struct IntervalTimings {
    let exerciseTime: Double
    let activeTime: Double  // Total of all intervals without the pauses between intervals/repeats
    let avgIntervalTime: Double
    let avgPhase0Time: Double
    let avgPhase1Time: Double
    let avgPhase2Time: Double
    let meanDeviationIntervalTime: Double
    let meanDeviationPhase0Time: Double
    let meanDeviationPhase1Time: Double
    let meanDeviationPhase2Time: Double
}

struct ConsistencyRating {
    let totalLength: Double
    let phase0: Double
    let phase1: Double
    let phase2: Double
//    let avgPhases: Double
    let phase0Percentage: Double
    let phase1Percentage: Double
    let phase2Percentage: Double
    // Within consistency
}

func calcStatistics(of intervals: [Interval]) -> (nRepeats: Int, consistency: ConsistencyRating, timings: IntervalTimings) {
    /* Assumes that the intervals are sorted. (For total exercise time) */
    if intervals.count == 0 {
        return (0, ConsistencyRating(totalLength: 0, phase0: 0, phase1: 0, phase2: 0, phase0Percentage: 0, phase1Percentage: 0, phase2Percentage: 0), IntervalTimings(exerciseTime: 0, activeTime: 0, avgIntervalTime: 0, avgPhase0Time: 0, avgPhase1Time: 0, avgPhase2Time: 0, meanDeviationIntervalTime: 0, meanDeviationPhase0Time: 0, meanDeviationPhase1Time: 0, meanDeviationPhase2Time: 0))
    }
    
    /* Mean absolute Deviation of total length and each of the phases */
    // TODO: Add within consistency -> Proportions between phases across intervals.
    // TODO: Add avgPhases consistency -> Simple average okey? -> Will be printed on the CardView representation of an Interval Statistics.
    let numOfIntervals = Double(intervals.count)
    // = [total, phase0, phase1, phase2]
    var meanOfLengths: [Double] = [0, 0, 0, 0]
    var meanAbsoluteDeviation: [Double] = [0, 0, 0, 0]
    // = [phase0, phase1, phase2]; percentage of the interval used by each phase respectively.
    var meanOfPercentages: [Double] = [0, 0, 0]
    var meanAbsoluteDeviationOfPercentages: [Double] = [0, 0, 0]
    var lengthOfInterval, lengthPhase0, lengthPhase1, lengthPhase2: Double
    for interval in intervals {
        lengthOfInterval = interval.stop - interval.start
        lengthPhase0 = interval.phase0End - interval.start
        lengthPhase1 = interval.phase1End - interval.phase0End
        lengthPhase2 = interval.stop - interval.phase1End
        meanOfLengths[0] += lengthOfInterval
        meanOfLengths[1] += lengthPhase0
        meanOfLengths[2] += lengthPhase1
        meanOfLengths[3] += lengthPhase2
        meanOfPercentages[0] += lengthPhase0 / lengthOfInterval
        meanOfPercentages[1] += lengthPhase1 / lengthOfInterval
        meanOfPercentages[2] += lengthPhase2 / lengthOfInterval
    }
    let totalIntervalTime = meanOfLengths[1...].reduce(0, +)
    meanOfLengths = meanOfLengths.map({ val in return val / numOfIntervals })
    meanOfPercentages = meanOfPercentages.map({ val in return val / numOfIntervals })
    for interval in intervals {
        lengthOfInterval = interval.stop - interval.start
        lengthPhase0 = interval.phase0End - interval.start
        lengthPhase1 = interval.phase1End - interval.phase0End
        lengthPhase2 = interval.stop - interval.phase1End
        meanAbsoluteDeviation[0] += abs(meanOfLengths[0] - lengthOfInterval)
        meanAbsoluteDeviation[1] += abs(meanOfLengths[1] - lengthPhase0)
        meanAbsoluteDeviation[2] += abs(meanOfLengths[2] - lengthPhase1)
        meanAbsoluteDeviation[3] += abs(meanOfLengths[3] - lengthPhase2)
        meanAbsoluteDeviationOfPercentages[0] += abs(meanOfPercentages[0] - (lengthPhase0 / lengthOfInterval))
        meanAbsoluteDeviationOfPercentages[1] += abs(meanOfPercentages[1] - (lengthPhase1 / lengthOfInterval))
        meanAbsoluteDeviationOfPercentages[2] += abs(meanOfPercentages[2] - (lengthPhase2 / lengthOfInterval))
    }
    meanAbsoluteDeviation = meanAbsoluteDeviation.map({ val in return val / numOfIntervals})
    meanAbsoluteDeviationOfPercentages = meanAbsoluteDeviationOfPercentages.map({ val in return val / numOfIntervals })

    return (intervals.count,
            ConsistencyRating(
                totalLength: meanAbsoluteDeviation[0]/meanOfLengths[0],
                phase0: meanAbsoluteDeviation[1]/meanOfLengths[1],
                phase1: meanAbsoluteDeviation[2]/meanOfLengths[2],
                phase2: meanAbsoluteDeviation[3]/meanOfLengths[3],
                phase0Percentage: meanAbsoluteDeviationOfPercentages[0]/meanOfPercentages[0],
                phase1Percentage: meanAbsoluteDeviationOfPercentages[1]/meanOfPercentages[1],
                phase2Percentage: meanAbsoluteDeviationOfPercentages[2]/meanOfPercentages[2]
            ),
            IntervalTimings(
                exerciseTime: intervals[intervals.count-1].stop - intervals[0].start,
                activeTime: totalIntervalTime,
                avgIntervalTime: totalIntervalTime / numOfIntervals,
                avgPhase0Time: meanOfLengths[1],
                avgPhase1Time: meanOfLengths[2],
                avgPhase2Time: meanOfLengths[3],
                meanDeviationIntervalTime: meanAbsoluteDeviation[0],
                meanDeviationPhase0Time: meanAbsoluteDeviation[1],
                meanDeviationPhase1Time: meanAbsoluteDeviation[2],
                meanDeviationPhase2Time: meanAbsoluteDeviation[3]
            )
    )
}


//func analyzeLinearAcceleration(data: [Double], timeStamps: [Double]) -> [Interval] {
//    var intervals: [Interval] = []
//
//    return intervals
//}


func analyzeLinearAcceleration(data: [Double], timeStamps: [Double]) -> [Interval] {
    var detectedIntervals: [Interval] = []
    
    let minimum_data = data.min()
    let maximum_data = data.max()
    guard let minimum_data, let maximum_data else {
        return detectedIntervals
    }
    
    let minimaBaseline = minimum_data / 3.05
    let maximaBaseline = maximum_data / 3.05

    if minimaBaseline >= 0 || maximaBaseline <= 0 {
        return []
    }

    let (minima, maxima) = calcExtremaWithBaseline(data: data, minimaBaseline: minimaBaseline, maximaBaseline: maximaBaseline)
    let (zeroPoints, zeroPlatos) = calcZeroPointsPlatos(data: data, time_stamps: timeStamps, zeroPlatoValRadius: 0.02, zeroPlatoSlopeRadius: 0.4, minPlatoLengthSecs: 0.0, allowedSkipLengthBetweenPlatosSecs: 0.0)

    // Requirement for at least one motion interval to exist / be detected by this method.
    if minima.count + maxima.count < 3 || minima.count == 0 || maxima.count == 0 {
        return []
    }

    // TODO: This is potentially more dangerous for linear acceleration. There are more odd bumps, similar to 'real' bumps.
    let intervalStartsWithMinima = minima[0] < maxima[0]

    let startStopExtrema: [Int]
    let middleExtrema: [Int]
    if intervalStartsWithMinima {
        startStopExtrema = minima
        middleExtrema = maxima
    } else {
        startStopExtrema = maxima
        middleExtrema = minima
    }

    var zeroPointI: Int = 0
    var zeroPlatoI: Int? = zeroPlatos.count > 0 ? 0 : nil
    var middleExtremaI: Int = 0
    for startStopExtrema_i in 0..<startStopExtrema.count-1 {
        let startExtremum = startStopExtrema_i
        let stopExtremum = startStopExtrema_i+1
        let middleExtremum0 = getFirstIndexOf(list: middleExtrema, startingAt: middleExtremaI, where: {el in return el > startStopExtrema[startExtremum]})
        if middleExtremum0 == nil {
            // No middle extrema left means no further intervals left.
            break
        }
        guard let middleExtremum0 = middleExtremum0, middleExtrema[middleExtremum0] < startStopExtrema[stopExtremum] else {
            continue
        }
        var middleExtremum1: Int? = nil
        if middleExtremum0+1 < middleExtrema.count && middleExtrema[middleExtremum0+1] < startStopExtrema[stopExtremum] {
            middleExtremum1 = middleExtremum0+1
            if middleExtremum0+2 < middleExtrema.count && middleExtrema[middleExtremum0+2] < startStopExtrema[stopExtremum] {
                middleExtremaI += 3
                continue
            }
        }

        // Valid interval found.
        // -- Find start point -------------------------------------------------
        // Find closest potential starting point on the left of the first extremum.
        // Find closest zero point.
        var keyPointCandidate0: Int? = nil
        var keyPointCandidate1: Int? = nil
        let closestLeftZeroPointI = getFirstIndexOf(list: zeroPoints, startingAt: zeroPointI, offset: -1, where: {el in return el > startStopExtrema[startExtremum]})
        if closestLeftZeroPointI! >= 0 {
            keyPointCandidate0 = zeroPoints[closestLeftZeroPointI!]
            // +3 since there are at least two more zero points between the first and last extrema that are of no interest.
            zeroPointI = closestLeftZeroPointI! + 3
        } else {
            keyPointCandidate0 = 0
        }
        // Find stop of closest zero plato.
        if zeroPlatoI != nil {
            var closestZeroPlatoI = getFirstIndexOf(list: zeroPlatos, startingAt: zeroPlatoI!, offset: -1, where: {interval in return interval.1 > startStopExtrema[startExtremum]})
            if closestZeroPlatoI == nil || closestZeroPlatoI! >= 0 {
                zeroPlatoI = closestZeroPlatoI
                if closestZeroPlatoI == nil {
                    closestZeroPlatoI = zeroPlatos.count - 1
                }
                if zeroPlatos[closestZeroPlatoI!].1 < startStopExtrema[startExtremum] {
                    keyPointCandidate1 = zeroPlatos[closestZeroPlatoI!].1
                }
            }
        }
        // Select closest:
        let intervalStart: Int
        if let keyPointCandidate1 {
            intervalStart = max(keyPointCandidate0!, keyPointCandidate1)
        } else {
            intervalStart = keyPointCandidate0!
        }

        // -- Find ends of phase0 && phase1 -----------------------------------
        // Use the first point below a baseline (calculated to be between the local max && min) from the left && from the right
        // as the start && stop respectively.
        let phase0End, phase1End: Int
        if let middleExtremum1 {
            let relativeUpper, relativeLower: Double
            if intervalStartsWithMinima {
                relativeUpper = min(data[middleExtrema[middleExtremum0]], data[middleExtrema[middleExtremum1]])
                relativeLower = max(0, data[middleExtrema[middleExtremum0]..<middleExtrema[middleExtremum1]].min()!)
                // baseline = relativeUpper - (relativeUpper - relativeLower) / 2.3
            } else {
                relativeUpper = max(data[middleExtrema[middleExtremum0]], data[middleExtrema[middleExtremum1]])
                relativeLower = min(0, data[middleExtrema[middleExtremum0]..<middleExtrema[middleExtremum1]].max()!)
                // baseline = relativeUpper - (relativeUpper - relativeLower) / 2.3
            }
            let baseline: Double = (relativeUpper + relativeLower) / 2
            let factor: Double = 1 - 2 * (1 - (intervalStartsWithMinima ? 1 : 0))  // Correct for searching below || above 0.
            let sliceBetweenMiddleExtrema: [Double] = Array(data[middleExtrema[middleExtremum0]...middleExtrema[middleExtremum1]])
            phase0End = middleExtrema[middleExtremum0] + getFirstIndexOf(list: sliceBetweenMiddleExtrema, startingAt: 0, where: {el in return el*factor < baseline*factor})!
            phase1End = middleExtrema[middleExtremum1] - getFirstIndexOf(list: sliceBetweenMiddleExtrema.reversed(), startingAt: 0, where: {el in return el*factor < baseline*factor})!
        } else {
            phase0End = middleExtrema[middleExtremum0]
            phase1End = phase0End
        }

        // -- Find end point ---------------------------------------------------
        // Find closest end of the motion interval from the right of the second extremum.
        // Find closest zero_point
        keyPointCandidate0 = nil
        keyPointCandidate1 = nil
        let closestRightZeroPointI = getFirstIndexOf(list: zeroPoints, startingAt: zeroPointI, where: {el in return el > startStopExtrema[stopExtremum]})
        if let closestRightZeroPointI {
            keyPointCandidate0 = zeroPoints[closestRightZeroPointI]
            zeroPointI = closestRightZeroPointI
        } else {
            keyPointCandidate0 = timeStamps.count - 1
        }
        // Find start of closest zero plato.
        if zeroPlatoI != nil {
            let closestZeroPlatoI = getFirstIndexOf(list: zeroPlatos, startingAt: zeroPlatoI!, where: {interval in return interval.0 > startStopExtrema[stopExtremum]})
            zeroPlatoI = closestZeroPlatoI
            if closestZeroPlatoI != nil && zeroPlatos[closestZeroPlatoI!].0 > startStopExtrema[stopExtremum] {
                keyPointCandidate1 = zeroPlatos[closestZeroPlatoI!].0
            }
        }
        // Select closest:
        let intervalStop: Int
        if let keyPointCandidate1 {
            intervalStop = min(keyPointCandidate0!, keyPointCandidate1)
        } else {
            intervalStop = keyPointCandidate0!
        }
        // ---------------------------------------------------------------------
        
        if intervalStart > phase0End || phase0End > phase1End || phase1End > intervalStop {
            debugPrint("Acceleration: Intervals are wrong")
            debugPrint("\(intervalStart), \(phase0End), \(phase1End), \(intervalStop)")
        }
        
        detectedIntervals.append(Interval(start: timeStamps[intervalStart], phase0End: timeStamps[phase0End], phase1End: timeStamps[phase1End], stop: timeStamps[intervalStop]))

        middleExtremaI += middleExtremum1 == nil ? 1 : 2
    }
    
    debugPrint("In Acc: #Intervals \(detectedIntervals.count)")

    return splitOverlappingIntervals(intervals: detectedIntervals)
}


func splitOverlappingIntervals(intervals: [Interval]) -> [Interval] {
    var intervals = intervals
    if intervals.count < 2 {
        return intervals
    }
    for i in 0..<intervals.count-1 {
        if intervals[i].stop > intervals[i+1].start {
            var interval: Interval = intervals[i]
            intervals[i] = Interval(start: interval.start, phase0End: interval.phase0End, phase1End: interval.phase1End, stop: interval.stop)
            interval = intervals[i+1]
            intervals[i+1] = Interval(start: interval.start, phase0End: interval.phase0End, phase1End: interval.phase1End, stop: interval.stop)
        }
    }
    return intervals
}


func analyzeRotationRate(data: [Double], timeStamps: [Double]) -> [Interval] {
    /* V2.3, however, the safety fixes from V2.1.1 are still implemented. They cost barely anything and ensure that they can't happen for real. */
    var detectedIntervals: [Interval] = []
    
    let minimum_data = data.min()
    let maximum_data = data.max()
    guard let minimum_data, let maximum_data else {
        return detectedIntervals
    }
    let minimaBaseline: Double = minimum_data / 2
    let maximaBaseline: Double = maximum_data / 2
    
    let (minima, maxima) = calcExtremaWithBaseline(data: data, minimaBaseline: minimaBaseline, maximaBaseline: maximaBaseline)


    if minima.count == 0 || maxima.count == 0{
        return detectedIntervals
    }
    
    let absMinExtrema = min(minima.map({abs(data[$0])}).min()!, maxima.map({data[$0]}).min()!)
    let d: [Double] = derivative(data)
    let maxD: Double = d.map({abs($0)}).max()!
    let maxValZeroPlato: Double = absMinExtrema / 5.8
    let maxSlopeZeroPlato: Double = maxD / 7
    
    let (zeroPoints, zeroPlatos) = calcZeroPointsPlatos(data: data, time_stamps: timeStamps, zeroPlatoValRadius: maxValZeroPlato, zeroPlatoSlopeRadius: maxSlopeZeroPlato, minPlatoLengthSecs: 0.0, allowedSkipLengthBetweenPlatosSecs: 0.1)
    
    let intervalStartsWithMinima: Bool = minima[0] < maxima[0]
    
    let starts: [Int]
    let stops: [Int]
    if intervalStartsWithMinima {
        starts = minima
        stops = maxima
    } else {
        starts = maxima
        stops = minima
    }
    
    var zeroPointI: Int = 0
    var zeroPlatoI: Int? = zeroPlatos.count > 0 ? 0 : nil
    var startI: Int = 0
    var stopI: Int = 0
    while (startI < starts.count) && (stopI < stops.count) {
        if starts[startI] > stops[stopI] {
            stopI += 1
            continue
        }
        if startI + 1 < starts.count && starts[startI+1] < stops[stopI] {
            startI += 1
            continue
        }
        let intervalStart: Double
        let endPhase0: Double
        let endPhase1: Double
        let intervalStop: Double
        // Start stop pair found.
        // Find edges of interval and end of phase 0 and phase 1:
        // -- Find start point -------------------------------------------------
        // Find closest potential starting point on the left of the first extremum.
        // Find closest zero point.
        var keyPointCandidate0: Int
        var keyPointCandidate1: Int? = nil
        // Will always return a value (default (nil) is not possible) as long as maxima and minima are not empty (there is always a zero point between them).
        var closestLeftZeroPointI: Int = getFirstIndexOf(list: zeroPoints, startingAt: zeroPointI, offset: -1, where: {el in return el > starts[startI]})!
        if closestLeftZeroPointI >= 0 {
            keyPointCandidate0 = zeroPoints[closestLeftZeroPointI]
            // +1 since from this point on only zero points after the first extrema are relevant.
            zeroPointI = closestLeftZeroPointI + 1
        } else {
            keyPointCandidate0 = 0
        }
        // Find stop of closest zero plato to the left of the first extremum.
        if let zeroPlatoICopy = zeroPlatoI {
            var closestZeroPlatoI = getFirstIndexOf(list: zeroPlatos, startingAt: zeroPlatoICopy, offset: -1, where: {interval in return interval.1 > starts[startI]})
            if (closestZeroPlatoI == nil) || (closestZeroPlatoI! >= 0) {
                zeroPlatoI = closestZeroPlatoI
                if closestZeroPlatoI == nil {
                    closestZeroPlatoI = zeroPlatos.count - 1
                }
                if zeroPlatos[closestZeroPlatoI!].1 < starts[startI] {
                    keyPointCandidate1 = zeroPlatos[closestZeroPlatoI!].1
                }
            }
        }
        // Select closest:
        if let keyPointCandidate1 {
            intervalStart = timeStamps[max(keyPointCandidate0, keyPointCandidate1)]
        } else {
            intervalStart = timeStamps[keyPointCandidate0]
        }
        
        // -- Find end of phase 0 ----------------------------------------------
        // Find closest end of phase 0 / start of phase 1 from the right of the first extremum.
        // Find closest zero point.
        keyPointCandidate1 = nil
        var closestRightZeroPointI = closestLeftZeroPointI + 1
        keyPointCandidate0 = zeroPoints[closestRightZeroPointI]
        // Find start of closest zero_plato to the right of the first extremum.
        if let zeroPlatoICopy = zeroPlatoI {
            let closestZeroPlatoI = getFirstIndexOf(list: zeroPlatos, startingAt: zeroPlatoICopy, where: {interval in return interval.0 > starts[startI]})
            zeroPlatoI = closestZeroPlatoI
            if let closestZeroPlatoI, zeroPlatos[closestZeroPlatoI].0 > starts[startI] {
                keyPointCandidate1 = zeroPlatos[closestZeroPlatoI].0
            }
        }
        // Select closest:
        if let keyPointCandidate1 {
            endPhase0 = timeStamps[min(keyPointCandidate0, keyPointCandidate1)]
        } else {
            endPhase0 = timeStamps[keyPointCandidate0]
        }
        
        // -- Find end of phase 1 ----------------------------------------------
        // Find closest end of phase 1 / start of phase 2 from the left of the second extremum.
        keyPointCandidate1 = nil
        // Get closest zero point to the left of the second extremum.
        closestLeftZeroPointI = getFirstIndexOf(list: zeroPoints, startingAt: zeroPointI, offset: -1, where: {el in return el > stops[stopI]}) ?? zeroPoints.count - 1
        keyPointCandidate0 = zeroPoints[closestLeftZeroPointI]
        zeroPointI = closestLeftZeroPointI
        // Get closest end of the closest zero plato to left of the second extremum.
        if let zeroPlatoICopy = zeroPlatoI {
            var closestZeroPlatoI = getFirstIndexOf(list: zeroPlatos, startingAt: zeroPlatoICopy, offset: -1, where: {interval in return interval.1 > stops[stopI]})
            if (closestZeroPlatoI == nil) || (closestZeroPlatoI! >= 0) {
                zeroPlatoI = closestZeroPlatoI
                if closestZeroPlatoI == nil {
                    closestZeroPlatoI = zeroPlatos.count - 1
                }
                if zeroPlatos[closestZeroPlatoI!].1 < stops[stopI] {
                    keyPointCandidate1 = zeroPlatos[closestZeroPlatoI!].1
                }
            }
        }
        // Select closest:
        if let keyPointCandidate1 {
            endPhase1 = timeStamps[max(keyPointCandidate0, keyPointCandidate1)]
        } else {
            endPhase1 = timeStamps[keyPointCandidate0]
        }
        
        // -- Find end point ---------------------------------------------------
        // Find closest end of the motion interval from the right of the second extremum.
        keyPointCandidate1 = nil
        // Find closest zero_point to the right of the second extremum.
        closestRightZeroPointI = getFirstIndexOf(list: zeroPoints, startingAt: zeroPointI, where: {el in return el > stops[stopI]}) ?? -1
        if closestRightZeroPointI >= 0 {
            keyPointCandidate0 = zeroPoints[closestRightZeroPointI]
            zeroPointI = closestRightZeroPointI
        } else {
            // Use end of recording (or end of the data given) as candidate 0.
            keyPointCandidate0 = timeStamps.count - 1
        }
        // Find start of the closest zero plato to the right if the second extremum.
        if let zeroPlatoICopy = zeroPlatoI {
            let closestZeroPlatoI = getFirstIndexOf(list: zeroPlatos, startingAt: zeroPlatoICopy, where: {interval in return interval.0 > stops[stopI]})
            zeroPlatoI = closestZeroPlatoI
            if let closestZeroPlatoI, zeroPlatos[closestZeroPlatoI].0 > stops[stopI] {
                keyPointCandidate1 = zeroPlatos[closestZeroPlatoI].0
            }
        }
        // Select closest:
        if let keyPointCandidate1 {
            intervalStop = timeStamps[min(keyPointCandidate0, keyPointCandidate1)]
        } else {
            intervalStop = timeStamps[keyPointCandidate0]
        }
        // ---------------------------------------------------------------------
        
        startI += 1
        stopI += 1
        
        if intervalStart > endPhase0 || endPhase0 > endPhase1 || endPhase1 > intervalStop {
            debugPrint("Intervals are wrong")
            debugPrint("\(intervalStart), \(endPhase0), \(endPhase1), \(intervalStop)")
        }
        detectedIntervals.append(Interval(start: intervalStart, phase0End: endPhase0, phase1End: endPhase1, stop: intervalStop))
    }
    
    return detectedIntervals
}

                                                                                

func analyzeGravity(data: [Double], timeStamps: [Double]) -> [Interval] {
    /// data is expected to be the derivative of the gravity data. As the derivative is required outside of this function as well this is done due to efficiency.
//    let gravityLowPassDerivative1 = derivative(data)
    let intervals = analyzeRotationRateV2_1_1g(data: data, timeStamps: timeStamps)
    
    return intervals
}

func analyzeRotationRateV2_1_1g(data: [Double], timeStamps: [Double]) -> [Interval] {
    var detectedIntervals: [Interval] = []
    
    let minimum_data = data.min()
    let maximum_data = data.max()
    guard let minimum_data, let maximum_data else {
        return detectedIntervals
    }
    let minimaBaseline: Double = minimum_data / 2
    let maximaBaseline: Double = maximum_data / 2
    
    let (minima, maxima) = calcExtremaWithBaseline(data: data, minimaBaseline: minimaBaseline, maximaBaseline: maximaBaseline)
    let (zeroPoints, zeroPlatos) = calcZeroPointsPlatos(data: data, time_stamps: timeStamps, zeroPlatoValRadius: 0.1, zeroPlatoSlopeRadius: 0.75, minPlatoLengthSecs: 0.0, allowedSkipLengthBetweenPlatosSecs: 0.1)

    if minima.count == 0 || maxima.count == 0 || zeroPoints.count == 0 {
        return detectedIntervals
    }
    
    let intervalStartsWithMinima: Bool = minima[0] < maxima[0]
    
    let starts: [Int]
    let stops: [Int]
    if intervalStartsWithMinima {
        starts = minima
        stops = maxima
    } else {
        starts = maxima
        stops = minima
    }
    
    var zeroPointI: Int = 0
    var zeroPlatoI: Int? = zeroPlatos.count > 0 ? 0 : nil
    var startI: Int = 0
    var stopI: Int = 0
    while (startI < starts.count) && (stopI < stops.count) {
        if starts[startI] > stops[stopI] {
            stopI += 1
            continue
        }
        if startI + 1 < starts.count && starts[startI+1] < stops[stopI] {
            startI += 1
            continue
        }
        let intervalStart: Double
        let endPhase0: Double
        let endPhase1: Double
        let intervalStop: Double
        // Start stop pair found.
        // Find edges of interval and end of phase 0 and phase 1:
        // -- Find start point -------------------------------------------------
        // Find closest potential starting point on the left of the first extremum.
        // Find closest zero point.
        var keyPointCandidate0: Int
        var keyPointCandidate1: Int? = nil
        // Will always return a value (default (nil) is not possible) as long as maxima and minima are not empty (there is always a zero point between them).
        var closestLeftZeroPointI: Int = getFirstIndexOf(list: zeroPoints, startingAt: zeroPointI, offset: -1, where: {el in return el > starts[startI]})!
        if closestLeftZeroPointI >= 0 {
            keyPointCandidate0 = zeroPoints[closestLeftZeroPointI]
            // +1 since from this point on only zero points after the first extrema are relevant.
            zeroPointI = closestLeftZeroPointI + 1
        } else {
            keyPointCandidate0 = 0
        }
        // Find stop of closest zero plato to the left of the first extremum.
        if let zeroPlatoICopy = zeroPlatoI {
            var closestZeroPlatoI = getFirstIndexOf(list: zeroPlatos, startingAt: zeroPlatoICopy, offset: -1, where: {interval in return interval.1 > starts[startI]})
            if (closestZeroPlatoI == nil) || (closestZeroPlatoI! >= 0) {
                zeroPlatoI = closestZeroPlatoI
                if closestZeroPlatoI == nil {
                    closestZeroPlatoI = zeroPlatos.count - 1
                }
                if zeroPlatos[closestZeroPlatoI!].1 < starts[startI] {
                    keyPointCandidate1 = zeroPlatos[closestZeroPlatoI!].1
                }
            }
        }
        // Select closest:
        if let keyPointCandidate1 {
            intervalStart = timeStamps[max(keyPointCandidate0, keyPointCandidate1)]
        } else {
            intervalStart = timeStamps[keyPointCandidate0]
        }
        
        // -- Find end of phase 0 ----------------------------------------------
        // Find closest end of phase 0 / start of phase 1 from the right of the first extremum.
        // Find closest zero point.
        keyPointCandidate1 = nil
        var closestRightZeroPointI = closestLeftZeroPointI + 1
        keyPointCandidate0 = zeroPoints[closestRightZeroPointI]
        // Find start of closest zero_plato to the right of the first extremum.
        if let zeroPlatoICopy = zeroPlatoI {
            let closestZeroPlatoI = getFirstIndexOf(list: zeroPlatos, startingAt: zeroPlatoICopy, where: {interval in return interval.0 > starts[startI]})
            zeroPlatoI = closestZeroPlatoI
            if let closestZeroPlatoI, zeroPlatos[closestZeroPlatoI].0 > starts[startI] {
                keyPointCandidate1 = zeroPlatos[closestZeroPlatoI].0
            }
        }
        // Select closest:
        if let keyPointCandidate1 {
            endPhase0 = timeStamps[min(keyPointCandidate0, keyPointCandidate1)]
        } else {
            endPhase0 = timeStamps[keyPointCandidate0]
        }
        
        // -- Find end of phase 1 ----------------------------------------------
        // Find closest end of phase 1 / start of phase 2 from the left of the second extremum.
        keyPointCandidate1 = nil
        // Get closest zero point to the left of the second extremum.
        closestLeftZeroPointI = getFirstIndexOf(list: zeroPoints, startingAt: zeroPointI, offset: -1, where: {el in return el > stops[stopI]}) ?? zeroPoints.count - 1
        keyPointCandidate0 = zeroPoints[closestLeftZeroPointI]
        zeroPointI = closestLeftZeroPointI
        // Get closest end of the closest zero plato to left of the second extremum.
        if let zeroPlatoICopy = zeroPlatoI {
            var closestZeroPlatoI = getFirstIndexOf(list: zeroPlatos, startingAt: zeroPlatoICopy, offset: -1, where: {interval in return interval.1 > stops[stopI]})
            if (closestZeroPlatoI == nil) || (closestZeroPlatoI! >= 0) {
                zeroPlatoI = closestZeroPlatoI
                if closestZeroPlatoI == nil {
                    closestZeroPlatoI = zeroPlatos.count - 1
                }
                if zeroPlatos[closestZeroPlatoI!].1 < stops[stopI] {
                    keyPointCandidate1 = zeroPlatos[closestZeroPlatoI!].1
                }
            }
        }
        // Select closest:
        if let keyPointCandidate1 {
            endPhase1 = timeStamps[max(keyPointCandidate0, keyPointCandidate1)]
        } else {
            endPhase1 = timeStamps[keyPointCandidate0]
        }
        
        // -- Find end point ---------------------------------------------------
        // Find closest end of the motion interval from the right of the second extremum.
        keyPointCandidate1 = nil
        // Find closest zero_point to the right of the second extremum.
        closestRightZeroPointI = getFirstIndexOf(list: zeroPoints, startingAt: zeroPointI, where: {el in return el > stops[stopI]}) ?? -1
        if closestRightZeroPointI >= 0 {
            keyPointCandidate0 = zeroPoints[closestRightZeroPointI]
            zeroPointI = closestRightZeroPointI
        } else {
            // Use end of recording (or end of the data given) as candidate 0.
            keyPointCandidate0 = timeStamps.count - 1
        }
        // Find start of the closest zero plato to the right if the second extremum.
        if let zeroPlatoICopy = zeroPlatoI {
            let closestZeroPlatoI = getFirstIndexOf(list: zeroPlatos, startingAt: zeroPlatoICopy, where: {interval in return interval.0 > stops[stopI]})
            zeroPlatoI = closestZeroPlatoI
            if let closestZeroPlatoI, zeroPlatos[closestZeroPlatoI].0 > stops[stopI] {
                keyPointCandidate1 = zeroPlatos[closestZeroPlatoI].0
            }
        }
        // Select closest:
        if let keyPointCandidate1 {
            intervalStop = timeStamps[min(keyPointCandidate0, keyPointCandidate1)]
        } else {
            intervalStop = timeStamps[keyPointCandidate0]
        }
        // ---------------------------------------------------------------------
        
        startI += 1
        stopI += 1
        
        if intervalStart > endPhase0 || endPhase0 > endPhase1 || endPhase1 > intervalStop {
            debugPrint("Intervals are wrong")
            debugPrint("\(intervalStart), \(endPhase0), \(endPhase1), \(intervalStop)")
        }
        detectedIntervals.append(Interval(start: intervalStart, phase0End: endPhase0, phase1End: endPhase1, stop: intervalStop))
    }
    
    return detectedIntervals
}
