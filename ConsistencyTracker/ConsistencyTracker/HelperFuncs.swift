//
//  HelperFuncs.swift
//  ConsistencyAnalysis
//
//  Created by Nicola Vidovic on 21.02.23.
//

import Foundation


func calcExtremaWithBaseline(data: [Double], minimaBaseline: Double, maximaBaseline: Double) -> ([Int], [Int]) {
    var minima: [Int] = []
    var maxima: [Int] = []
    var min_i: Int = 0
    var min_val: Double = 0
    var max_i: Int = 0
    var max_val: Double = 0
    for (i, val) in data.enumerated() {
        if val <= minimaBaseline {
            if val < min_val{
                min_val = val
                min_i = i
            }
        } else if min_val < 0 {
            minima.append(min_i)
            min_val = 0
        }
        if val >= maximaBaseline {
            if val > max_val {
                max_val = val
                max_i = i
            }
        } else if max_val > 0 {
            maxima.append(max_i)
            max_val = 0
        }
    }
    
    return (minima, maxima)
}

func slope(index: Int, data: [Double]) -> Double {
    /*
     Calculate the slope of index in array
     Slope between index-1 and index+1 -> no edges allowed.
     Assumes perfect timing in sample frequency -> Simpler and in experience smoother curves during hick ups
     */
    return (data[index+1] - data[index-1]) / (2/SAMPLE_FREQUENCY)
}

func getFirstIndexOf<T>(list: [T], startingAt startI: Int, offset: Int = 0, where condition: (T) -> Bool) -> Int? {
    for i in startI..<list.count {
        if condition(list[i]) {
            return i + offset
        }
    }
    return nil
}

func calcZeroPointsPlatos(data: [Double], time_stamps: [Double], zeroPlatoValRadius: Double, zeroPlatoSlopeRadius: Double, minPlatoLengthSecs: Double, allowedSkipLengthBetweenPlatosSecs: Double) -> (zeroPoints: [Int], zeroPlatos: [(Int, Int)]) {
    /*
     Features:
     - Calc zero points (sign changed compared to the prior one)
     - Zero plato ranges -> [Int, Int)
     */
    
    let allowedSkipLengthBetweenPlatosInNumPoints: Int = Int(round(allowedSkipLengthBetweenPlatosSecs * SAMPLE_FREQUENCY))
    let minPlatoLengthInNumPoints: Int = Int(round(minPlatoLengthSecs * SAMPLE_FREQUENCY))
    
    var zeroPlatos: [(Int, Int)] = []
    var zeroPoints: [Int] = []
    var lastSign: Bool = data[1] >= 0
    var platoStart: Int? = nil
    for (i, val) in data[1..<data.count-1].enumerated() {
        let valIndex = i + 1
        let sign = val >= 0
        if sign != lastSign {
            lastSign = sign
            zeroPoints.append(i)
        }
        let isValidZeroPlatoPoint = (abs(val) <= zeroPlatoValRadius) && (abs(slope(index: valIndex, data: data)) <= zeroPlatoSlopeRadius)
        if platoStart != nil && !isValidZeroPlatoPoint {
            zeroPlatos.append((platoStart!, valIndex))
            platoStart = nil
        } else if platoStart == nil && isValidZeroPlatoPoint {
            platoStart = valIndex
        }
    }
    if platoStart != nil {
        zeroPlatos.append((platoStart!, data.count))
    }
    
    // Connect platos, which are close.
    for i in stride(from: zeroPlatos.count-1, through: 1, by: -1) {
        if zeroPlatos[i].0 - zeroPlatos[i-1].1 <= allowedSkipLengthBetweenPlatosInNumPoints {
            zeroPlatos[i-1] = (zeroPlatos[i-1].0, zeroPlatos[i].1)
            zeroPlatos.remove(at: i)
        }
    }
    
    // Remove platos with non significant length.
    for i in stride(from: zeroPlatos.count-1, through: 1, by: -1) {
        if zeroPlatos[i].1 - zeroPlatos[i].0 < minPlatoLengthInNumPoints {
            zeroPlatos.remove(at: i)
        }
    }
    
    return (zeroPoints, zeroPlatos)
}

func autoChooseBestAxis(axes: [[Double]]) -> Int {
    var bestI: Int = 0
    var bestVal: Double = -1
    for (i, axis) in axes.enumerated() {
        let val = axis.reduce(0) { partialResult, x in
            return partialResult + abs(x)
        }
        if val > bestVal {
            bestVal = val
            bestI = i
        }
    }
    return bestI
}

func isMotionDataSorted(frames: [Frame]) -> Bool {
    for i in 0..<frames.count-1 {
        if frames[i].frameStamp > frames[i+1].frameStamp {
            return false
        }
    }
    return true
}
