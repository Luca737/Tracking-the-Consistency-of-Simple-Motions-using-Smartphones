//
//  Filter.swift
//  ConsistencyTracker
//
//  Created by Nicola Vidovic on 25.02.23.
//

import Foundation

func lowPass3rdOrder_2_5Hz(_ x: [Double]) -> [Double] {
    let b: [Double] = [0.00041655, 0.00124964, 0.00124964, 0.00041655]
    let a: [Double] = [1.0, -2.6861574, 2.41965511, -0.73016535]
    
    return lowPass3rdOrder(x, b, a)
}

func lowPass3rdOrder_1_2Hz(_ x: [Double]) -> [Double] {
    let b: [Double] = [4.97574358e-05, 1.49272307e-04, 1.49272307e-04, 4.97574358e-05]
    let a: [Double] = [1.0, -2.8492391, 2.70962913, -0.85999198]
    
    return lowPass3rdOrder(x, b, a)
}

func lowPass3rdOrder(_ x: [Double], _ b: [Double], _ a: [Double]) -> [Double] {
    // This filter introduces a phase shift of about 310ms, therefore the last 310ms are 'deleted'.
    // Butter Worth Coefficients calculated using scipy signal.butter with cutoff 2.5 and sampling rate of 100HZ
    var y: [Double] = Array(repeating: 0.0, count: x.count)
    
    for i in 3..<x.count {
        y[i] = b[0] * x[i] + b[1] * x[i-1] + b[2] * x[i-2] + b[3] * x[i-3] - a[1] * y[i-1] - a[2] * y[i-2] - a[3] * y[i-3]
    }
    
    return y
}

func lowPass2ndOrder2_5Hz(_ x: [Double]) -> [Double] {
    // This filter introduces a phase shift of about 100ms, therefore the last 100ms are 'deleted'.
    // Butter Worth Coefficients calculated using scipy signal.butter with cutoff 2.5 and sampling rate of 100HZ
    let b: [Double] = [0.00554272, 0.01108543, 0.00554272]
    let a: [Double] = [1.0, -1.77863178,  0.80080265]
    var y: [Double] = Array(repeating: 0.0, count: x.count)
    
    for i in 2..<x.count {
        y[i] = b[0] * x[i] + b[1] * x[i-1] + b[2] * x[i-2] - a[1] * y[i-1] - a[2] * y[i-2]
    }
    
    return y
}

func derivative(_ data: [Double]) -> [Double] {
    var out: [Double] = Array(repeating: 0.0, count: data.count)
    out[0] = (data[1] - data[0]) / (1/SAMPLE_FREQUENCY)
    out[data.count-1] = (data[data.count-1] - data[data.count-2]) / (1/SAMPLE_FREQUENCY)
    for i in 1..<data.count-1 {
        out[i] = (data[i+1] - data[i-1]) / (2/SAMPLE_FREQUENCY)
    }
    return out
}

// Due to potential copyright violations, I have removed the source code for PCA.
//func pca(_ data: [[Double]]) -> [[Double]] {
////    debugPrint("WARNING NOT IMPLEMENTED, returning dummy")
//    // This works like trash and may be used wrong altogether. It is also not clear in what structure the motion data should be passed in.
//    var out: [[Double]] = [[]]
//    let (success, output, _, _ , _, _) =  PCA.sharedInstance.dimensionReduction(data, dimension: 3)
//    if success {
//        out = output
//    } else {
//        out = data
//    }
//    return out
//}
