//
//  SensorData.swift
//  CollectSensorData
//
//  Created by Nicola Vidovic on 15.12.22.
//

import Foundation
import CoreMotion

struct SensorData: Codable {
    let timeStamp: Date
    let accelerationX: Double
    let accelerationY: Double
    let accelerationZ: Double
    let gravityX: Double
    let gravityY: Double
    let gravityZ: Double
    let rotationRateX: Double
    let rotationRateY: Double
    let rotationRateZ: Double
}

class GetMotionData {
    private let updateInterval: TimeInterval = 1/100
    private lazy var motionManager = CMMotionManager()
    
    var sendMotionDataDelegate: MotionDelegate?
    
    func startMotionCapture() {
        guard motionManager.isAccelerometerAvailable && motionManager.isGyroAvailable else {
            debugPrint("Accelerometer or Gyroscope not available")
            return
        }
        
        motionManager.deviceMotionUpdateInterval = updateInterval
        motionManager.startDeviceMotionUpdates(to: .main) { [self] (data, error) in
            guard let data = data, error == nil else {
                print("Caught error in motion")
                debugPrint(error as Any)
                return
            }
            let sensDat = SensorData(
                timeStamp: NSDate() as Date,
                accelerationX: data.userAcceleration.x,
                accelerationY: data.userAcceleration.y,
                accelerationZ: data.userAcceleration.z,
                gravityX: data.gravity.x,
                gravityY: data.gravity.y,
                gravityZ: data.gravity.z,
                rotationRateX: data.rotationRate.x,
                rotationRateY: data.rotationRate.y,
                rotationRateZ: data.rotationRate.z
            )
            self.sendMotionDataDelegate?.handleMotionData(sensorData: sensDat)
        }
    }
    
    func stopMotionCapture() {
        if motionManager.isDeviceMotionActive {
            motionManager.stopDeviceMotionUpdates()
        } else {
            debugPrint("Motion Engine not active")
        }
    }
}
