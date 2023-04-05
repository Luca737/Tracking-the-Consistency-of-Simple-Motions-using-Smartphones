//  Created by Nicola Vidovic

import Foundation
import CoreMotion

struct SensorData {
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

protocol SensorDataRecorderDelegate {
    func handleMotionData(sensorData: SensorData)
}

class SensorDataRecorder {
    private let updateInterval: TimeInterval = 1 / SAMPLE_FREQUENCY
    private lazy var motionManager = CMMotionManager()
    
    var sendMotionDataDelegate: SensorDataRecorderDelegate?
    
    var isRecodingActive: Bool {
        motionManager.isDeviceMotionActive
    }
    
    func startMotionCapture() {
        guard motionManager.isAccelerometerAvailable && motionManager.isGyroAvailable else {
            debugPrint("Accelerometer or Gyroscope not available")
            return
        }
        guard sendMotionDataDelegate != nil else {
            debugPrint("No Delegate to handle the motion data was given")
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
