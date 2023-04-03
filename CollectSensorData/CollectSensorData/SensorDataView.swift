import SwiftUI
import Foundation

protocol MotionDelegate {
    func buttonPress()
    func handleMotionData(sensorData: SensorData)
}

struct SensorDataView: View, MotionDelegate {
    @State var sensorActive: Bool = false
    @State var sensorData = SensorData(timeStamp: NSDate() as Date, accelerationX: 0, accelerationY: 0, accelerationZ: 0, gravityX: 0, gravityY: 0, gravityZ: 0, rotationRateX: 0, rotationRateY: 0, rotationRateZ: 0)
    
    private let motionHandler = GetMotionData()
    private let connectionListener = ConnectionListener(name: "SensApp", type: "_sensApp._tcp")
    @State private var connection: TCPConnection?
    
    @State private var motionDataRecording: Frames?
    @State private var recordingSessionStartTime: Date?
    private var networkQueue: DispatchQueue = .main
    private let jsonEncoder: JSONEncoder = JSONEncoder()
    private let dateFormatter: DateFormatter
    
    @State private var state: String = "Initial State"
    
    init() {
        dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss."
    }
    
    var body: some View {
        VStack {
            Text("Acceleration - x: \(sensorData.accelerationX), y:\(sensorData.accelerationY), z: \(sensorData.accelerationZ)")
            Text("Gravity - x: \(sensorData.gravityX), y:\(sensorData.gravityY), z: \(sensorData.gravityZ)")
            Text("TimeStamp - \(sensorData.timeStamp)")
            CustomButton(parentActive: $sensorActive, delegate: self)
            Text(state)
        }
        .padding()
        .onAppear {
            self.connectionListener.start(delegate: self, queue: self.networkQueue)
            self.motionHandler.sendMotionDataDelegate = self
        }
    }
    
    func buttonPress() {
        self.sensorActive.toggle()
        if sensorActive {
            debugPrint("Trying to send start signal to Connection \(String(describing: self.connection))")
            self.connection?.send(packageType: TypeOfPackage.start, completionHandler: {
                DispatchQueue.main.async {
                    self.recordingSessionStartTime = Date()
                    self.motionDataRecording = Frames(startTime: formatDateAppendedMicroSecs(date: self.recordingSessionStartTime!, formatter: self.dateFormatter), frames: [])
                    self.motionHandler.startMotionCapture()
                }
            })
        } else {
            motionHandler.stopMotionCapture()
            self.connection?.send(packageType: TypeOfPackage.stop, completionHandler: {
                DispatchQueue.main.async {
                    let jsonMotionData = try! self.jsonEncoder.encode(self.motionDataRecording)
                    self.connection?.send(packageType: TypeOfPackage.motionData, data: jsonMotionData)
                }
            })
        }
    }
    
    func handleMotionData(sensorData: SensorData) {
        self.sensorData = sensorData
        let sensorDataPoint = SensorDataPoint(
            acceleration_X: sensorData.accelerationX,
            acceleration_Y: sensorData.accelerationY,
            acceleration_Z: sensorData.accelerationZ,
            gravity_X: sensorData.gravityX,
            gravity_Y: sensorData.gravityY,
            gravity_Z: sensorData.gravityZ,
            rotationRate_X: sensorData.rotationRateX,
            rotationRate_Y: sensorData.rotationRateY,
            rotationRate_Z: sensorData.rotationRateZ
        )
        let timeSinceSessionStart: Double = sensorData.timeStamp.timeIntervalSince(self.recordingSessionStartTime!)
        let frame = Frame(frameStamp: timeSinceSessionStart, frameAttributes: sensorDataPoint)
        self.motionDataRecording?.frames.append(frame)
    }
    
}

extension SensorDataView: ConnectionListenerDelegate {
    
    func listenerStarted() {
        debugPrint("Listener Started")
        self.state = "Listener Started"
    }
    
    func connectionFound(connection: TCPConnection) {
        debugPrint("Connection Found")
        self.connection = connection
        self.connection?.start(delegate: self, queue: self.networkQueue)
        self.state = "Connection Found"
    }
    
}

extension SensorDataView: TCPConnectionDelegate {
    
    func connectionEstablished() {
        debugPrint("Connection Established")
        self.state = "Connection Established"
    }
    
    func connectionError(error: Error) {
        debugPrint("Connection Lost")
        self.state = "Connection Lost"
        print(error)
    }
    
    func dataReceived(data: Data) {
        debugPrint("Data Received: But no method implemented to handle it")
    }
    
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        SensorDataView()
    }
}
