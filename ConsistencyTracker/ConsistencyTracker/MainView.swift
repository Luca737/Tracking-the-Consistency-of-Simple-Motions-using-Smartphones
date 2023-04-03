//
//  ContentView.swift
//  ConsistencyTracker
//
//  Created by Nicola Vidovic on 24.02.23.
//

import SwiftUI

struct MainView: View {
    @State private var useVideoRecording: Bool = false
    @State private var networkingEnabled: Bool = false
    @State private var isConnected: Bool = false
    @State private var isRecoding: Bool = false
    @State private var isShowingStatistics: Bool = false
    @State private var deactivateAllUserInput: Bool = false
    
    private var connectionListener: ConnectionListener = ConnectionListener(name: "CTApp", type: "_ctApp._tcp")
    @State private var connection: TCPConnection?
    private let jsonEncoder: JSONEncoder = JSONEncoder()
    
    private var sensorDataPipeline = SensorDataPipeline()
    
    var body: some View {
        VStack {
            ZStack {
                HStack {
                    Button(action: toggleConnection) {
                        Text(isConnected ? "Disconnect" : networkingEnabled ? "Searching" : "Connect")
                    }
                }
                HStack {
                    Spacer()
                    Toggle("", isOn: $useVideoRecording)
                        .toggleStyle(CheckboxStyle())
                        .opacity(deactivateAllUserInput || isRecoding ? 0.5 : 1)
                    Spacer()
                        .frame(width: 70)
                }
            }
            .disabled(deactivateAllUserInput || isRecoding)
            Spacer()
                .frame(height: 200)
            Button(isRecoding ? "Stop" : "Start", action: toggleRecording)
                .tint(isRecoding ? .red : .blue)
                .disabled(deactivateAllUserInput)
            Spacer()
                .frame(height: 200)
            HStack {
                Button("Show Statistics", action: showStatistics)
                Button("Send", action: sendData)
            }
            .disabled(deactivateAllUserInput || isRecoding)
            
        }
        .padding()
        .buttonStyle(.borderedProminent)
        .onAppear {
            sensorDataPipeline.delegate = self
        }
        .popover(isPresented: $isShowingStatistics) {
            StatisticsCardContainerView(allStatistics: sensorDataPipeline.analysisResults ?? AllStatistics())
        }
        .onChange(of: useVideoRecording) { enableVideoModule in
            if enableVideoModule {
                connection?.send(packageType: .enableVideoCaptureModule)
            }
        }
    }
    
    func toggleConnection() {
        if isConnected || networkingEnabled {
            self.connectionListener.stop()
            self.connection?.close()
            self.isConnected = false
            UIApplication.shared.isIdleTimerDisabled = false
        } else {
            self.connectionListener.start(delegate: self, queue: .main)
            UIApplication.shared.isIdleTimerDisabled = true
        }
        networkingEnabled.toggle()
    }
    
    func toggleRecording() {
        if isRecoding {
            deactivateAllUserInput = true
    
            sensorDataPipeline.stopRecording()
            connection?.send(packageType: .stop)
            sensorDataPipeline.analyze()
            
            deactivateAllUserInput = false
        } else {
            if useVideoRecording {
                connection?.send(packageType: .start)
            }
            sensorDataPipeline.startRecording()
        }
        // isRecording toggled in delegate
    }
    
    func sendData() {
        guard isConnected, sensorDataPipeline.motionFrames.frames.count > 0, sensorDataPipeline.allIntervals.count > 0 else {
            return
        }
        debugPrint("Making Data ready for sending")
        var toSend: [Data] = []
        toSend.append(try? self.jsonEncoder.encode(self.sensorDataPipeline.motionFrames))
        // Send filtered data as well for debugging.
//        var filteredFrames: FramesA = FramesA(startTime: self.sensorDataPipeline.motionFrames.startTime, frames: [])
//        let fD = self.sensorDataPipeline.filteredData
//        for i in 0..<fD.rotRate[0].count {
//            filteredFrames.frames.append(FrameA(frameStamp: self.sensorDataPipeline.motionFrames.frames[i].frameStamp, frameAttributes: SensorDataPointA(Aacceleration_X: fD.linAcc[0][i], Aacceleration_Y: fD.linAcc[1][i], Aacceleration_Z: fD.linAcc[2][i], Agravity_X: fD.grav[0][i], Agravity_Y: fD.grav[1][i], Agravity_Z: fD.grav[2][i], ArotationRate_X: fD.rotRate[0][i], ArotationRate_Y: fD.rotRate[1][i], ArotationRate_Z: fD.rotRate[2][i])))
//        }
//        toSend.append(try? self.jsonEncoder.encode(filteredFrames))
        // -----------------------------
        for annotations in self.sensorDataPipeline.allIntervals {
            toSend.append(try? self.jsonEncoder.encode(convertAnnotationsToVIT(intervals: annotations)))
        }
        // ------
        debugPrint("Sending Data")
        self.connection?.send(packageType: .motionData, data: toSend, completionHandler: {
            debugPrint("Sending Completion Handler Called")
        })
    }
}

extension MainView: ConnectionListenerDelegate {
    func listenerStarted() {
        debugPrint("Listener Started")
    }
    
    func connectionFound(connection: TCPConnection) {
        debugPrint("Connection Found")
        self.connection = connection
        self.connection?.start(delegate: self, queue: .main)
    }
}

extension MainView: TCPConnectionDelegate {
    func connectionEstablished() {
        debugPrint("Connection Established")
        self.isConnected = true
    }
    
    func connectionError(error: Error) {
        debugPrint("Connection Error")
        self.connection?.close()
    }
    
    func dataReceived(data: Data) {
        debugPrint("Data Received: Discarding")
    }
}

extension MainView: SensorDataPipelineDelegate {
    func analysisFinished() {
        showStatistics()
    }
    
    func recordingHasStarted() {
        self.isRecoding = true
    }
    
    func recordingHasStopped() {
        self.isRecoding = false
    }
    
    func showStatistics() {
        isShowingStatistics = true
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        MainView()
    }
}
