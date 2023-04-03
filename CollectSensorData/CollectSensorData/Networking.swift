import Foundation
import Network

enum TypeOfPackage: Int {
    case heartbeat = 0
    case start = 1
    case stop = 2
    case motionData = 3
}

protocol ConnectionListenerDelegate {
    func listenerStarted()
    mutating func connectionFound(connection: TCPConnection)
}

class ConnectionListener {
    private var delegate: ConnectionListenerDelegate?
    private let listener: NWListener?
    
    init(name: String, type: String) {
        self.listener = try? NWListener(using: .tcp)
        self.listener?.service = NWListener.Service(name: name, type: type)
    }
    
    func start(delegate: ConnectionListenerDelegate, queue: DispatchQueue) {
        self.delegate = delegate
        self.listener?.newConnectionLimit = 1
        self.listener?.newConnectionHandler = {connection in
            self.delegate?.connectionFound(connection: TCPConnection(connection: connection))
            self.stop()
        }
        self.listener?.stateUpdateHandler = {state in
            switch state {
            case .ready:
                self.delegate?.listenerStarted()
            case .failed(let err):
                print("Listener: Connection Error \(err)")
            case .cancelled:
                print("Listener: Connection cancelled")
            default:
                break
            }
        }
        self.listener?.start(queue: queue)
    }
    
    func stop() {
        self.listener?.cancel()
    }
}

protocol ConnectionBrowserDelegate {
    mutating func connectionFound(connection: TCPConnection)
}

class ConnectionBrowser {
    private var delegate: ConnectionBrowserDelegate?
    private let browser: NWBrowser
    
    init(type: String) {
        self.browser = NWBrowser(for: .bonjour(type: type, domain: nil), using: .tcp)
    }
    
    func start(delegate: ConnectionBrowserDelegate, queue: DispatchQueue) {
        self.delegate = delegate
        self.browser.stateUpdateHandler = { state in
            switch state {
            case .failed(let err):
                print("Browser Error: \(err)")
            default:
                break
            }
        }
        self.browser.browseResultsChangedHandler = { results, changes in
            results.forEach{ device in
                self.delegate?.connectionFound(connection: TCPConnection(to: device.endpoint))
                self.stop()
                return
            }
        }
    }
    
    func stop() {
        self.browser.cancel()
    }
    
}

protocol TCPConnectionDelegate {
    func connectionEstablished()
    func connectionError(error: Error)
    func dataReceived(data: Data)
}

class TCPConnection {
    private var delegate: TCPConnectionDelegate?
    private let connection: NWConnection
    
    init(connection: NWConnection) {
        self.connection = connection
    }
    
    init(to: NWEndpoint) {
        self.connection = NWConnection(to: to, using: .tcp)
    }
    
    func start(delegate: TCPConnectionDelegate, queue: DispatchQueue) {
        self.delegate = delegate
        self.connection.stateUpdateHandler = {state in
            switch state {
            case .ready:
                self.delegate?.connectionEstablished()
            case .failed(let err):
                print("Connection lost, Error \(err)")
                self.delegate?.connectionError(error: err)
            case .cancelled:
                print("Connection cancelled")
            default:
                debugPrint("Connection in State: \(state)")
            }
        }
        self.connection.start(queue: queue)
    }
    
    private func sendData(packageTypePrefix: Data, data: Data, completionHandler: (() -> Void)?) {
        debugPrint("Sending \(data.count) bytes")
        let sizePrefix = withUnsafeBytes(of: UInt32(data.count).bigEndian) { Data($0) }
        
        self.connection.batch {
            sendPackageTypePrefix(packageTypePrefix: packageTypePrefix, completionHandler: nil)
            self.connection.send(content: sizePrefix, completion: .contentProcessed({err in
                if let err = err {
                    print("Connection Error while sending sizePrefix: \(err)")
                    self.delegate?.connectionError(error: err)
                }
            }))
            self.connection.send(content: data, isComplete: true, completion: .contentProcessed({err in
                if let err = err {
                    print("Connection Error while sending data: \(err)")
                    self.delegate?.connectionError(error: err)
                } else {
                    completionHandler?()
                }
            }))
        }
    }
    
    private func sendPackageTypePrefix(packageTypePrefix: Data, completionHandler: (() -> Void)?) {
        debugPrint("Actually sending Prefix")
        self.connection.send(content: packageTypePrefix, completion: .contentProcessed({err in
            debugPrint("Entering standard completion method")
            if let err = err {
                print("Connection Error while sending packageTypePrefix: \(err)")
                self.delegate?.connectionError(error: err)
            } else {
                debugPrint("Reached Handler")
                completionHandler?()
                debugPrint("Now the handler should have been called")
            }
        }))
    }
    
    func send(packageType: TypeOfPackage, data: Data? = nil, completionHandler: (() -> Void)? = nil) {
        debugPrint("Entered send function")
        let packageTypePrefix = withUnsafeBytes(of: UInt8(packageType.rawValue).bigEndian) { Data($0) }
        switch packageType {
        case .motionData:
            guard let data = data else {
                debugPrint("Error sending motion Data: No motion data given")
                return
            }
            sendData(packageTypePrefix: packageTypePrefix, data: data, completionHandler: completionHandler)
        case .start, .stop, .heartbeat:
            debugPrint("Sending Prefix")
            sendPackageTypePrefix(packageTypePrefix: packageTypePrefix, completionHandler: completionHandler)
        }
    }
    
    func receive() {
        debugPrint("Receiving Data is Not Implemented\nStarting Dummy Receive")
        self.dummyReceive()
    }
    
    private func dummyReceive() {
        self.connection.receive(minimumIncompleteLength: 1, maximumLength: 65536, completion: {(data, _, isComplete, err) in
            if let data = data, !data.isEmpty
            {
                self.delegate?.dataReceived(data: data)
            }
            if isComplete
            {
                self.close()
            }
            else if let err = err
            {
                self.delegate?.connectionError(error: err)
            }
            else
            {
                self.dummyReceive()
            }
        })
    }
    
    func close() {
        self.connection.cancel()
    }
}
