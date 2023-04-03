import Foundation

typealias FormattedElapsedTime = String
typealias FormattedDate = String

func formatTimeInterval(interval: TimeInterval) -> FormattedElapsedTime {
    let fractionalPart = String(String(String(interval).split(separator: ".")[1]).prefix(6))
    let nonFractionalPart = Int(interval)
    let seconds = nonFractionalPart % 60
    let minutes = (nonFractionalPart / 60) % 60
    let hours = nonFractionalPart / (60 * 60)
    
    return String(format: "%02u:%02u:%02u.", hours, minutes, seconds, fractionalPart).appending(fractionalPart) as FormattedElapsedTime
}

func formatDateAppendedMicroSecs(date: Date, formatter: DateFormatter) -> FormattedDate {
    let fractionalPart = String(String(String(date.timeIntervalSinceReferenceDate).split(separator: ".")[1]).suffix(6))
    return formatter.string(from: date).appending(fractionalPart)
}

struct Frames: Encodable {
    let startTime: Date
    var frames: [Frame]
    
    enum CodingKeys: String, CodingKey {
        case startTime
        case applicationName
        case frames
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss."
        let startTimeString = formatDateAppendedMicroSecs(date: startTime, formatter: dateFormatter)
        try container.encode(startTimeString, forKey: .startTime)
        dateFormatter.dateFormat = "dd.MM_HH:mm:ss.SSS"
        try container.encode(dateFormatter.string(from: Date()), forKey: .applicationName)
        try container.encode(frames, forKey: .frames)
    }
}

struct Frame: Encodable {
    let frameStamp: Double
    let frameAttributes: SensorDataPoint

    enum CodingKeys: String, CodingKey {
        case frameStamp
        case frameAttributes
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        let frameStampString: FormattedElapsedTime = formatTimeInterval(interval: frameStamp)
        try container.encode(frameStampString, forKey: .frameStamp)
        try container.encode(frameAttributes, forKey: .frameAttributes)
    }
}

struct SensorDataPoint: Encodable {
    let acceleration_X: Double
    let acceleration_Y: Double
    let acceleration_Z: Double
    let gravity_X: Double
    let gravity_Y: Double
    let gravity_Z: Double
    let rotationRate_X: Double
    let rotationRate_Y: Double
    let rotationRate_Z: Double
}

struct FramesA: Encodable {
    let startTime: Date
    var frames: [FrameA]
    
    enum CodingKeys: String, CodingKey {
        case startTime
        case applicationName
        case frames
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "HH:mm:ss."
        let startTimeString = formatDateAppendedMicroSecs(date: startTime, formatter: dateFormatter)
        try container.encode(startTimeString, forKey: .startTime)
        dateFormatter.dateFormat = "dd.MM_HH:mm:ss.SSS"
        try container.encode(dateFormatter.string(from: Date()), forKey: .applicationName)
        try container.encode(frames, forKey: .frames)
    }
}

struct FrameA: Encodable {
    let frameStamp: Double
    let frameAttributes: SensorDataPointA

    enum CodingKeys: String, CodingKey {
        case frameStamp
        case frameAttributes
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        let frameStampString: FormattedElapsedTime = formatTimeInterval(interval: frameStamp)
        try container.encode(frameStampString, forKey: .frameStamp)
        try container.encode(frameAttributes, forKey: .frameAttributes)
    }
}

struct SensorDataPointA: Encodable {
    let Aacceleration_X: Double
    let Aacceleration_Y: Double
    let Aacceleration_Z: Double
    let Agravity_X: Double
    let Agravity_Y: Double
    let Agravity_Z: Double
    let ArotationRate_X: Double
    let ArotationRate_Y: Double
    let ArotationRate_Z: Double
}
