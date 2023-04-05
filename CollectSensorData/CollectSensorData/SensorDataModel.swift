//  Created by Nicola Vidovic

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
    let startTime: FormattedDate
    var frames: [Frame]
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
        let frameStampString = formatTimeInterval(interval: frameStamp)
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

