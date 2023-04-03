//
//  ExternalCode.swift
//  ConsistencyTracker
//
//  Created by Nicola Vidovic on 27.02.23.
//

import Foundation
import SwiftUI

struct CheckboxStyle: ToggleStyle {
    /* Source: https://stackoverflow.com/a/65895802 by Nate Bird */
    func makeBody(configuration: Self.Configuration) -> some View {
        return HStack {
            Image(systemName: configuration.isOn ? "video.fill" : "video.slash")
//                .resizable()
                .frame(width: 30, height: 16)
                .foregroundColor(configuration.isOn ? .red : .gray)
                .font(.system(size: 20, weight: .regular, design: .default))
                configuration.label
        }
        .onTapGesture { configuration.isOn.toggle() }
    }
}
