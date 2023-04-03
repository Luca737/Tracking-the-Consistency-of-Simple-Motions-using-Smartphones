//
//  CustomButton.swift
//  CollectSensorData
//
//  Created by Nicola Vidovic on 16.12.22.
//

import SwiftUI

struct CustomButton: View {
    @Binding var parentActive: Bool
    var delegate: MotionDelegate?
    
    var label: String {
        parentActive ? "Deactivate" : "Activate"
    }
    
    var body: some View {
        Button {
            self.delegate?.buttonPress()
        } label: {
            Text(label)
                .foregroundColor(.white)
        }
        .frame(alignment: .center)
        .padding()
        .background(parentActive ? .red : .blue)
        .cornerRadius(10)
    }
}
