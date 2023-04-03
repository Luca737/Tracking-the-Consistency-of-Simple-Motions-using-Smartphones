//
//  BaseExtensions.swift
//  ConsistencyTracker
//
//  Created by Nicola Vidovic on 18.03.23.
//

import Foundation

extension Array {
    public mutating func append(_ newElement: Element?) {
        if let element = newElement {
            self.append(element)
        }
    }
}
