//
//  ContentView.swift
//  TestT5Encoder
//
//  Created by hanbitchan on 9/23/24.
//

import SwiftUI

struct ContentView: View {
  var body: some View {
      VStack {
        Button(action: generate) {
          Text("Click").font(.title)
        }.buttonStyle(.borderedProminent)
      }
      .padding()
  }

func generate() {
    do {
      let pipeline = try Pipeline(resourcesAt: Bundle.main.bundleURL)
        print("Click")
        pipeline.sample(prompt: "Test the attention mask")
    } catch {
        print("Error: Can't initiallize Pipeline")
    }
  }
}

#Preview {
    ContentView()
}
