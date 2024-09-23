//
//  Pipeline.swift
//  TestT5Encoder
//
//  Created by hanbitchan on 9/23/24.
//

import Foundation
import CoreML
import Accelerate
import Tokenizers
import Hub

struct ResourceURLs {
    public let configT5URL: URL
    public let dataT5URL: URL
    public let modelURL: URL
    // To do: change the file format to mlpackage
    public init(resourcesAt baseURL: URL) {
        configT5URL = baseURL.appending(path: "tokenizer_config.json")
        dataT5URL = baseURL.appending(path: "tokenizer.json")
        modelURL = baseURL.appending(path: "t5-base.mlmodelc")
    }
}

public struct Pipeline {
  // need to initialize the required models. ex) stdit, vae and so on.
  let textEncoderT5: TextEncoderT5?
  
  init(resourcesAt baseURL: URL, configuration config: MLModelConfiguration = .init(), reduceMemory: Bool = false) throws {
    
    let urls = ResourceURLs(resourcesAt: baseURL)
    
    // initialize Models for Text Encoding
    if FileManager.default.fileExists(atPath: urls.configT5URL.path),
       FileManager.default.fileExists(atPath: urls.dataT5URL.path)
    {
      let config = MLModelConfiguration()
      config.computeUnits = .all
      let tokenizerT5 = try PreTrainedTokenizer(tokenizerConfig: Config(fileURL: urls.configT5URL), tokenizerData: Config(fileURL: urls.dataT5URL))
      textEncoderT5 = TextEncoderT5(tokenizer: tokenizerT5, modelAt: urls.modelURL, configuration: config)
    } else {
      textEncoderT5 = nil
    }
  }
  func sample(prompt: String) {
    // To do: make the sample process
    do {
      guard let _ = try textEncoderT5?.encode(prompt) else {
        print("Error: Can't Encoding.")
        return
      }
    } catch {
      print("Error: Can't make sample.")
    }
  }
}
