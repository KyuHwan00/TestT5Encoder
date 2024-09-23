# TestT5Encoder

## Test condtions
- iOS Version: iOS 17
- Virtual Environment for python: torch, transformers, coremltools
```
pip install torch torchvision torchaudio
pip install transformers
pip install coreml
```

## How to use
1. For exporting T5, set the Virtual Environment
2. Export T5 model using with t5export.py 
3. Put the exported and converted t5 mlpackage into xcode project.
4. Run and Test 
