import torch
from transformers import T5EncoderModel, AutoTokenizer
import coremltools as ct

def convert_t5_to_coreml():
    # Load the T5-small model and tokenizer
    model_name = "google/t5-v1_1-base"
    
    model = T5EncoderModel.from_pretrained(model_name, return_dict=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the model to evaluation mode
    model.eval()
    # Prepare input for tracing
    prompt = "Test the attention mask"
    inputs = tokenizer(prompt,
                       padding= "max_length",
                       truncation=True,
                       return_attention_mask=True,
                       add_special_tokens=True,
                       return_tensors="pt",)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print("Input IDs Shape:", input_ids)
    print("Attention Mask Shape:", attention_mask)

    baseline_out = model(input_ids, attention_mask)[0]
    print("Baseline: ", baseline_out[0,0,:101])
    
    # Trace the model
    traced_model = torch.jit.trace(model, (input_ids, attention_mask))

    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input_ids", shape=input_ids.shape),
            ct.TensorType(name="attention_mask", shape=attention_mask.shape)
        ],
        outputs=[
            ct.TensorType(name="encoder_hidden_states")
        ],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17
    )

    # Save the Core ML model
    mlmodel.save("t5-base.mlpackage")
    print("T5-base model converted and saved as t5-base.mlpackage")

if __name__ == "__main__":
    convert_t5_to_coreml()
