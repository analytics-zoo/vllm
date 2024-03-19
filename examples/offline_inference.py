from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

# Create an LLM.
# /home/wangruonan/llm-models/Mistral-7B-Instruct-v0.1/
# llm = LLM(model="/home/wangruonan/llm-models/Mistral-7B-v0.1/", device="xpu", enforce_eager=True, dtype="float16")
llm = LLM(model="/home/wangruonan/llm-models/Mistral-7B-Instruct-v0.1/", device="xpu", enforce_eager=True, dtype="float16")
# llm = LLM(model="/home/wangruonan/llm-models/Llama-2-7b-chat-hf/", device="xpu", enforce_eager=True, dtype="float16")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
