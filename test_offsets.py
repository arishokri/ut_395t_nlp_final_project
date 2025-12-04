from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "./trained_models/emrqa-with-cartography/", use_fast=True
)

question = "What is the answer?"
context = "The answer is 42 and it is correct."
answer_start = 14  # Position of '42' in context
answer_text = "42"

inputs = tokenizer(question, context, return_offsets_mapping=True, return_tensors="pt")

print("Question:", question)
print("Context:", context)
print("Answer:", answer_text, "at position", answer_start, "in context")
print()

for i in range(min(25, len(inputs["input_ids"][0]))):
    offset = inputs["offset_mapping"][0][i]
    token_id = inputs["input_ids"][0][i]
    type_id = inputs["token_type_ids"][0][i]
    token = tokenizer.decode([token_id])
    print(
        f"Token {i:2d}: {token!r:15} offset={str(tuple(offset.tolist())):12} type_id={type_id}"
    )
