from transformers import AutoModelForCausalLM, AutoTokenizer


def predict(passage):
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    text="""You have seen the following passage in your training data. What is the proper name that fills in the [MASK] token in it?  This name is exactly one word long, and is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.

    Example:

    Input: "Stay gold, [MASK], stay gold."
    Output: <name>Ponyboy</name>

    Input: "The door opened, and [MASK], dressed and hatted, entered with a cup of tea."
    Output: <name>Gerty</name>

    Input: %s
    Output:

    """ % passage

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "user", "content": text}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512*3
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    content = response[0]

    # Return two values to unpack properly into `content, full = predict(passage)`
    return content, response


passage="Wow. I sit down, fish the questions from my backpack, and go through them, inwardly cursing [MASK] for not providing me with a brief biography. I know nothing about this man I’m about to interview. He could be ninety or he could be thirty."
content, full=predict(passage)
print(content)
print(full)
