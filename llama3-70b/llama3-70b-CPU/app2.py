from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def gerar_texto_pirata(texto_inicial):

    """
    Generate pirate-themed text based on the initial input.

    Args:
    initial_text (str): The initial text to generate a response from.

    Returns:
    str: The generated pirate-themed text.

    """
    model_id = "/app/Meta-Llama-3-70B-Instruct"
 
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": texto_inicial}, 
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]
    return(tokenizer.decode(response, skip_special_tokens=True))

if __name__ == "__main__":
    print("\n\n----GERADOR DE TEXTO - PIRATA----")
    texto_inicial = input("\nInsira o texto inicial: ")
    print("\n\n\n")
    texto_gerado = gerar_texto_pirata(texto_inicial)
    print("\n\n\nTexto Gerado:\n")
    print(texto_gerado)



