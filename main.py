
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", type=str, default = None, help="model path")
parser.add_argument("-server_name", type=str, default = "101.6.68.214", help="ip")
parser.add_argument("-server_port", type=int, default = 7860, help="port")
args = parser.parse_args()
model = AutoModelForCausalLM.from_pretrained(args.m).cuda()
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.m)
class AsymmetricSteganography:
    
    def __init__(self, secure_parameter, tokenizer, func_type):
        self.secure_parameter = secure_parameter
        self.tokenizer = tokenizer
        self.length = int(np.ceil(np.log2(tokenizer.vocab_size)))
        self.func_type = func_type
        
    def sampling_function(self, r, i):
        results = [np.cos(np.pi * r), np.sign(0.5 - r), np.log2(2-r) - 0.5573]
        return results[i]
    
    def int_to_bin(self, number):
        length = self.length
        binary_string = bin(number)[2:]
        return binary_string.zfill(length)

    def bin_to_int(self, binary_string):
        return int(binary_string, 2)
    
    def bits_to_text(self, output_bits, start_index, end_index):
        bit_length = self.length  
        if end_index > len(output_bits) or start_index < 0:
            raise ValueError("The provided indices are out of bounds.")
        if (end_index - start_index) % bit_length != 0:
            raise ValueError("The number of bits between start and end indices is not a multiple of the bit length per token.")
        selected_bits = output_bits[start_index:end_index]
        tokens = [self.bin_to_int(''.join(map(str, selected_bits[i:i+bit_length]))) for i in range(0, len(selected_bits), bit_length)]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return text
    
    def encode(self, prompt, model, secret_bits, seed = "12345", max_new_tokens = 1000, temperature = 1.0):

        segments = []
        step_probs = []
        segment = ""
        messages = [{"role": "user", "content": prompt},]
        H = 0
        prompt_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            past_key_values = None
            x = prompt_ids
            output_ids = prompt_ids
            current_bit = secret_bits[0]
            sum_v = 0
            cnt_v = 0
            bit_cnt = 0
            threshold = 2 ** (-self.secure_parameter)
            for token_cnt in tqdm(range(max_new_tokens)):
                output = model(x, past_key_values= past_key_values)
                logits = output.logits[0, -1, :]
                past_key_values = output.past_key_values
                probs = torch.softmax(logits/temperature, dim = -1)
                group = [0,2**self.length]
                split = 2**(self.length - 1)
                for _ in range(self.length):
                    random.seed(seed+str(cnt_v))
                    random_number = random.random()
                    # print(probs[group[0]:split].sum().item())
                    if current_bit == "0":
                        if random_number <= probs[group[0]:split].sum() / (probs[group[0]:group[1]].sum()):
                            next_bit = current_bit
                            sum_v += self.sampling_function(random_number, self.func_type)
                            H += (-np.log(probs[group[0]:split].sum().item() / (probs[group[0]:group[1]].sum().item())))
                            group[1] = split
                            # step_probs.append(-np.log(probs[group[0]:split].sum().item() / (probs[group[0]:group[1]].sum().item())))
                            
                        else:
                            next_bit = str(1 - int(current_bit))
                            sum_v += self.sampling_function(1 - random_number, self.func_type)
                            H += (-np.log(probs[split:group[1]].sum().item() / (probs[group[0]:group[1]].sum().item())))
                            group[0] = split
                            
                    else:
                        if random_number <= probs[split:group[1]].sum() / (probs[group[0]:group[1]].sum()):
                            next_bit = "1"
                            sum_v += self.sampling_function(1 - random_number, self.func_type)
                            H += (-np.log(probs[split:group[1]].sum().item() / (probs[group[0]:group[1]].sum().item())))
                            group[0] = split
                            
                        else:
                            next_bit = "0"
                            sum_v += self.sampling_function(random_number, self.func_type)
                            H += (-np.log(probs[group[0]:split].sum().item() / (probs[group[0]:group[1]].sum().item())))
                            group[1] = split
                            
                        
                    cnt_v += 1
                    split = int(np.ceil((group[0]+group[1])/2))
                    delta = sum_v / cnt_v
                    crit = np.exp(-2*delta**2 * cnt_v)
                    # print(crit,threshold)
                    segment += next_bit
                    step_probs.append(H)
                    if crit <= threshold:
                        if ((current_bit == "1" and delta >= 0) or (current_bit == "0"  and delta <= 0)):
                            print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                        if delta * (0.5 - int(current_bit)) >= 0:
                                seed = seed + str("a") 
                                print(seed)
                                print(segment)
                                sum_v = 0 
                                cnt_v = 0
                                bit_cnt += 1
                                current_bit = secret_bits[bit_cnt]
                                segments.append(segment)
                                segment = ""
                next_token = torch.LongTensor([group[0]]).view(1,1).cuda()
                # print(probs[next_token])
                
                output_ids = torch.cat((output_ids,next_token),dim=1)
                x = next_token
        # end_time = time.time()
        return output_ids.tolist()[0][len(prompt_ids[0]):], secret_bits[:bit_cnt], segments, step_probs
    # end_time - start_time
    
    def regular_decode(self, output_ids, seed = "12345"):
        output_bits = "".join([self.int_to_bin(j) for j in output_ids])
        string_per_bit = []
        secret_bits = ""
        sum_v = 0
        cnt_v = 0
        threshold = 2 ** (-self.secure_parameter)
        string = ""
        for bit in output_bits:
            random.seed(seed+str(cnt_v))
            random_number = random.random()
            # random_coin = random.choice([0,1])
            # if random_coin == 1:
            #     random_number = 1 - random_number
            sum_v += (int(bit))*self.sampling_function(1 - random_number, self.func_type) + (1-int(bit))*self.sampling_function(random_number, self.func_type)
            cnt_v += 1
            mu = cnt_v
            delta = sum_v / mu
            crit = np.exp(-2*delta**2 * cnt_v)
            string += bit
            # print(delta,crit)
            if crit <= threshold and delta >= 0:
                secret_bits += "0"
                sum_v = 0
                cnt_v = 0
                seed = seed + str("a")
                string_per_bit.append(string)
                string = ""
                continue
            if crit <= threshold and delta <= 0:
                secret_bits += "1"
                sum_v = 0
                cnt_v = 0
                seed = seed + str("a")
                string_per_bit.append(string)
                string = ""
                continue
            
        # print(secret_bits)
        return secret_bits, string_per_bit
    
    def robust_decode_1_bit(self, output_bits, r):
        threshold = 2**(-self.secure_parameter)
        output_bits = np.array(output_bits, dtype=int)
        r = np.array(r, dtype=float)
        cos_terms = (1 - output_bits)* (self.sampling_function(r,self.func_type)) + output_bits * (self.sampling_function(1- r,self.func_type))
        cumsum_v = np.cumsum(cos_terms)
        idx = np.arange(len(output_bits))
        mu = idx + 1 
        delta = cumsum_v / mu

        crit = np.exp(-2*delta**2 * mu)

        valid_indices = (crit <= threshold)
        if valid_indices.any():
            index = valid_indices.argmax()
            if delta[index] <= 0:
                secret_bits = "1"
            else:
                secret_bits = "0"
            return secret_bits, index + 1  
        
        return "", len(output_bits)  
    
    def robust_decode(self, output_ids, seed = "12345"):
        output_bits = "".join([self.int_to_bin(j) for j in output_ids])
        output_bits = [int(j) for j in output_bits]
        from multiprocessing import Pool
        from tqdm import tqdm
        import os
        num_cores = 40
        secret_bits = ""
        bit_num = 0
        bit_cnt = 0
        ret = []
        while bit_num < len(output_bits):
            r = []
            for k in range(len(output_bits)):
                random.seed(seed + str(k))
                random_number = random.random()
                random_coin = random.choice([0,1])
                if random_coin == 1:
                    random_number = 1 - random_number
                r.append(random_number)
            with Pool(processes=num_cores) as pool:
                results = []
                for cnt in tqdm(range(bit_num, min( bit_num + 1000, len(output_bits)))):
                    result = pool.apply_async(self.robust_decode_1_bit, args=(output_bits[cnt:], r[:len(r)-cnt]))
                    results.append(result)
                secret_bit = ""
                for result in tqdm(results):
                    secret_bit, cnt_v = result.get()
                    if secret_bit!= "":
                        secret_bits += secret_bit
                        ret.append([secret_bit, self.bits_to_text(output_bits, bit_num // self.length * self.length, (bit_num + cnt_v) // self.length * self.length)])
                        bit_num += cnt_v
                        print(bit_num)
                        bit_cnt += 1
                        pool.close()
                        seed = seed + str("a")
                        break
                if secret_bit == "":
                    bit_num += min( bit_num + 1000, len(output_bits)) - bit_num
                    print(bit_num)
        return ret


import gradio as gr

states = {"output_ids": []}
# seed = ""
def gen_text(prompt, secret_bits, seed, max_new_tokens, temperature, secure_parameter, func_type):
    stega = AsymmetricSteganography(secure_parameter, tokenizer, func_type)
    output_ids, embedded_bits, segments, step_probs = stega.encode(prompt, model, secret_bits, seed = seed, max_new_tokens=max_new_tokens, temperature= temperature)
    states["output_ids"] = output_ids
    
    return tokenizer.decode(output_ids), embedded_bits

def extract_text(seed, secure_parameter, func_type, output_text):
    stega = AsymmetricSteganography(secure_parameter, tokenizer, func_type)

    print(output_text)
    output_ids = states["output_ids"]
    
    secret_bits, string_per_bit = stega.regular_decode(output_ids, seed)
    index = 0
    output_bits = "".join([stega.int_to_bin(j) for j in output_ids])
    segments = []
    cnt = 0
    highlighted_text = []
    
    for i in string_per_bit:
        if secret_bits[cnt] == "0":
            segments.append(stega.bits_to_text(output_bits, index // stega.length * stega.length, (index + len(i)) // stega.length * stega.length))
            highlighted_text += [(j, "0") for j in segments[-1].replace("\n", "\\n\n")]
        else:
            segments.append(stega.bits_to_text(output_bits, index // stega.length * stega.length, (index + len(i)) // stega.length * stega.length))
            highlighted_text += [(j, "1") for j in segments[-1].replace("\n", "\\n\n")]
        index += len(i)
        cnt += 1
        highlighted_text.append(("|", "split"))
    return highlighted_text, secret_bits

def robust_extract_text(seed, secure_parameter, func_type, output_text):
    output_ids = tokenizer.encode(output_text)
    with open("ids.json", "w") as f:
        json.dump(output_ids, f)
        
    os.system(f"python fast_decode.py -secure_param={secure_parameter} -func_type={func_type} -seed={seed}")
    with open("ret.json", "r") as f:
        ret_list = json.load(f)
        
    # ret_list = stega.robust_decode(output_ids, seed)
    secret_bits = ""
    highlighted_text = []
    for i in ret_list:
        if i[0] == "0":
            highlighted_text += [(j, "0") for j in i[1].replace("\n", "\\n\n")]
        else:
            highlighted_text += [(j, "1") for j in i[1].replace("\n", "\\n\n")]
        # index += len(i)
        # cnt += 1
        highlighted_text.append(("|", "split"))
        # secret_bits.append(i[0])
        secret_bits += i[0]
    return highlighted_text, secret_bits

import gradio as gr

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Let Watermarks Speak: A Robust and Unforgeable Watermark for Language Models")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="What is watermarking?")
            secret_bits = gr.Textbox(label="Watermark signals (bits)", value="".join([str(i) for i in np.random.randint(0,2,100).tolist()]))
            seed = gr.Textbox(label="Seed", value="12345")
            max_new_tokens = gr.Slider(label="Max New Tokens", minimum=1, maximum=1000, step=1, value=100)
            output_text = gr.Textbox(label="Watermarked Text")
            used_secret_bits = gr.Textbox(label="Embedded Watermark signals")
            submit_button = gr.Button("Generate")

        with gr.Column():
            
            
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, step=0.01, value=1.0)
            secure_parameter = gr.Slider(label="Security Parameter", minimum=10, maximum=64, step=1, value=32)
            func_type = gr.Slider(label="Type of Function \n 0 = cos(\pi x), 1 = sign(x-0.5)", minimum=0, maximum=1, step=1, value=1)

            
            generated_text = gr.HighlightedText(
                label="Decoding the Text",
                combine_adjacent=True,
                show_legend=True,
                color_map={"0": "red", "1": "blue", "split": "purple"}
            )
            extracted_secret_bits = gr.Textbox(label="Extracted Watermark Signals")
            extract_button = gr.Button("Regularly Extract")
            robust_extract_button = gr.Button("Robustly Extract")

    submit_button.click(
        fn=gen_text,
        inputs=[prompt, secret_bits, seed, max_new_tokens, temperature, secure_parameter, func_type],
        outputs=[output_text, used_secret_bits]
    )
    extract_button.click(
        fn=extract_text,
        inputs=[seed, secure_parameter, func_type, output_text],
        outputs=[generated_text, extracted_secret_bits]
    )

demo.launch(server_name=args.server_name,server_port=args.server_port)

