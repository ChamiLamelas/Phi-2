#!/usr/bin/env python3.8

import signal
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datetime import timedelta
import os

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

while True:
    prompt = input("Query? ")
    with torch.no_grad():
        ti = time.time()
        token_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
        )
        output = tokenizer.decode(output_ids[0][token_ids.size(1) :])
        tf = time.time() 

        print("=" * os.get_terminal_size()[1])
        print(output)
        print("=" * os.get_terminal_size()[1])
        print(f"response time:\t{timedelta(seconds=tf - ti)}")
        print("=" * os.get_terminal_size()[1])
