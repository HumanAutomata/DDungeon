"""
Adapted from the Ctrl-G tutorial notebook: https://github.com/joshuacnf/Ctrl-G/blob/main/tutorial_ctrlg.ipynb
"""

import os
import time
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set your cuda device
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import ctrlg
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

# load the pretrained base_model and hmm_model; see README.md for a complete list of
# released checkpoints. note that the hmm_model and base_model must share the same
# vocabulary of tokens: i.e., one cannot apply hmm_gpt2-large_common-gen_4096 to
# tulu2-7b_writing-prompts. To apply Ctrl-G to a custom base_model or to achieve
# best performance on a specific domain, users would need to distill an hmm_model
# from the base_model. Please refer to tutorial_distillation.ipynb for details.
BASE_MODEL_PATH = f'ctrlg/gpt2-large_common-gen' # a gpt2-large checkpoint domain adapted to the common-gen corpus
HMM_MODEL_PATH = f'ctrlg/hmm_gpt2-large_common-gen_4096' # alternatively 'ctrlg/hmm_gpt2-large_common-gen_32768' for better quality

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)
base_model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
hmm_model = ctrlg.HMM.from_pretrained(HMM_MODEL_PATH).to(device)


# Start the timer
start_time = time.time()

vocab_size = hmm_model.vocab_size
eos_token_id = hmm_model.eos_token_id

print('=== Settings and Prompts ===\n')

##################################### prefix, suffix, prompt #####################################
prefix = 'You see a' # generate text starting with nothing
suffix = '. Greeekkk!<|endoftext|>' # generate text ending with '<|endoftext|>'; a suffix must end with the eos token
prompt = '<|endoftext|> You see a' # prompt the base model with the '<|endoftext|>' token

prefix_ids = tokenizer.encode(prefix)
suffix_ids = tokenizer.encode(suffix)
prompt_ids = tokenizer.encode(prompt)

print(f'prefix = {prefix}')
print(f'suffix = {suffix}')
print(f'prompt = {prompt}')
##################################### prefix, suffix, prompt #####################################


##################################### DFA Construction #####################################
ac_builder = ctrlg.AhoCorasickBuilder(vocab_size)
word_count_builder = ctrlg.WordCountBuilder(tokenizer, vocab_size)

dfa_graphs = []

# constraint 1:
keyphrases = [
    ['dark', 'dingy'],
    ['room', 'bed']
]
for keyphrase in keyphrases:
    print(f'keyphrase = {keyphrase}')
    patterns = [tokenizer.encode(x) for x in keyphrase]
    dfa_graphs.append(ac_builder.build(patterns))

# constraint 2: generate exactly 10 words
a, b = 8, 15
print(f'min length = {a}, max length = {b}')
dfa_graphs.append(word_count_builder.build(a, b))

dfa_graph = ctrlg.DFA_prod(dfa_graphs, mode='intersection')
dfa_model = ctrlg.DFAModel(dfa_graph, vocab_size).to(device)
##################################### DFA Construction #####################################


##################################### token length #####################################
min_new_tokens = 5
max_new_tokens = 25
print(f'min_new_tokens = {min_new_tokens}, max_new_tokens = {max_new_tokens}')
##################################### token length #####################################

# initialze the constraints logits processor
# Note: this part pre-computes & cache certain conditional probability tables;
# one simple optimization is to re-use the same constraint_logits_processor for
# base_model.generate if the constraints do not change.
constraint_logits_processor = ctrlg.ConstraintLogitsProcessor(
    hmm_model, dfa_model,
    min_new_tokens, max_new_tokens,
    prompt_ids, prefix_ids=prefix_ids, suffix_ids=suffix_ids)


# set beam_size for beam search; usually the larger the beam_size the
# higher the generation quality
beam_size = 32
print(f'beam_size = {beam_size}')

# set the hmm_batch_size depending on the resource available;
# uses more memory with larger hmm_batch_size but attains best speed
# when it is set to beam_size
constraint_logits_processor.hmm_batch_size = 1
print(f'hmm_batch_size = {constraint_logits_processor.hmm_batch_size}')

# generate with beam search
input_ids = torch.tensor([prompt_ids], device=device)
outputs = base_model.generate(
        input_ids=input_ids, do_sample=False, length_penalty=0.2,
        num_beams=beam_size, num_return_sequences=beam_size,
        min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList([constraint_logits_processor]),
        pad_token_id=tokenizer.eos_token_id,
    )

# extract the generated ids; removing prompt ids; remove suffix ids that are (partially) generated
generated_ids = ctrlg.extract_generated_ids(outputs.tolist(), prompt_ids, suffix_ids, eos_token_id)

# rank the generated ids by the base_model probability
generated_ids = ctrlg.rank_generated_ids(base_model, generated_ids, prompt_ids, suffix_ids, length_penalty=0.2)

print('\n=== Generated Text ===\n')

# print top 10 outputs
for idx, generated in enumerate(generated_ids[:10]):
    print(f'{idx}. ' + tokenizer.decode(prefix_ids, skip_special_tokens=True) + \
          '\033[1m' + tokenizer.decode(generated, skip_special_tokens=True) + '\033[0m' + \
          tokenizer.decode(suffix_ids, skip_special_tokens=True))

end_time = time.time()
print(f'Generated in {end_time - start_time} seconds')

# separator
print('\n\n---------------------------------------------------------------------------------------------------------------------------\n\n')
