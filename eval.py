
import collections
import json
import os

import torch
from transformers import BertTokenizer
from tqdm import tqdm

from qgevalcap.eval import eval_qg
from squad_utils import evaluate, write_predictions
from utils import batch_to_device

def to_string(index, tokenizer):
    tok_tokens = tokenizer.convert_ids_to_tokens(index)
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace("[PAD]", "")
    tok_text = tok_text.replace("[SEP]", "")
    tok_text = tok_text.replace("[CLS]", "")
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    return tok_text

class Result(object):

    def __init__(self,
                 context,
                 real_question,
                 generated_question,
                 answer):
        self.context = context
        self.real_question = real_question
        self.generated_question = generated_question
        self.answer = self.answer


def eval_vae(epoch, args, trainer, eval_data):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])

    eval_loader, eval_examples, eval_features = eval_data

    all_results = []
    qa_results = []
    qg_results = {}
    res_dict = {}
    example_index = -1

    for batch in tqdm(eval_loader, desc="Eval iter", leave=False, position=3):
        c_ids, q_ids, a_ids = batch_to_device(batch, args.device)
        batch_size = c_ids.size(0)
        batch_q_ids = q_ids.cpu().tolist()

        generated_q_ids = trainer.model.generate(c_ids, a_ids)

        generated_q_ids = generated_q_ids.cpu().tolist()

        for i in range(batch_size):
            example_index += 1
            eval_feature = eval_features[example_index]
            unique_id = int(eval_feature.unique_id)

            real_question = to_string(batch_q_ids[i], tokenizer)
            generated_question = to_string(generated_q_ids[i], tokenizer)

            qg_results[unique_id] = generated_question
            res_dict[unique_id] = real_question
    bleu = eval_qg(res_dict, qg_results)

    return bleu
