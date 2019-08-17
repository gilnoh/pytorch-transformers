# minimal eval example code
#
# Gil

from __future__ import absolute_import, division, print_function
import logging
import torch
from torch.utils.data import (DataLoader, SequentialSampler)
from tqdm import tqdm
import numpy as np
from pytorch_transformers import (BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from utils_glue import (simple_accuracy)
from minimal_training2 import (load_examples)


# globals including loggers and model class definitions
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   '
                    '%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}

# consts - need to be changed for different models
MAX_SEQUENCE_LENGTH = 128
DO_LOWER_CASE = True
CHECKPOINT = "/home/tailblues/temp/MRPC_MULTI_TESTING"
DATA = "/home/tailblues/progs/glue/mrpc_dev.tsv"
LABEL_LIST = ["0", "1"]
EVAL_BATCH_SIZE = 8
config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]


# work functions
def evaluate(eval_dataset, model, tokenizer):
    results = {}

    # simple, not distributed sampler
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", EVAL_BATCH_SIZE)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(
                preds,
                logits.detach().cpu().numpy(),
                axis=0)
            out_label_ids = np.append(
                out_label_ids,
                inputs['labels'].detach().cpu().numpy(),
                axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)  # converting logits to label id
    result = {  # simple accuray as the measure
        "acc": simple_accuracy(preds, out_label_ids)}
    results.update(result)

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return results


#
# main
#

# load model and tokenizer instances
logger.info("Loading model from the following checkpoint: %s", CHECKPOINT)
tokenizer = tokenizer_class.from_pretrained(CHECKPOINT,
                                            do_lower_case=DO_LOWER_CASE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = model_class.from_pretrained(CHECKPOINT)
model.eval()  # put the model in eval mode (e.g. no drop out)
model.to(device)

# load evaluation dataset and run eval
eval_dataset = load_examples(DATA, LABEL_LIST, tokenizer, MAX_SEQUENCE_LENGTH)
result = evaluate(eval_dataset, model, tokenizer)
# logger.info(result)
