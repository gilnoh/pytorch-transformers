# minimal prediction example code.
#
# Gil


# imports
from __future__ import absolute_import, division, print_function
import logging
import timeit
import torch
import numpy as np
import operator
from pytorch_transformers import (BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from utils_glue import (convert_examples_to_features, InputExample)

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
checkpoint = "/home/tailblues/temp/CHITCHAT_MULTI_TESTING"
config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
LABEL_LIST = ["POSITIVE", "ABOUT_ME", "TALK_WITH_AGENT",
              "WEATHER", "FAREWELL", "NEGATIVE_NOT_MEANT",
              "OPENING", "CURRENT_TIME",
              "GREETING", "GRATITUDE", "QUESTION"]
# load model and tokenizer instances
logger.info("Loading model from the following checkpoint: %s", checkpoint)
tokenizer = tokenizer_class.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = model_class.from_pretrained(checkpoint)
model.eval()  # put the model in eval mode (e.g. no drop out)
model.to(device)


# pre-processing
example1 = InputExample(guid=1,
                        text_a="verbinden bitte",
                        label="QUESTION")
example2 = InputExample(guid=2,
                        text_a="mitarbeiter",
                        label="QUESTION")
features = convert_examples_to_features([example1, example2],
                                        LABEL_LIST,
                                        MAX_SEQUENCE_LENGTH,
                                        tokenizer,
                                        "classification",
                                        cls_token_at_end=False,
                                        cls_token=tokenizer.cls_token,
                                        sep_token=tokenizer.sep_token,
                                        cls_token_segment_id=0,
                                        pad_on_left=False,
                                        pad_token_segment_id=0)

# convert feature list to torch tensor
all_input_ids = torch.tensor(
    [f.input_ids for f in features], dtype=torch.long).to(device)
all_input_mask = torch.tensor(
    [f.input_mask for f in features], dtype=torch.long).to(device)
all_segment_ids = torch.tensor(
    [f.segment_ids for f in features], dtype=torch.long).to(device)
all_label_ids = torch.tensor(
    [f.label_id for f in features], dtype=torch.long).to(device)


# actual prediction call
start = timeit.timeit()  # to measure time
with torch.no_grad():
    outputs = model(input_ids=all_input_ids,
                    attention_mask=all_input_mask,
                    token_type_ids=all_segment_ids)
#                    labels=all_label_ids)
#    loss, logits = outputs[:2]
    logits = outputs[0]
    preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
    softmax_probs = (torch.nn.functional.softmax(logits, dim=1)
                     .detach().cpu().numpy())
    raw_logits = logits.detach().cpu().numpy()
    logistic_probs = [(np.exp(x) / (1 + np.exp(x))) for x in logits
                      .detach().cpu().numpy()]
    sigmoid_probs = [(1 / (1 + np.exp(-x))) for x in logits
                     .detach().cpu().numpy()]

    logger.info(logits)  # logits before softmax
    # logger.info(softmax_probs)   # (pseudo-) prob from softmax
    # logger.info(logistic_probs)  # alternative prob from logistic f

    # prepare detailed result, logistic prob sorted list of labels
    probs = softmax_probs
    result = []
    for entry in probs:
        label_scores = []
        for i in range(len(entry)):
            tuple = (LABEL_LIST[i], entry[i])
            label_scores.append(tuple)
        label_scores.sort(key=operator.itemgetter(1), reverse=True)
        result.append(label_scores)
    logger.info(result)

    # finally, shorter version of result
    label_preds = [LABEL_LIST[i] for i in preds]
    label_probs = [max(p) for p in probs]
    logger.info(list(zip(label_preds, label_probs)))   # decision

end = timeit.timeit()
print("time took: %.8f" % (end - start))
