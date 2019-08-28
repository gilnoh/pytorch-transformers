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
CHECKPOINT = "/home/tailblues/temp/CHITCHAT_MULTI_TESTING"
LABEL_LIST = ["POSITIVE", "ABOUT_ME", "TALK_WITH_AGENT",
              "WEATHER", "FAREWELL", "NEGATIVE_NOT_MEANT",
              "OPENING", "CURRENT_TIME",
              "GREETING", "GRATITUDE", "QUESTION"]

# global variables that are set once
config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model and tokenizer instances
# logger.info("Loading model from the following checkpoint: %s", checkpoint)
# tokenizer = tokenizer_class.from_pretrained(checkpoint)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")
# model = model_class.from_pretrained(checkpoint)
# model.eval()  # put the model in eval mode (e.g. no drop out)
# model.to(device)


#
# methods
#
def load_tokenizer(checkpoint):
    """
    Returns tokenizer from pretrained model pointed at the checkpoint
    """
    return tokenizer_class.from_pretrained(checkpoint)


def load_model(checkpoint):
    """
    Loads pre-trained model from the given checkpoint
    """
    model = model_class.from_pretrained(checkpoint)
    model.eval()  # put the model in eval mode (e.g. no drop out)
    model.to(device)
    return model


def predict(model, tokenizer, textlist):
    """
    gets a list of text and returns a list of predictions
    """
    examples = []
    for i in range(len(textlist)):
        examples.append(InputExample(
            guid=100 + i,  # guid doesn't really matter (only id)
            text_a=textlist[i],  # text to be predicted (real input)
            label=LABEL_LIST[-1]))  # label (don't affect, but must not empty)

    features = convert_examples_to_features(examples,
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
    with torch.no_grad():
        outputs = model(input_ids=all_input_ids,
                        attention_mask=all_input_mask,
                        token_type_ids=all_segment_ids)
        logits = outputs[0]
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        softmax_probs = (torch.nn.functional.softmax(logits, dim=1)
                         .detach().cpu().numpy())
        raw_logits = logits.detach().cpu().numpy()

    # prepare detailed result output
    probs = softmax_probs
    result = []
    for entry in probs:
        label_scores = {}
        for i in range(len(entry)):
            label_scores[LABEL_LIST[i]] = entry[i]
        result.append(label_scores)

    # decision
    label_preds = [LABEL_LIST[i] for i in preds]

    # score of that decision
    label_probs = label_probs = [max(p) for p in probs]

    # return full results as tuple (decision, decision score, full details)
    return (label_preds, label_probs, result)


#
# main; usage example
#
def main():
    # prepare tokenizer and model, takes time, need to be done only once
    # note that, saved model CHECKPOINT path (string) is the only param.
    # all other options like base-model, max seq length, etc are hard coded
    # in this file, and set at import time.
    tokenizer = load_tokenizer(CHECKPOINT)
    model = load_model(CHECKPOINT)

    textlist = ["verbiden bitte",
                "Guten Tag!",
                "Bonjour.",
                "How do you feel today?",
                "das ist nicht was ich meine"]

    (decision, decision_score, details) = predict(model, tokenizer, textlist)
    logger.info(decision)
    logger.info(decision_score)
    logger.info(details)


if __name__ == "__main__":
    main()
