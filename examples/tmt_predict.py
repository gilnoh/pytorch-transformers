# TMT task, prediction example

from __future__ import absolute_import, division, print_function
import logging
import torch
import numpy as np
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

# global variables that are set once
config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# default values for model dependent values
LABEL_LIST = ["NO", "YES"]
MAX_SEQUENCE_LENGTH = 128
DO_LOWER_CASE = True


#
# methods
#
def load_tokenizer(checkpoint, do_lower_case=DO_LOWER_CASE):
    """
    Returns tokenizer from pretrained model pointed at the checkpoint
    """
    return tokenizer_class.from_pretrained(
        checkpoint, do_lower_case=do_lower_case)


def load_model(checkpoint):
    """
    Loads pre-trained model from the given checkpoint
    """
    model = model_class.from_pretrained(checkpoint)
    model.eval()  # put the model in eval mode (e.g. no drop out)
    model.to(device)
    return model


def predict(model, tokenizer, textlist,
            label_list=LABEL_LIST,
            max_sequence_length=MAX_SEQUENCE_LENGTH):
    """
    gets a list of text and returns a list of predictions
    """
    examples = []
    for i in range(len(textlist)):
        examples.append(InputExample(
            guid=100 + i,  # guid doesn't really matter (only id)
            text_a=textlist[i][0],  # text to be predicted (real input)
            text_b=textlist[i][1],
            label=label_list[-1]))  # label (don't affect, but must not empty)

    features = convert_examples_to_features(examples,
                                            label_list,
                                            max_sequence_length,
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
            label_scores[label_list[i]] = entry[i]
        result.append(label_scores)

    # decision
    label_preds = [label_list[i] for i in preds]

    # score of that decision
    label_probs = label_probs = [max(p) for p in probs]

    # return full results as tuple (decision, decision score, full details)
    return (label_preds, label_probs, result, raw_logits)


#
# main; usage example
#
def main():
    # model's configurations
    CHECKPOINT = "/home/tailblues/omq/models/tmt_base"

    # prepare tokenizer and model, takes time, need to be done only once
    tokenizer = load_tokenizer(CHECKPOINT)
    model = load_model(CHECKPOINT)

    textlist = [("He likes the movie very much",
                 "He loves the movie."),
                ("I have a dog and a cat",
                 "I have pets"),
                ("i don't know um do you do a lot of camping.",
                 "I know exactly."),
                ("The sacred is not mysterious to her.",
                 "The woman is familiar with the sacred."),
                ("In the summer, the Sultan's Pool, a vast outdoor amphitheatre, stages rock concerts or other big-name events.",
                 "Most rock concerts take place in the Sultan's Pool amphitheatre."),
                ("Around 1500 b.c. , a massive volcanic eruption at Santorini destroyed not only Akrotiri under feet of ash and pumice but the whole Minoan civilization.",
                 "The entire Minoan civilization was destroyed by a volcanic eruption."),
                ("Small towns like Louisian lay scattered all over the Oil Fields; the main train line branched between them.",
                 "There were a lot of small towns in the oil fields.")]

    (decision, decision_score, details, raw_logits) =\
        predict(model, tokenizer, textlist)
    logger.info(decision)
    logger.info(decision_score)
    logger.info(details)
    logger.info(raw_logits)


if __name__ == "__main__":
    main()
