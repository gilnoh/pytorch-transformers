# minimal prediction example code.
#
# Gil


# imports
from __future__ import absolute_import, division, print_function
import logging
import timeit
import torch
from pytorch_transformers import (BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from utils_glue import (convert_examples_to_features,
                        output_modes, processors, InputExample)

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
checkpoint = "/home/tailblues/temp/MRPC_OUT_TESTING"
config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]


# load model and tokenizer instances
logger.info("Loading model from the following checkpoint: %s", checkpoint)
tokenizer = tokenizer_class.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = model_class.from_pretrained(checkpoint)
model.eval()  # put the model in eval mode (e.g. no drop out)
model.to(device)


# pre-processing
processor = processors["mrpc"]()
output_mode = output_modes["mrpc"]
example1 = InputExample(guid=1,
                        text_a="What I watched that night was not the news, but a football game.",
                        text_b="I watched football that night.",
                        label="1")
example2 = InputExample(guid=2,
                        text_a="I like drinking beer.",
                        text_b="Something that is totally not related.",
                        label="1")
features = convert_examples_to_features([example1, example2],
                                        processor.get_labels(),
                                        MAX_SEQUENCE_LENGTH,
                                        tokenizer,
                                        output_mode,
                                        cls_token_at_end=False,
                                        cls_token=tokenizer.cls_token,
                                        sep_token=tokenizer.sep_token,
                                        cls_token_segment_id=0,
                                        pad_on_left=False,
                                        pad_token_segment_id=0)


# convert feature list to torch tensor
all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)
all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(device)
all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(device)


# actual prediction call
start = timeit.timeit()  # to measure time
with torch.no_grad():
    outputs = model(input_ids=all_input_ids,
                    attention_mask=all_input_mask,
                    token_type_ids=all_segment_ids,
                    labels=all_label_ids)
    loss, logits = outputs[:2]
    logger.info(loss)
    logger.info(logits)  # logits before softmax
end = timeit.timeit()
print("time took: %.8f" % (end - start))
