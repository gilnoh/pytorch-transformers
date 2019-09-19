# TMT task, minimal training example

# imports
from __future__ import absolute_import, division, print_function
import logging
import os
import random
import numpy as np
import torch
import csv
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from pytorch_transformers import (BertConfig, BertForSequenceClassification,
                                  BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer,
                                  XLNetConfig, XLNetForSequenceClassification,
                                  XLNetTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule
from utils_glue import (convert_examples_to_features, InputExample)

# globals
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   '
                    '%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}
config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training params
TRAIN_BATCH_SIZE = 8     # aka per-gpu batch size
NUM_TRAIN_EPOCHS = 3.0
LEARNING_RATE = 2e-5  # 2e-5, default from run_glue example description
WEIGHT_DECAY = 0.0
ADAM_EPSILON = 1e-8  # 1e-8, default from run_glue args
WARMUP_STEPS = 0
MAX_GRAD_NORM = 1.0
SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
LOGGING_STEPS = 50
SAVE_STEPS = 1000
LOCAL_RANK = -1  # ftm. (local rank not used) changing this won't work
N_GPU = 1        # ftm. (multi GPU not used) changing this won't work
FP16 = False
FP16_OPT_LEVEL = 'O1'


# work functions
def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)


def train(train_dataset, model, tokenizer, output_path):
    """ Train the model """
    tb_writer = SummaryWriter()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
    t_total = (len(train_dataloader) //
               GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [
            p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)
        ], 'weight_decay': WEIGHT_DECAY},
        {'params': [
            p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)
        ], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=LEARNING_RATE,
        eps=ADAM_EPSILON)
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=WARMUP_STEPS,
        t_total=t_total)

    if FP16:  # for FP16
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from"
                "https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=FP16_OPT_LEVEL)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", NUM_TRAIN_EPOCHS)
    logger.info("  Instantaneous batch size per GPU = %d", TRAIN_BATCH_SIZE)
    logger.info("  Gradient Accumulation steps = %d",
                GRADIENT_ACCUMULATION_STEPS)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(NUM_TRAIN_EPOCHS), desc="Epoch", disable=False)
    set_seed(SEED)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            if N_GPU > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            if FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer),
                    MAX_GRAD_NORM)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM)

            tr_loss += loss.item()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # logging (tb_writer)
                if (LOCAL_RANK in [-1, 0] and
                        LOGGING_STEPS > 0 and
                        global_step % LOGGING_STEPS == 0):
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss',
                        (tr_loss - logging_loss)/LOGGING_STEPS, global_step)
                    logging_loss = tr_loss

                # save model checkpoint
                if (LOCAL_RANK in [-1, 0] and
                        SAVE_STEPS > 0 and
                        global_step % SAVE_STEPS == 0):
                    output_dir = os.path.join(
                        output_path, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model.save_pretrained(output_dir)
                    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    # above line: saving training arguments, we skip this.
                    logger.info("Saving model checkpoint to %s", output_dir)

    tb_writer.close()
    return global_step, tr_loss / global_step


def read_mode4_tsv(file):
    examples = []
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        entries = []
        for line in reader:
            entries.append(line)
    for (i, entry) in enumerate(entries):
        guid = "%s" % i
        if(len(entry) > 2):
            examples.append(
                InputExample(guid=guid,
                             text_a=entry[0],
                             text_b=entry[1],
                             label=entry[2]))
        else:
            examples.append(
                InputExample(guid=guid,
                             text_a=entry[0],
                             label=entry[1]))

    return examples


def load_examples(data, label_list, tokenizer, max_sequence_length):
    logger.info("Creating features from dataset file at %s", data)
    # label_list = processor.get_labels()
    examples = read_mode4_tsv(data)
    logger.info("Loaded %s examples", len(examples))

    # convert loaded InputExample to InputFeatures
    features = convert_examples_to_features(
        examples,
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

    # convert the list of InputFeature to Tensors, and build a dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset


#
# main
#
def main():
    # consts
    MAX_SEQUENCE_LENGTH = 128
    DO_LOWER_CASE = True

    # model and data
    PRETRAINED = "bert-base-multilingual-uncased"
    OUTPUT_DIR = "/home/tailblues/omq/models/tmt_base"
    DATA = "/home/tailblues/omq/dataset/mnli_dev_matched.tmt.tsv"
    LABEL_LIST = ["NO", "YES"]

    # prepare model and tokenizer
    logger.info(
        "Preparing model from the following pretrained: %s",
        PRETRAINED)
    config = config_class.from_pretrained(
        PRETRAINED,
        num_labels=len(LABEL_LIST))
    tokenizer = tokenizer_class.from_pretrained(
        PRETRAINED, do_lower_case=DO_LOWER_CASE)
    model = model_class.from_pretrained(
        PRETRAINED, from_tf=False, config=config)
    model.to(device)

    # prepare dataset
    train_dataset = load_examples(
        DATA, LABEL_LIST, tokenizer, MAX_SEQUENCE_LENGTH)
    # logger.info(train_dataset)

    # prep output dir and call train
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    global_step, tr_loss = train(train_dataset, model, tokenizer, OUTPUT_DIR)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # save trained data, OUTPUT_DIR already exist due to train() function
    logger.info("Saving model checkpoint to %s", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
