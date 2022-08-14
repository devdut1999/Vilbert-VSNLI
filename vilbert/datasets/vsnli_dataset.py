# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import jsonlines
import _pickle as cPickle
import logging
import pandas as pd
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers.tokenization_bert import BertTokenizer

from ._image_features_reader import ImageFeaturesH5Reader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

LABEL_MAP = {"contradiction": 0, "neutral": 1, "entailment": 2}


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(item):
    entry = {
        "question_id": item["question_id"],
        "image_id": item["image_id"],
        "hypothesis": item["hypothesis"],
        "premise": item["premise"],
        "answer": item,
    }
    return entry


def _load_dataset(dataroot, name, clean_datasets):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'dev', 'test'
    """

    datapath = dataroot+name+".csv"
    # if name == "train":
    #     x = pd.read_csv("datasets/paco/train.csv")
    # elif name == "dev" or name == "test":
    #     x = pd.read_csv("datasets/paco/test.csv")
    x = pd.read_csv(datapath)
        # remove_ids = []
        # if clean_datasets:
        #     remove_ids = np.load(
        #         os.path.join(dataroot, "cache", "flickr_test_ids.npy")
        #     )
        #     remove_ids = [int(x) for x in remove_ids]
        # Build an index which maps image id with a list of hypothesis annotations.
    items = []
    y = len(x)
    cnt = 0
    while cnt < y:
        dictionary = {}
        dictionary["image_id"] = x.iloc[cnt]["filename"]
        # if name == "train" and dictionary["image_id"] in remove_ids:
        #     continue
        dictionary["question_id"] = cnt
        dictionary["hypothesis"] = x.iloc[cnt]["precondition"]
        dictionary["premise"] = x.iloc[cnt]["query"]
        
        dictionary["labels"] = [x.iloc[cnt]["label"]]
        # dictionary["image_id"] = x.iloc[cnt]["image"]
        # if name == "train" and dictionary["image_id"] in remove_ids:
        #     continue
        # dictionary["question_id"] = cnt
        # dictionary["hypothesis"] = x.iloc[cnt]["sentence2"]
        # dictionary["premise"] = x.iloc[cnt]["sentence1"]
        # dictionary["labels"] = [
        #   int(LABEL_MAP[x.iloc[cnt]["gold_label"]])
        # ]
        
        dictionary["scores"] = [1.0]
        items.append(dictionary)
        cnt += 1

    entries = []
    for item in items:
        entries.append(_create_entry(item))
    return entries


def npy_feature_extraction(image_id):
    infile = "/nas/home/devadutt/Vilbert-VSNLI/datasets/paco/extracted_features/"+image_id[:-4]+".npy"
    reader = np.load(infile, allow_pickle=True)
    item = {}
    item["image_id"] = reader.item().get("image_id")
    img_id = str(item["image_id"]).encode()
    # id_list.append(img_id)
    item["image_h"] = reader.item().get("image_height")
    item["image_w"] = reader.item().get("image_width")
    item["num_boxes"] = reader.item().get("num_boxes")
    item["boxes"] = reader.item().get("bbox")
    item["features"] = reader.item().get("features")

    # print("Image Id : ",item["image_id"])
    # print("Features : ", item["features"].shape, " ", type(item["features"]))
    # print("Boxes : ", item["boxes"].shape, " ", type(item["boxes"]))
    # print("num_boxes : ", item["num_boxes"], " ", type(item["num_boxes"]))
    # print("image_h : ", item["image_h"], " ", type(item["image_h"]))
    # print("image_w : ", item["image_w"], " ", type(item["image_w"]))
    
    image_h = int(item["image_h"])
    image_w = int(item["image_w"])
    features = item["features"].reshape(-1, 2048)
    boxes = item["boxes"].reshape(-1, 4)

    num_boxes = features.shape[0]
    g_feat = np.sum(features, axis=0) / num_boxes
    num_boxes = num_boxes + 1
    features = np.concatenate(
        [np.expand_dims(g_feat, axis=0), features], axis=0
    )

    image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
    image_location[:, :4] = boxes
    image_location[:, 4] = (
        (image_location[:, 3] - image_location[:, 1])
        * (image_location[:, 2] - image_location[:, 0])
        / (float(image_w) * float(image_h))
    )

    image_location_ori = copy.deepcopy(image_location)
    image_location[:, 0] = image_location[:, 0] / float(image_w)
    image_location[:, 1] = image_location[:, 1] / float(image_h)
    image_location[:, 2] = image_location[:, 2] / float(image_w)
    image_location[:, 3] = image_location[:, 3] / float(image_h)

    g_location = np.array([0, 0, 1, 1, 1])
    image_location = np.concatenate(
        [np.expand_dims(g_location, axis=0), image_location], axis=0
    )

    g_location_ori = np.array([0, 0, image_w, image_h, image_w * image_h])
    image_location_ori = np.concatenate(
        [np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0
    )

    return features, num_boxes, image_location, image_location_ori

class VisualNLIDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        bert_model,
        clean_datasets,
        padding_index: int = 0,
        max_seq_length: int = 16,
        max_region_num: int = 37,
    ):
        super().__init__()
        self.split = split
        self.num_labels = 3
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index

        clean_train = "_cleaned" if clean_datasets else ""

        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + clean_train
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                "Paco" + "_" + split + "_" + str(max_seq_length) + clean_train + ".pkl",
            )

        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split, clean_datasets)
            print("Tokenizing and Tensorizing ... ")
            self.tokenize(max_seq_length)
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        cnt  = 0
        for entry in self.entries:
            # tokens = self._tokenizer.tokenize(entry["hypothesis"])
            # tokens = ["[CLS]"] + tokens + ["[SEP]"]

            # tokens = [
            #     self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
            #     for w in tokens
            # ]
            # print(cnt)
            # print(type(entry["hypothesis"]))
            # print(entry["hypothesis"])

            if(type(entry["hypothesis"]) != float):
                tokens_1 = self._tokenizer.encode(entry["premise"])
                tokens_2 = self._tokenizer.encode(entry["hypothesis"])

                self.truncate_seq_pair(tokens_1, tokens_2,max_length - 3)
                tokens = self._tokenizer.add_special_tokens_sentences_pair(tokens_1,tokens_2)
                x = [0]*(2 + len(tokens_1))
                y = [1]*(len(tokens_2) + 1)
                x = x + y
                # print(x)
                segment_ids = x
                # tokens = ["[CLS]"] + tokens_1 + ["[SEP]"] + tokens_2 + ["[SEP]"]
                

                # tokens = self._tokenizer.encode(entry["hypothesis"])
                # tokens = tokens[: max_length - 2]
                # tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)
            else:
                # self.entries.remove(entry)
                # continue
                tokens = []
                tokens_1 = []
                tokens_2 = []
                segment_ids = []

            # segment_ids = [0] * len(tokens)

            

            # print(tokens)
            # print(type(tokens))
            input_mask = [1] * len(tokens)
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids
            
            # print(entry["hypothesis"])
            # print("Tokens", tokens)
            cnt += 1
        # print(cnt)

    def tensorize(self):

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            answer = entry["answer"]
            # print(answer["labels"])
            labels = np.array(answer["labels"])
            scores = np.array(answer["scores"], dtype=np.float32)
            if labels.size:
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry["answer"]["labels"] = labels
                entry["answer"]["scores"] = scores
            else:
                entry["answer"]["labels"] = None
                entry["answer"]["scores"] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        # print("Image id: ",image_id)
        question_id = entry["question_id"]
        # print("Question id: ",question_id)
        # features, num_boxes, boxes, _ = self._image_features_reader[image_id]
        features, num_boxes, boxes, _ = npy_feature_extraction(image_id)

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        hypothesis = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        target = torch.zeros(self.num_labels)

        answer = entry["answer"]
        labels = answer["labels"]
        scores = answer["scores"]
        if labels is not None:
            target.scatter_(0, labels, scores)

        return (
            features,
            spatials,
            image_mask,
            hypothesis,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id,
        )

    def __len__(self):
        return len(self.entries)

