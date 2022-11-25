# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import tensorflow as tf
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3FeatureExtractor

from typing import Dict, Optional, List
import torch.nn as nn
import os
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor, LayoutLMv3Processor
from dataclasses import dataclass
from PIL import Image
import torch
import cv2

from get_aug_image import get_image
from src_utils import bbox_string

def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
  """Checks whether the casing config is consistent with the checkpoint name."""

  # The casing has to be passed in by the user and there is no explicit check
  # as to whether it matches the checkpoint. The casing information probably
  # should have been stored in the bert_config.json file, but it's not, so
  # we have to heuristically detect it to validate.

  if not init_checkpoint:
    return

  m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
  if m is None:
    return

  model_name = m.group(1)

  lower_models = [
      "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
      "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
  ]

  cased_models = [
      "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
      "multi_cased_L-12_H-768_A-12"
  ]

  is_bad_config = False
  if model_name in lower_models and not do_lower_case:
    is_bad_config = True
    actual_flag = "False"
    case_name = "lowercased"
    opposite_flag = "True"

  if model_name in cased_models and do_lower_case:
    is_bad_config = True
    actual_flag = "True"
    case_name = "cased"
    opposite_flag = "False"

  if is_bad_config:
    raise ValueError(
        "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
        "However, `%s` seems to be a %s model, so you "
        "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
        "how the model was pre-training. If this error is wrong, please "
        "just comment out this check." % (actual_flag, init_checkpoint,
                                          model_name, case_name, opposite_flag))


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = self._tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False

def get_subword_start_end(word_start, word_end, subword_idx2word_idx):
    ## find the separator between the questions and the text
    start_of_context = -1
    for i in range(len(subword_idx2word_idx)):
      if subword_idx2word_idx[i] is None and subword_idx2word_idx[i + 1] is None:
        start_of_context = i + 2
        break
    num_question_tokens = start_of_context
    assert start_of_context != -1, "Could not find the start of the context"
    subword_start = -1
    subword_end = -1
    for i in range(start_of_context, len(subword_idx2word_idx)):
      if word_start == subword_idx2word_idx[i] and subword_start == -1:
        subword_start = i
      if word_end == subword_idx2word_idx[i]:
        subword_end = i
    return subword_start, subword_end, num_question_tokens


def tokenize_docvqa(examples,
                    tokenizer: LayoutLMv3TokenizerFast,
                    img_dir: Dict[str, str],
                    add_metadata: bool = True,
                    combine_train_val_as_train: bool = False):
  """

  :param examples:
  :param tokenizer:
  :param max_seq_length:
  :param img_dir: {"train_dir": xxxx, "val_dir":xxx}
  :param add_metadata:
  :param shrink_vocab_mapper:
  :return:
  """
  features = {"input_ids": [], "image": [], "bbox": [], "start_positions": [], "end_positions": [], "metadata": []}
  current_split = examples["data_split"][0]
  for idx, (question, image_path, words, layout) in enumerate(
          zip(examples["question"], examples["image"], examples["words"], examples["layout"])):
    current_metadata = {}
    file = os.path.join(img_dir[examples["data_split"][idx]], image_path)
    # img = Image.open(file).convert("RGB")
    answer_list = examples["processed_answers"][idx] if "processed_answers" in examples else []
    original_answer = examples["original_answer"][idx] if "original_answer" in examples else []
    image_id = f"{examples['ucsf_document_id'][idx]}_{examples['ucsf_document_page_no'][idx]}"
    if len(words) == 0 and current_split == "train":
      continue
    tokenized_res = tokenizer.encode_plus(text=question, text_pair=words, boxes=layout, add_special_tokens=True,
                                          return_tensors="pt", max_length=512, truncation="only_second",
                                          return_offsets_mapping=True)

    input_ids = tokenized_res["input_ids"][0]

    subword_idx2word_idx = tokenized_res.encodings[0].word_ids
    img = cv2.imread(file)
    height, width = img.shape[:2]
    if current_split == "train" or (current_split == "val" and combine_train_val_as_train):
      # for troaining, we treat instances with multiple answers as multiple instances
      for answer in answer_list:
        if answer["start_word_position"] == -1:
          continue
        subword_start, subword_end, num_question_tokens = get_subword_start_end(answer["start_word_position"],
                                                                                answer["end_word_position"],
                                                                                subword_idx2word_idx)
        if subword_start == -1:
          continue
        if subword_end == -1:
          subword_end = 511 - 1  ## last is </s>, second last
        features["image"].append(file)
        features["input_ids"].append(input_ids)
        # features["attention_mask"].append(tokenized_res["attention_mask"])
        # features["bbox"].append(tokenized_res["bbox"][0])
        boxes_norms = []
        for box in tokenized_res["bbox"][0]:
          box_norm = bbox_string([box[0], box[1], box[2], box[3]], width, height)
          assert box[2] >= box[0]
          assert box[3] >= box[1]
          assert box_norm[2] >= box_norm[0]
          assert box_norm[3] >= box_norm[1]
          boxes_norms.append(box_norm)
        features["bbox"].append(boxes_norms)
        features["start_positions"].append(subword_start)
        features["end_positions"].append(subword_end)
        current_metadata["original_answer"] = original_answer
        current_metadata["question"] = question
        current_metadata["num_question_tokens"] = num_question_tokens
        current_metadata["words"] = words
        current_metadata["subword_idx2word_idx"] = subword_idx2word_idx
        current_metadata["questionId"] = examples["questionId"][idx]
        current_metadata["data_split"] = examples["data_split"][idx]
        features["metadata"].append(current_metadata)
        if not add_metadata:
          features.pop("metadata")
    else:
      # for validation and test, we treat instances with multiple answers as one instance
      # we just use the first one, and put all the others in the "metadata" field
      # find the first answer that has start and end
      final_start_word_pos = 1  ## if not found, just for nothing, because we don't use it anyway for evaluation
      final_end_word_pos = 1
      for answer in answer_list:
        if answer["start_word_position"] == -1:
          continue
        else:
          final_start_word_pos = answer["start_word_position"]
          final_end_word_pos = answer["end_word_position"]
          break
      subword_start, subword_end, num_question_tokens = get_subword_start_end(final_start_word_pos, final_end_word_pos,
                                                                              subword_idx2word_idx)
      if subword_end == -1:
        subword_end = 511 - 1  ## last is </s>, second last
      features["image"].append(file)
      features["input_ids"].append(input_ids)
      # features["attention_mask"].append(tokenized_res["attention_mask"])
      # features["bbox"].append(tokenized_res["bbox"][0])
      boxes_norms = []
      for box in tokenized_res["bbox"][0]:
        box_norm = bbox_string([box[0], box[1], box[2], box[3]], width, height)
        assert box[2] >= box[0]
        assert box[3] >= box[1]
        assert box_norm[2] >= box_norm[0]
        assert box_norm[3] >= box_norm[1]
        boxes_norms.append(box_norm)
      features["bbox"].append(boxes_norms)
      features["start_positions"].append(subword_start)
      features["end_positions"].append(subword_end)
      current_metadata["original_answer"] = original_answer
      current_metadata["question"] = question
      current_metadata["num_question_tokens"] = num_question_tokens
      current_metadata["words"] = words
      current_metadata["subword_idx2word_idx"] = subword_idx2word_idx
      current_metadata["questionId"] = examples["questionId"][idx]
      current_metadata["data_split"] = examples["data_split"][idx]
      features["metadata"].append(current_metadata)
      if not add_metadata:
        features.pop("metadata")
  return features


@dataclass
class DocVQACollator:
  tokenizer: LayoutLMv3TokenizerFast
  feature_extractor: LayoutLMv3FeatureExtractor
  padding: bool = True
  model: Optional[nn.Module] = None

  def __call__(self, batch: List):

    for feature in batch:

      images, images_aug = get_image(feature["image"])

      feature['pixel_values_image'] = images
      feature['pixel_values_image_aug'] = images_aug

      if 'image' in feature: feature.pop('image')

    batch = self.tokenizer.pad(
      batch,
      padding='max_length',
      pad_to_multiple_of=None,
      return_tensors="pt",
      return_attention_mask=True,
      max_length=512
    )
    return batch