# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
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

"""Format-specific dataset builders for CoNLL-like formatted data."""
import collections
# import dataclasses
from typing import List, Optional, Sequence, Union

from etils import epath
import tensorflow as tf
from tensorflow_datasets.core import dataset_builder
from tensorflow_datasets.core import dataset_info
from tensorflow_datasets.core import split_builder as split_builder_lib
from tensorflow_datasets.core.features import top_level_feature
from tensorflow_datasets.core.features.features_dict import FeaturesDict
from tensorflow_datasets.core.features.tensor_feature import Tensor


# TODO(py3.10): Should update ConllBuilderConfig to use @dataclasses.dataclass.
class ConllBuilderConfig(dataset_builder.BuilderConfig):
  """Base class for CoNLL-like formatted data configuration."""

  def __init__(self, *, separator: str,
               ordered_features: collections.OrderedDict[str, Union[
                   Tensor, top_level_feature.TopLevelFeature]], **kwargs):
    """Builder config for Conll datasets.

    Args:
      separator: The separator used for splitting feature columns in the input
        lines. For CoNLL-formatted data, this is usually a tab or a space.
      ordered_features: An OrderedDict specifying the features names and their
        type, in the same order as they appear as columns in the input lines.
      **kwargs: keyword arguments forwarded to super.
    """
    super(ConllBuilderConfig, self).__init__(**kwargs)
    self.separator = separator
    self.ordered_features = ordered_features

  @property
  def features_dict(self) -> FeaturesDict:
    return FeaturesDict(self.ordered_features)


class ConllDatasetBuilder(dataset_builder.GeneratorBasedBuilder):
  """Base class for CoNLL-like formatted datasets.

  It provides functionalities to ease the processing of CoNLL-like datasets.
  Users can overwrite `_generate_examples` to customize the pipeline.
  """
  BUILDER_CONFIGS: Sequence[ConllBuilderConfig] = []

  @property
  def builder_config(self) -> ConllBuilderConfig:
    """`tfds.core.BuilderConfig` for this builder."""
    return self._builder_config

  def create_dataset_info(
      self,
      description: str,
      supervised_keys: Optional[dataset_info.SupervisedKeysType] = None,
      homepage: Optional[str] = None,
      citation: Optional[str] = None,
  ) -> dataset_info.DatasetInfo:
    return dataset_info.DatasetInfo(
        builder=self,
        description=description,
        features=self.builder_config.features_dict,
        supervised_keys=supervised_keys,
        homepage=homepage,
        citation=citation,
    )

  def _generate_examples(
      self,
      path: Union[epath.PathLike, List[epath.PathLike]],
      use_beam: bool = False,
  ) -> split_builder_lib.SplitGenerator:
    """Function to process CoNLL-like datasets and generate examples.

    Args:
      path: The filepaths of the input data. Could be a list of paths for
        multiple input files, or a single path.
      use_beam: Whether to call a beam-pipeline or not. Default is False.

    Yields:
      Generated examples.

    Raises:
      ValueError if the number of column features encountered doesn't match the
        expected number of features.
    """
    if not isinstance(path, list):
      path = [path]

    input_sequences = {feature: [] for feature in self.info.features}

    example_id = 0
    for filepath in path:
      with tf.io.gfile.GFile(filepath) as fin:
        for line in fin:
          if line.startswith("-DOCSTART-") or line == "\n" or not line:
            if input_sequences["tokens"]:
              yield example_id, input_sequences
              example_id += 1
              input_sequences = {feature: [] for feature in self.info.features}
          else:
            splits = line.split(self.builder_config.separator)
            if len(splits) != len(self.builder_config.ordered_features):
              raise ValueError(
                  (f"Mismatch in the number of features found in line: {line}\n"
                   f"Should be {len(self.builder_config.ordered_features)}, "
                   f"but found {len(splits)}"))
            for index, feature in enumerate(
                self.builder_config.ordered_features.keys()):
              input_sequences[feature].append(splits[index].rstrip())

      # Last example from file.
      yield example_id, input_sequences
      example_id += 1
      input_sequences = {feature: [] for feature in self.info.features}
