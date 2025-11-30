# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Helper functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from flax import nnx, traverse_util
from flax.typing import VariableDict  # pylint: disable=g-importing-member,g-multiple-import

M = TypeVar('M', bound='nnx.Module')


def _flatten_path(path: tuple[str | int, ...]) -> str:
  def f(item) -> str:
    if isinstance(item, str):
      return f'{item}'
    elif isinstance(item, int):
      return f'[{item}]'
    else:
      raise ValueError(f'Unexpected type {type(item)}')

  return '.'.join([f(item) for item in path]).replace('.[', '[')


def module_from_linen_variables(
    module_factory: Callable[[], M],
    variables: VariableDict,
    map_key_fn: Callable[[tuple[str, ...]], tuple[str | int, ...]] | None = None,
    assign_val_fn: Callable[[Any, tuple[str | int, ...], Any], Any] | None = None,
) -> M:
  """Returns an `nnx.Module` initialized with the `variables` of a linen module.

  Args:
    module_factory: A no-args callable that returns an `nnx.Module`.
    variables: A dictionary of variables.
    map_key_fn: An optional function for mapping keys in the `variables`
      dictionary to keys in the `nnx.Module`'s state. If not provided it is
      assumed that after removing the collection name the keys in the
      `variables` dictionary are the same as the keys in the `nnx.Module`'s
      state.
  """
  def _default_map_key_fn(path: tuple[str, ...]) -> tuple[str | int, ...]:
    return path[1:] if 'params' in variables else path

  def _default_assign_val_fn(
      state: Any,
      mapped_path: tuple[str | int, ...],
      val: Any,
  ) -> Any:
    state[mapped_path].set_value(val)
    return state

  key_mapper = map_key_fn if map_key_fn is not None else _default_map_key_fn
  val_assigner = assign_val_fn if assign_val_fn is not None else _default_assign_val_fn

  mdl: M = nnx.eval_shape(module_factory)
  graph_def, state = nnx.split(mdl)
  state = dict(nnx.to_flat_state(state))
  for path, val in traverse_util.flatten_dict(variables).items():
    mapped_path = key_mapper(path)
    if mapped_path not in state:
      raise ValueError(
          f"'{mdl.__class__.__name__}.{_flatten_path(mapped_path)}' doesn't "
          f' exist (original path={path}).'
      )
    state = val_assigner(state, mapped_path, val)
  state = nnx.from_flat_state(state)

  return nnx.merge(graph_def, state)
