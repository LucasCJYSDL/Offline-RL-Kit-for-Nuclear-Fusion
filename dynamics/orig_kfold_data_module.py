"""
A data module that will separate the data into different folds. This data module

Each fold is separated by contiguous shot numbers.
"""
from typing import Dict, Union, List, Sequence, Tuple, Optional
import os

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from dynamics_toolbox.utils.storage.qdata import load_from_hdf5


class KFoldFusionDataModule(LightningDataModule):

    def __init__(
            self,
            data_path: str,
            batch_size: int,
            n_folds: int,
            te_fold: int,
            file_name: str = 'full.hdf5',
            prop_validation: float = 0.1,
            num_workers: int = 1,
            pin_memory: bool = True,
            seed: int = 1,
            save_shot_nums: bool = True,
            additional_file_names: Optional[Sequence] = None,
            val_shot_nums: Optional[Sequence] = None,
            loaded_data=None,
            **kwargs,
    ):
        """Constructor.

        Args:
            data_path: Path to the directory containing tr.hdf, te.hdf5 etc.
            batch_size: Batch size.
            n_folds: The number of folds to separate the data into. If the number of
                folds is 1. Then no test fold will be allocated.
            te_fold: The fold to hold out for testing, can take values {1, ..., n_folds}
            file_name: The file name of the data that should be loaded in.
            prop_validation: The proporton of the total data to be used for validation.
                This is important for preventing overfitting.
            num_workers: Number of workers.
            pin_memory: Whether to pin memory.
            seed: The seed. The randomness here is the selection of the validation
                shots.
            save_shot_nums: Whether to save the shot numbers used for training,
                validation, and testing. These are saved in the current working
                directory.
            additional_file_names: More file names that should be included on top
                of the file_name.
            val_shot_nums: Shot numbers that should be included in the validation set.
                These will be included on top of the other validation set that is
                formed.
            loaded_data: The previously loaded data to use instead of loading in more
                data.
        """
        del kwargs
        if n_folds > 1 and (te_fold < 1 or te_fold > n_folds):
            raise ValueError(f'te_fold must be either 1, ..., {n_folds}')
        super().__init__()
        np.random.seed(seed)
        if loaded_data is None:
            data = load_from_hdf5(os.path.join(data_path, file_name))
        else:
            data = loaded_data
        if additional_file_names is not None:
            for add_name in additional_file_names:
                additional_data = load_from_hdf5(os.path.join(data_path, add_name))
                for k, v in data.items():
                    if len(v.shape) == 1:
                        data[k] = np.concatenate([v, additional_data[k]])
                    else:
                        data[k] = np.vstack([v, additional_data[k]])
        self.tr_shot_nums, self.val_shot_nums, self.te_shot_nums = \
            self._separate_shot_nums(data, n_folds, te_fold, prop_validation,
                                     val_shot_nums)
        self.tr_x, self.tr_y = self._assemble_dataset(data, self.tr_shot_nums)
        self.val_x, self.val_y = self._assemble_dataset(data, self.val_shot_nums)
        self.te_x, self.te_y = self._assemble_dataset(data, self.te_shot_nums)
        self._tr_dataset = TensorDataset(torch.Tensor(self.tr_x),
                                         torch.Tensor(self.tr_y))
        self._te_dataset = TensorDataset(torch.Tensor(self.te_x),
                                         torch.Tensor(self.te_y))
        self._val_dataset = TensorDataset(torch.Tensor(self.val_x),
                                          torch.Tensor(self.val_y))
        # Load the validation data.
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        print('Data Loaded in!')
        print(f'\t Number training shots: {len(self.tr_shot_nums)}')
        print(f'\t Number validation shots: {len(self.val_shot_nums)}')
        print(f'\t Number test shots: {len(self.te_shot_nums)}')
        if save_shot_nums:
            np.save('tr_shots.npy', np.array([si for si in self.tr_shot_nums]))
            np.save('val_shots.npy', np.array([si for si in self.val_shot_nums]))
            np.save('te_shots.npy', np.array([si for si in self.te_shot_nums]))

    def train_dataloader(self) ->\
            Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Get the training dataloader."""
        return DataLoader(
            self._tr_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) ->\
            Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Get the training dataloader."""
        if len(self._val_dataset):
            return DataLoader(
                self._val_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=self._pin_memory,
            )
        else:
            None

    def test_dataloader(self) ->\
            Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Get the training dataloader."""
        if len(self._te_dataset):
            return DataLoader(
                self._te_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=self._pin_memory,
            )
        else:
            None

    def _separate_shot_nums(
            self,
            data: Dict[str, np.ndarray],
            n_folds: int,
            te_fold: int,
            prop_validation: float,
            val_shot_nums: Optional[Sequence] = None,
    ) -> Sequence[set]:
        """Separate all of the shot numbers into a train, validation, and test set.

        Args:
            data: The full dataset loaded in.
            n_folds: The number of folds to separate the data into.
            te_fold: The fold to hold out for testing, can take values {1, ..., n_folds}
            val_shot_nums: Shot numbers that should be included in the validation set.
                These will be included on top of the other validation set that is
                formed.

        Returns:
            The train, validation, and test shots.
        """
        shots = np.sort(list(set(data['shotnum'])))
        if n_folds == 1:  # In this case there are no test shots.
            te_shots = set([])
            remaining_shots = shots
        else:
            shots_per_fold = len(shots) // n_folds
            lidx = int(shots_per_fold * (te_fold - 1))
            ridx = int(shots_per_fold * te_fold)
            te_shots = set(shots[lidx:ridx])
            if te_fold == 1:
                remaining_shots = shots[ridx:]
            elif te_fold == n_folds:
                remaining_shots = shots[:lidx]
            else:
                remaining_shots = np.concatenate([shots[:lidx], shots[ridx:]])
        if prop_validation <= 0:
            tr_shots = set(remaining_shots)
            val_shots = set([])
        else:
            num_val = int(prop_validation * len(shots))
            if num_val > len(remaining_shots):
                raise ValueError('Validation proportion too high.')
            val_idxs = np.random.choice(len(remaining_shots), size=num_val,
                                        replace=False)
            val_shots = set([rs for rs in remaining_shots[val_idxs]])
            tr_shots = set([rs for rs in remaining_shots if rs not in val_shots])
        if val_shot_nums is not None:
            val_shot_nums = set(val_shot_nums)
            tr_shots = tr_shots - val_shot_nums
            te_shots = te_shots - val_shot_nums
            val_shots = val_shots.union(val_shot_nums)
        return tr_shots, val_shots, te_shots

    def _assemble_dataset(
            self,
            data: Dict[str, np.ndarray],
            shot_nums: set,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assembe the dataset.

        Args:
            data: The dataset loaded in.

        Returns:
            The x data and the y data.
        """
        if len(shot_nums) == 0:
            return np.array([]), np.array([])
        row_idxs = [idx for idx, sn in enumerate(data['shotnum']) if sn in shot_nums]
        x_data = np.hstack([
            data['states'][row_idxs],
            data['actuators'][row_idxs],
            data['next_actuators'][row_idxs],
        ])
        return x_data, data['next_states'][row_idxs]

    @property
    def data(self) -> Sequence[np.array]:
        """Get all of the data."""
        return self.tr_x, self.tr_y

    @property
    def input_data(self) -> np.array:
        """The input data.."""
        return self.tr_x

    @property
    def output_data(self) -> np.array:
        """The output data."""
        return self.tr_y

    @property
    def input_dim(self) -> int:
        """Observation dimension."""
        return self.tr_x.shape[-1]

    @property
    def output_dim(self) -> int:
        return self.tr_y.shape[-1]

    @property
    def num_train(self) -> int:
        """Number of training points."""
        return len(self.tr_x)

    @property
    def num_validation(self) -> int:
        """Number of training points."""
        return len(self.val_x)

    @property
    def num_test(self) -> int:
        """Number of training points."""
        return len(self.te_x)
