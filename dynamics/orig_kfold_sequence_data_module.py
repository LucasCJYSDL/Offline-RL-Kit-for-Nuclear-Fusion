"""
A data module that separates data into chronological folds and trains on
full sequences.

Adapted from FusionControl's KFoldSequenceFusionDataModule.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import os
import pickle as pkl

from dynamics_toolbox.utils.storage.qdata import load_from_hdf5
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from dynamics.data_util_compat import sort_by_continuous_snippets


class KFoldSequenceFusionDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        shot_train_length: int,
        min_shot_amt: int,
        n_folds: int,
        te_fold: int,
        file_name: str = "full.hdf5",
        prop_validation: float = 0.1,
        num_workers: int = 1,
        pin_memory: bool = True,
        seed: int = 1,
        save_shot_nums: bool = True,
        additional_file_names: Optional[Sequence] = None,
        val_shot_nums_path: str = None,
        train_from_start_only: bool = True,
        loaded_data=None,
        time_diff=None,
        remove_states: List[str] = [],
        remove_actuators: List[str] = [],
        bootstrap: bool = False,
        **kwargs,
    ):
        """Constructor."""
        del kwargs
        if n_folds > 1 and (te_fold < 1 or te_fold > n_folds):
            raise ValueError(f"te_fold must be either 1, ..., {n_folds}")
        super().__init__()
        with open(os.path.join(data_path, "info.pkl"), "rb") as handle:
            self.info = pkl.load(handle)
        np.random.seed(seed)
        self.shot_train_length = shot_train_length
        self.min_shot_amt = min_shot_amt
        self.train_from_start_only = train_from_start_only
        self.time_diff = time_diff
        if loaded_data is not None:
            data = loaded_data
        else:
            data = load_from_hdf5(os.path.join(data_path, file_name))

        state_idxs = [self.info["state_space"].index(state) for state in remove_states]
        if state_idxs:
            data["states"] = np.delete(data["states"], state_idxs, axis=1)
            data["next_states"] = np.delete(data["next_states"], state_idxs, axis=1)

        actuator_idxs = [
            self.info["actuator_space"].index(actuator) for actuator in remove_actuators
        ]
        if actuator_idxs:
            data["actuators"] = np.delete(data["actuators"], actuator_idxs, axis=1)
            data["next_actuators"] = np.delete(data["next_actuators"], actuator_idxs, axis=1)

        if additional_file_names is not None:
            for add_name in additional_file_names:
                additional_data = load_from_hdf5(os.path.join(data_path, add_name))
                for key, value in data.items():
                    if len(value.shape) == 1:
                        data[key] = np.concatenate([value, additional_data[key]])
                    else:
                        data[key] = np.vstack([value, additional_data[key]])
        if val_shot_nums_path is not None:
            val_shot_nums = np.load(val_shot_nums_path)
        else:
            val_shot_nums = None

        print("Bootstrapping: ", bootstrap)
        if bootstrap:
            shot_data = sort_by_continuous_snippets(
                data, info=self.info, min_amt_needed=self.min_shot_amt, dt=self.time_diff
            )
            self.te_shot_nums = self._get_test_shots(data, n_folds, te_fold)

            test_shot_data, other_shot_data = [], []
            for shot in shot_data:
                if shot["shotnum"][0] in self.te_shot_nums:
                    test_shot_data.append(shot)
                else:
                    other_shot_data.append(shot)

            durations = np.array([len(shot["shotnum"]) for shot in other_shot_data])
            probs = durations / durations.sum()
            bootstrap_sample = np.random.choice(
                other_shot_data, size=len(other_shot_data), replace=True, p=probs
            )
            shot_data = test_shot_data + list(bootstrap_sample)

            self.tr_shot_nums, self.val_shot_nums = self._separate_shot_nums_by_shotnumber(
                shot_data, prop_validation
            )
            self.tr_set, self.val_set, self.te_set = self._assemble_datasets_bootstrap(
                shot_data, [self.tr_shot_nums, self.val_shot_nums, self.te_shot_nums]
            )
        else:
            self.tr_shot_nums, self.val_shot_nums, self.te_shot_nums = self._separate_shot_nums(
                data, n_folds, te_fold, prop_validation, val_shot_nums
            )
            self.tr_set, self.val_set, self.te_set = self._assemble_datasets(
                data, [self.tr_shot_nums, self.val_shot_nums, self.te_shot_nums]
            )

        for name, dset in (("tr", self.tr_set), ("val", self.val_set), ("te", self.te_set)):
            setattr(
                self,
                f"_{name}_dataset",
                TensorDataset(torch.Tensor(dset[0]), torch.Tensor(dset[1]), torch.Tensor(dset[2])),
            )
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        print("Data Loaded in!")
        print(f"\t Number training shots: {len(self.tr_shot_nums)}")
        print(f"\t Number validation shots: {len(self.val_shot_nums)}")
        print(f"\t Number test shots: {len(self.te_shot_nums)}")
        if save_shot_nums:
            np.save("tr_shots.npy", np.array([shot for shot in self.tr_shot_nums]))
            np.save("val_shots.npy", np.array([shot for shot in self.val_shot_nums]))
            np.save("te_shots.npy", np.array([shot for shot in self.te_shot_nums]))

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            self._tr_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        if len(self._val_dataset):
            return DataLoader(
                self._val_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=self._pin_memory,
            )
        return None

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        if len(self._te_dataset):
            return DataLoader(
                self._te_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=self._pin_memory,
            )
        return None

    def _get_test_shots(self, data, n_folds, te_fold):
        shots = np.sort(list(set(data["shotnum"])))
        if n_folds == 1:
            te_shots = set([])
        else:
            shots_per_fold = len(shots) // n_folds
            lidx = int(shots_per_fold * (te_fold - 1))
            ridx = int(shots_per_fold * te_fold)
            te_shots = set(shots[lidx:ridx])
        return te_shots

    def _separate_shot_nums_by_shotnumber(self, shot_data, prop_validation: float) -> Sequence[set]:
        remaining_shots = []
        for shot in shot_data:
            if shot["shotnum"][0] not in self.te_shot_nums:
                remaining_shots.append(shot["shotnum"][0])
        remaining_shots = np.array(list(set(remaining_shots)))
        num_val = int(prop_validation * len(remaining_shots))

        if num_val <= 0:
            tr_shots = set(remaining_shots)
            val_shots = set([])
        else:
            val_idxs = np.random.choice(len(remaining_shots), size=num_val, replace=False)
            val_shots = set([shot for shot in remaining_shots[val_idxs]])
            tr_shots = set([shot for shot in remaining_shots if shot not in val_shots])
        return tr_shots, val_shots

    def _separate_shot_nums(
        self,
        data: Dict[str, np.ndarray],
        n_folds: int,
        te_fold: int,
        prop_validation: float,
        val_shot_nums: Optional[Sequence] = None,
    ) -> Sequence[set]:
        shots = np.sort(list(set(data["shotnum"])))
        if n_folds == 1:
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
                raise ValueError("Validation proportion too high.")
            val_idxs = np.random.choice(len(remaining_shots), size=num_val, replace=False)
            val_shots = set([shot for shot in remaining_shots[val_idxs]])
            tr_shots = set([shot for shot in remaining_shots if shot not in val_shots])
        if val_shot_nums is not None:
            val_shot_nums = set(val_shot_nums)
            tr_shots = tr_shots - val_shot_nums
            te_shots = te_shots - val_shot_nums
            val_shots = val_shots.union(val_shot_nums)
        return tr_shots, val_shots, te_shots

    def _assemble_datasets_bootstrap(self, shot_data, shot_nums: Sequence[set]) -> Tuple[np.ndarray, np.ndarray]:
        tr, val, te = [[[] for _ in range(3)] for _ in range(3)]
        for sdata in shot_data:
            snum = sdata["shotnum"][0]
            if snum in shot_nums[0]:
                to_add = tr
            elif snum in shot_nums[1]:
                to_add = val
            else:
                to_add = te
            self._append_sequence_snippets(to_add, sdata)
        for dt in [tr, val, te]:
            for didx, dlist in enumerate(dt):
                dt[didx] = np.array(dlist)
                if len(dt[didx].shape) < 3:
                    dt[didx] = dt[didx][..., np.newaxis]
        return tr, val, te

    def _assemble_datasets(
        self,
        data: Dict[str, np.ndarray],
        shot_nums: Sequence[set],
    ) -> Tuple[np.ndarray, np.ndarray]:
        tr, val, te = [[[] for _ in range(3)] for _ in range(3)]
        shot_data = sort_by_continuous_snippets(
            data, self.info, min_amt_needed=self.min_shot_amt, dt=self.time_diff
        )
        for sdata in shot_data:
            snum = sdata["shotnum"][0]
            if snum in shot_nums[0]:
                to_add = tr
            elif snum in shot_nums[1]:
                to_add = val
            else:
                to_add = te
            self._append_sequence_snippets(to_add, sdata)
        for dt in [tr, val, te]:
            for didx, dlist in enumerate(dt):
                dt[didx] = np.array(dlist)
                if len(dt[didx].shape) < 3:
                    dt[didx] = dt[didx][..., np.newaxis]
        return tr, val, te

    def _append_sequence_snippets(self, to_add, sdata) -> None:
        if self.train_from_start_only or len(sdata["states"]) <= self.shot_train_length:
            pad_needed = max(self.shot_train_length - len(sdata["states"]), 0)
            to_add[0].append(
                np.concatenate(
                    [
                        np.concatenate(
                            [sdata["states"], sdata["actuators"], sdata["next_actuators"]],
                            axis=-1,
                        ),
                        np.zeros(
                            (
                                pad_needed,
                                sdata["states"].shape[1]
                                + sdata["actuators"].shape[1]
                                + sdata["next_actuators"].shape[1],
                            )
                        ),
                    ],
                    axis=0,
                )[: self.shot_train_length]
            )
            to_add[1].append(
                np.concatenate(
                    [
                        sdata["next_states"],
                        np.zeros((pad_needed, sdata["next_states"].shape[1])),
                    ],
                    axis=0,
                )[: self.shot_train_length]
            )
            to_add[-1].append(
                np.append(
                    np.ones((min(len(sdata["states"]), self.shot_train_length), 1)),
                    np.zeros((pad_needed, 1)),
                )
            )
        else:
            num_snippets = len(sdata["states"]) - self.shot_train_length
            for ns in range(num_snippets):
                to_add[0].append(
                    np.concatenate(
                        [
                            sdata["states"][ns : ns + self.shot_train_length],
                            sdata["actuators"][ns : ns + self.shot_train_length],
                            sdata["next_actuators"][ns : ns + self.shot_train_length],
                        ],
                        axis=-1,
                    )
                )
                to_add[1].append(sdata["next_states"][ns : ns + self.shot_train_length])
                to_add[-1].append(np.ones((self.shot_train_length, 1)))

    @property
    def data(self) -> Sequence[np.array]:
        return self.tr_set[:2]

    @property
    def input_data(self) -> np.array:
        return self.tr_set[0]

    @property
    def output_data(self) -> np.array:
        return self.tr_set[1]

    @property
    def input_dim(self) -> int:
        return int(self.tr_set[0].shape[-1])

    @property
    def output_dim(self) -> int:
        return int(self.tr_set[1].shape[-1])

    @property
    def num_train(self) -> int:
        return len(self.tr_set[0])

    @property
    def num_validation(self) -> int:
        return len(self.val_set[0])

    @property
    def num_test(self) -> int:
        return len(self.te_set[0])
