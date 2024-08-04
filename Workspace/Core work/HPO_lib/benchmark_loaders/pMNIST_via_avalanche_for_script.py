import numpy as np

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.datasets.external_datasets.mnist import get_mnist_dataset
from avalanche.benchmarks.utils import (
    _make_taskaware_classification_dataset,
    DefaultTransformGroups,
)
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks import split_validation_random


from typing import Sequence, Union
import torch
from torch.utils.data import DataLoader
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import (
    ToTensor,
    ToPILImage,
    Compose,
    Normalize,
)



class PixelsPermutation(object):
    """
    Apply a fixed permutation to the pixels of the given image.

    Works with both Tensors and PIL images. Returns an object of the same type
    of the input element.
    """

    def __init__(self, index_permutation: Sequence[int]):
        self.permutation = index_permutation
        self._to_tensor = ToTensor()
        self._to_image = ToPILImage()

    def __call__(self, img: Union[Image, Tensor]):
        is_image = isinstance(img, Image)
        if (not is_image) and (not isinstance(img, Tensor)):
            raise ValueError("Invalid input: must be a PIL image or a Tensor")

        image_as_tensor: Tensor
        if is_image:
            image_as_tensor = self._to_tensor(img)
        else:
            image_as_tensor = img

        image_as_tensor = image_as_tensor.view(-1)[self.permutation].view(
            *image_as_tensor.shape
        )

        if is_image:
            img = self._to_image(image_as_tensor)
        else:
            img = image_as_tensor

        return img



def PermutedMNIST(n_experiences,train_percentage,difficulty="easy",batch_size=128, *,
                  return_task_id=False,seed=None, global_seed=88,
                  dataset_root = None) :

    # Reproducibility
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

    list_train_dataset = []
    list_val_dataset = []
    val_loaders_list = []
    test_loaders_list = []
    rng_permute = np.random.RandomState(seed)

    mnist_train, mnist_test = get_mnist_dataset(dataset_root)

    # for every incremental experience
    for _ in range(n_experiences):
        # choose a random permutation of the pixels in the image
        if difficulty == "easy":
            idx_permute = torch.arange(784)
            easy_permute = torch.from_numpy(rng_permute.permutation(8*8))
            easy_indices = torch.concat([idx_permute.view(28,28)[i][10:18] for i in range(10,18)])
            easily_permuted_indices = easy_indices[easy_permute]
            idx_permute[easy_indices] = easily_permuted_indices
        else :
            idx_permute = torch.from_numpy(rng_permute.permutation(784)).type(torch.int64)

        permutation = PixelsPermutation(idx_permute)

        _default_mnist_transform = Compose([Normalize((0.1307,), (0.3081,)),
                                                  permutation])

        # Freeze the permutation
        permuted_train = make_avalanche_dataset(
            _make_taskaware_classification_dataset(mnist_train),
            frozen_transform_groups=DefaultTransformGroups((_default_mnist_transform, None)),
        )
        permuted_train, permuted_val = split_validation_random(1-train_percentage, shuffle=False, dataset=permuted_train)

        permuted_test = make_avalanche_dataset(
            _make_taskaware_classification_dataset(mnist_test),
            frozen_transform_groups=DefaultTransformGroups((_default_mnist_transform, None)),
        )

        list_train_dataset.append(permuted_train)
        list_val_dataset.append(permuted_val)
        val_loader = DataLoader(permuted_val, batch_size=batch_size, shuffle=True)
        val_loaders_list.append(val_loader)
        test_loader = DataLoader(permuted_test, batch_size=batch_size, shuffle=True)
        test_loaders_list.append(test_loader)

        train_loaders_list = nc_benchmark(
                                list_train_dataset,
                                list_val_dataset,
                                n_experiences=len(list_train_dataset),
                                task_labels=return_task_id,
                                shuffle=False,
                                class_ids_from_zero_in_each_exp=True,
                                one_dataset_per_exp=True,
                                train_transform=None,
                                eval_transform=None,
                            )
    return train_loaders_list, val_loaders_list, test_loaders_list






def train_and_val_PermutedMNIST(n_experiences,train_percentage,difficulty="easy",batch_size=128, *,
                  return_task_id=False,seed=None, global_seed=88,
                  dataset_root = None) :

    # Reproducibility
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

    list_train_dataset = []
    list_val_dataset = []
    val_loaders_list = []
    rng_permute = np.random.RandomState(seed)

    mnist_train, _ = get_mnist_dataset(dataset_root)

    # for every incremental experience
    for _ in range(n_experiences):
        # choose a random permutation of the pixels in the image
        if difficulty == "easy":
            idx_permute = torch.arange(784)
            easy_permute = torch.from_numpy(rng_permute.permutation(8*8))
            easy_indices = torch.concat([idx_permute.view(28,28)[i][10:18] for i in range(10,18)])
            easily_permuted_indices = easy_indices[easy_permute]
            idx_permute[easy_indices] = easily_permuted_indices
        else :
            idx_permute = torch.from_numpy(rng_permute.permutation(784)).type(torch.int64)

        permutation = PixelsPermutation(idx_permute)

        _default_mnist_transform = Compose([Normalize((0.1307,), (0.3081,)),
                                                  permutation])

        # Freeze the permutation
        permuted_train = make_avalanche_dataset(
            _make_taskaware_classification_dataset(mnist_train),
            frozen_transform_groups=DefaultTransformGroups((_default_mnist_transform, None)),
        )
        permuted_train, permuted_val = split_validation_random(1-train_percentage, shuffle=False, dataset=permuted_train)

        list_train_dataset.append(permuted_train)
        list_val_dataset.append(permuted_val)
        val_loader = DataLoader(permuted_val, batch_size=batch_size, shuffle=True)
        val_loaders_list.append(val_loader)

        train_loaders_list = nc_benchmark(
                                list_train_dataset,
                                list_val_dataset,
                                n_experiences=len(list_train_dataset),
                                task_labels=return_task_id,
                                shuffle=False,
                                class_ids_from_zero_in_each_exp=True,
                                one_dataset_per_exp=True,
                                train_transform=None,
                                eval_transform=None,
                            )
    return train_loaders_list, val_loaders_list