# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from io import BytesIO
from typing import Optional

import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import TRANSFORMS, BaseTransform


@TRANSFORMS.register_module()
class LoadNpyFromFile(BaseTransform):
    """Load an npy from file.

    Required Keys:

    - npy_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty numpy array or
        file path not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.cvt_func = lambda x: np.ascontiguousarray(
            (x * 255.).round().astype(np.uint8).transpose(1, 2, 0))

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            with BytesIO(img_bytes) as f:
                img = np.load(f)
            if self.cvt_func is not None:
                img = self.cvt_func(img)

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f"cvt_func='{self.cvt_func}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str
