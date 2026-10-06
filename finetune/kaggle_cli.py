#!/usr/bin/env python
"""The Kaggle CLI, routed through www.kaggle.com instead of api.kaggle.com.

Some networks (e.g. campus Wi-Fi) block the ``api.kaggle.com`` host that the
Kaggle client uses, while ``www.kaggle.com`` works. Kaggle serves the same API
under ``https://www.kaggle.com/api/v1/...``, so this wrapper just changes the
host and runs the normal CLI. Usage is identical to ``kaggle``:

    python finetune/kaggle_cli.py kernels list --mine
"""

import sys

from kagglesdk import kaggle_env

# In PROD the client builds URLs as f"{endpoint}/v1/{service}/{method}".
kaggle_env._env_to_endpoint[kaggle_env.KaggleEnv.PROD] = "https://www.kaggle.com/api"

from kaggle.cli import main  # noqa: E402  (import after patching)

if __name__ == "__main__":
    sys.exit(main())
