import torch
import logging

logger = logging.getLogger(__name__)


class DeviceManager:

    @staticmethod
    def get_device() -> str:
        """
        Returns:
            cuda
            mps (Mac)
            cpu
        """

        if torch.cuda.is_available():
            device = "cuda"

            gpu_name = torch.cuda.get_device_name(0)

            logger.info(
                f"Using CUDA GPU: {gpu_name}"
            )

        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = "mps"

            logger.info(
                "Using Apple Metal (MPS)"
            )

        else:
            device = "cpu"

            logger.warning(
                "CUDA not available. Using CPU."
            )

        return device