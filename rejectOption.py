import numpy as np
import torch
import os

from torch.nn import functional as F
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

def get_device():
    """Get the device to use for the reject option

    Returns:
        torch.device: device to use
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


class RejectOption:
    def __init__(
        self,
        dataset: DataLoader = DataLoader([]),  # type: ignore
        model: torch.nn.Module = torch.nn.Module(),
        logits: torch.Tensor = torch.empty(0),
        repartition_type: str = "simple",
        metric_type: str = "max_2_diff",
    ):
        """init with Reject Option,
        Reject Option needs logits to be calibrated, so there is two ways to calibrate it :
        - provide a model and a dataset to calibrate it
        - provide logits to calibrate it

        Args:
            dataset (torch.utils.data.DataLoader): DataLoader of the calibration dataset
            model : model used to calibrate reject option
            logits (torch.Tensor): logits of the model output should be (batch_size, head_count, classes)
            repartition_type (str, optional): type of repartition function. Defaults to "simple". Possible values : ["simple", "per_class"]
            metric_type (str, optional): type of metric to use. Defaults to "max". Possible values : ["max", "max_2_diff", "simple_threshold"]
        """

        possible_metric_types = ["max", "max_2_diff", "simple_threshold"]

        if metric_type not in possible_metric_types:
            raise ValueError(f"type should be one of {possible_metric_types}")

        self.metric_type = metric_type
        self.device = get_device()

        self.calibration_set = torch.empty(0)

        # Calibration with logits
        if logits.shape[0] > 0:
            self.calibrate_with_logits(logits=logits, metric_type=metric_type)

        # Calibration with model and dataset
        elif len(dataset) > 0 and len(list(model.parameters())) > 0:
            self.model = model
            if next(model.parameters()).device != self.device:
                logger.warning(
                    f"Model is not on the same device as the RejectOption : {next(model.parameters()).device} vs {self.device}"
                )
                self.model = model.to(self.device)

            self.calibrate(dataset, model, metric_type)

        else:
            raise ValueError(
                "You should provide either logits or a dataset and a model to calibrate the reject option"
            )

        logger.debug(
            f"RejectOption Initialized with repartition_type : {repartition_type} and metric_type : {metric_type} shape of calibration : {self.calibration_set.shape}"
        )

    def calibrate_with_logits(
        self, logits: torch.Tensor, metric_type: str, max_calibration_size: int = 1000
    ):
        """Recalibrate the reject option with logits.

        Args:
            logits (torch.Tensor): logits of the model output should be (batch_size, head_count, classes)
        """
        logger.info(f"Calibrating RejectOption with logits of shape {logits.shape}")

        if len(logits.shape) == 2:
            logger.debug(
                f"Adding a dimension to logits : {logits.shape} -> {logits.unsqueeze(1).shape}"
            )
            logits = logits.unsqueeze(1)

        if logits.shape[0] > max_calibration_size:
            logger.debug(f"Truncating logits to {max_calibration_size} samples")
            logits = logits[:max_calibration_size]

        self.add_to_repartition_function(
            logits,
            metric_type=metric_type,
        )

    def calibrate(
        self,
        dataset: DataLoader,
        model: torch.nn.Module,
        metric_type: str,
        max_calibration_size: int = 1000,
    ):
        """Recalibrate the reject option with a new dataset.

        Args:
            dataset (torch.utils.data.DataLoader)): _description_
            model : model to use for classification. Defaults to self.model.
            metric_type (str, optional): type of metric to use. Defaults to "max". Possible values : ["max", "max_2_diff", "simple_threshold"]
            max_calibration_size (int, optional): max number of samples to use for calibration. Defaults to 1000.
        """
        model.eval()
        __transpose_flag = False
        # get one data to get the shape of the output
        images, _ = next(iter(dataset))
        images = images.to(self.device)
        logits = model(images)

        if (
            logits.shape[0] != dataset.batch_size
            and logits.shape[1] == dataset.batch_size
        ):
            logger.warning(
                f"Model seems to output data in shape (heads, batch_size, classes) instead of (batch_size, heads, classes).\
                    Transposing logits from {logits.shape} to {(logits.shape[1], logits.shape[0], logits.shape[2])}"
            )
            __transpose_flag = True

        for images, _ in tqdm(dataset):
            images = images.to(self.device)
            logits = model(images)

            if __transpose_flag:
                logits = logits.transpose(0, 1)

            self.add_to_repartition_function(
                logits,
                metric_type=metric_type,
            )
            if self.calibration_set.shape[-1] > max_calibration_size:
                break

    def add_to_repartition_function(self, logits: torch.Tensor, metric_type: str):
        """store in self.outputs the repartition function of the model output.

        Args:
            logits (torch.Tensor): logits of the model output should be (batch_size, head_count, classes)
            metric_type (str): type of metric to use. Defaults to "max". Possible values : ["max", "max_2_diff", "simple_threshold"]
        """
        self.calibration_set = torch.cat(
            (self.calibration_set, self.apply_type(logits, metric_type).values.detach().cpu()), -1
        ).detach().cpu()
        assert (
            len(self.calibration_set.shape) == 2
        ), f"Calibration set should be 2D (head, elements_calibrated) but is {self.calibration_set.shape}"

    def to_file(self, path: str):
        """save the repartition function to a file

        Args:
            path (str): path to the file
        """
        torch.save(self.calibration_set, path)
        logger.info(f"Calibration of the RejectOption saved to {path}")

    def apply_type(self, logits: torch.Tensor, type: str):
        """apply a type of metric to the logits.

        Args:
            logits (torch.Tensor): logits of the model output should be (batch_size, head_count, classes)
            type (str, optional): type of metric to use. Defaults to "". Possible values : ["max", "max_2_diff", "simple_threshold"]

        Returns:
            torch.Tensor: tensor of shape (batch_size, head_count, 1)
        """
        logits = logits.to(self.device)
        if type == "max" or type == "simple_threshold":
            return torch.max(logits, dim=-1)
        elif type == "max_2_diff":
            topk_tensor = torch.topk(logits, 2, dim=-1).values
            return torch.max(
                torch.sub(topk_tensor[..., 0], topk_tensor[..., 1]).unsqueeze(-1),
                dim=-1,
            )
        else:
            raise ValueError(f"Type {type} not supported")

    def is_rejected(
        self,
        logits: torch.Tensor,
        epsilon: torch.Tensor | float,
        head: int | None = None,
        repartition: torch.Tensor = torch.empty(0),
    ):
        """check if a value is rejected

        Args:
            value (float): float value to be checked, should be max probability of the model output
            epsilon (float): rejection threshold. example: epsilon=0.1 means 10% of the data is rejected
            head (int, optional): head to use for the rejection. Defaults to None.
            repartition (torch.Tensor, optional): calibration set can be provided if modified during inference. Defaults to torch.empty(0).

        Returns:
            bool: true if the value is rejected, false otherwise
        """

        if self.metric_type == "simple_threshold":
            criterion = torch.softmax(logits, dim=-1)

            is_rejected = criterion.values < torch.tensor(epsilon)
            return is_rejected.to("cpu")
        logits, _ = self.apply_type(logits, self.metric_type)
        # if repartition is not provided, use the calibration set and select the correct repartition from every head
        if repartition.shape[0] == 0 and head is not None:
            repartition = self.calibration_set[:, head]
        elif repartition.shape[0] == 0 and head is None:
            logger.warning(
                "repartition is empty and head is not provided, cannot select the correct repartition doing with empty repartition"
            )
            return torch.zeros(logits.shape[0], dtype=torch.bool).to("cpu")
        elif repartition.shape[0] != 0 and head is not None:
            logger.warning(
                "repartition is provided and head is provided, ignoring head"
            )
        elif repartition.shape[0] == 0 and self.calibration_set.shape[0] == 0:
            raise ValueError(
                "repartition is empty and calibration set is empty, looks like reject Option wasn't calibrated"
            )
        try:
            index = torch.searchsorted(torch.sort(repartition).values, logits).cpu()
        except NotImplementedError:
            # not implemented on mps backend so falling back to cpu
            index = torch.searchsorted(torch.sort(repartition).values.cpu(), logits.cpu())
        index = torch.clamp(index, 0, len(repartition) - 1)
        is_rejected = torch.linspace(0, 1, len(repartition) + 1)[index] < epsilon.cpu()

        return is_rejected

class LLMRejectOption(RejectOption):
    """RejectOption implemented specifically for NanoGPT LLM model (really specific implementation)."""

    def __init__(
        self,
        model: torch.nn.Module = None,
        metric_type: str = "max_2_diff",
        repartition_type: str = "simple",
    ):
        """Init the RejectOption.

        Args:
            model (torch.nn.Module): model to calibrate
            metric_type (str, optional): type of metric to use. Defaults to "max_2_diff". Possible values : ["max", "max_2_diff", "simple_threshold"]
            repartition_type (str, optional): type of repartition to use. Defaults to "simple". Possible values : ["simple", "per_class"]
        """
        possible_metric_types = ["max", "max_2_diff", "simple_threshold"]

        if metric_type not in possible_metric_types:
            raise ValueError(f"type should be one of {possible_metric_types}")

        self.metric_type = metric_type
        if model is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = get_device()
        self.calibration_set = torch.empty(0)

    @staticmethod
    def get_batch(
        data: np.ndarray, batch_size: int, block_size: int, device: torch.device
    ):
        """get a batch of data.

        Args:
            data (np.ndarray): data to get the batch from
            batch_size (int): size of the batch
            block_size (int): size of the block
            device (torch.device): device to use

        Returns:
            np.ndarray: batch of data
        """
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
                for i in ix
            ]
        )

        if device == torch.device("cuda"):
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def calibrate(
        self,
        data: np.ndarray,
        block_size: int,
        batch_size: int,
        max_calibration_size: int = 1000,
        model: torch.nn.Module = None,
    ):
        """Calibrate the RejectOption.

        Args:
            data (np.ndarray): data to calibrate the RejectOption
            block_size (int): block size of the model
            batch_size (int): batch size to use for the calibration
            max_calibration_size (int, optional): max size of the calibration set. Defaults to 1000.
            model (torch.nn.Module, optional): model to calibrate. Defaults to None.
        """
        if model is None:
            model = self.model

        model.eval()

        self.calibration_set = torch.empty(0)


        if max_calibration_size > len(data) - block_size:
            logger.warning(
                f"max_calibration_size is bigger than the dataset size, reducing it to {len(data) - block_size}"
            )
            max_calibration_size = len(data) - block_size
            
        while self.calibration_set.shape[-1] < max_calibration_size:
            X, Y = self.get_batch(data, batch_size, block_size, self.device)

            logits, _ = model(X, Y)
            logits = F.softmax(logits, dim=-1)
            last_word_logit = logits[:, :, -1, :]
            self.add_to_repartition_function(last_word_logit, self.metric_type)

    def eval_reject_option(self,
                           model: torch.nn.Module,
                           dataset: torch.utils.data.Dataset,
                           sampling: int = 10,
                           min_epsilon: float = 0.,
                           max_epsilon: float = 1.,
                           block_size: int = 512,
                           out_dir: str = "./",
                           plot_tag: str = "0",
                           eval_iters: int = 200):
        """evaluate the reject option on a dataset by using reject option on it while varying epsilon. create a plot of the results.
        
        Args:
            model (torch.nn.Module): model to calibrate
            dataset (torch.utils.data.Dataset): dataset to calibrate on
            sampling (int, optional): number of samples to use for the evaluation. Defaults to 10.
            min_epsilon (float, optional): minimum epsilon to use. Defaults to 0.
            max_epsilon (float, optional): maximum epsilon to use. Defaults to 1.
            out_dir (str, optional): directory to save the plot. Defaults to "./".
            plot_tag (str, optional): tag to use for the plot. Defaults to "0".
        """
        # TODO Use exponential weights aggregation
        
        # fix epsilon, iter on dataset, for each data point, calculate output based on reject option. save head and calculate loss. Plot epsilon/loss
        # TODO add thop to plot power usage/loss
        model.eval()
        out = {}
        losses = []
        # TODO better compute perplexity here instead of using the loss
        for epsilon in torch.linspace(min_epsilon, max_epsilon, sampling):
            # print log every 1/4 of the sampling
            if len(out) % (sampling // 4) == 0:
                print(f"evaluated {len(out)} epsilons")
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(dataset, 1, block_size, self.device)
                logits, loss = model(X, Y)
                logits = logits[:, :, -1, :]
                for head, proba_head in enumerate(logits):
                    if not self.is_rejected(proba_head, epsilon=epsilon, head=head):
                        break
                losses[k] = torch.exp(loss[head]).item()
            out[epsilon] = losses.mean()

        # plot and save epsilon/loss
        plt.plot(list(out.keys()), list(out.values()))
        plt.xlabel("epsilon")
        plt.ylabel("loss")
        plt.savefig(os.path.join(out_dir, f"loss_{plot_tag}.png"))
        plt.close()
        
        model.train()