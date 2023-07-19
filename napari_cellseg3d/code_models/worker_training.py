import platform
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch

# MONAI
from monai.data import (
    CacheDataset,
    DataLoader,
    PatchDataset,
    decollate_batch,
    pad_list_data_collate,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImaged,
    # NormalizeIntensityd,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    SpatialPadd,
)
from monai.utils import set_determinism

# Qt
from napari.qt.threading import GeneratorWorker

# local
from napari_cellseg3d import config, utils
from napari_cellseg3d.code_models.workers_utils import (
    PRETRAINED_WEIGHTS_DIR,
    LogSignal,
    QuantileNormalizationd,
    TrainingReport,
    WeightsDownloader,
)

logger = utils.LOGGER
VERBOSE_SCHEDULER = True
logger.debug(f"PRETRAINED WEIGHT DIR LOCATION : {PRETRAINED_WEIGHTS_DIR}")

"""
Writing something to log messages from outside the main thread needs specific care,
Following the instructions in the guides below to have a worker with custom signals,
a custom worker function was implemented.
"""

# https://python-forum.io/thread-31349.html
# https://www.pythoncentral.io/pysidepyqt-tutorial-creating-your-own-signals-and-slots/
# https://napari-staging-site.github.io/guides/stable/threading.html


class TrainingWorker(GeneratorWorker):
    """A custom worker to run training jobs in.
    Inherits from :py:class:`napari.qt.threading.GeneratorWorker`"""

    def __init__(
        self,
        worker_config: config.TrainingWorkerConfig,
    ):
        """Initializes a worker for inference with the arguments needed by the :py:func:`~train` function. Note: See :py:func:`~train`

        Args:
            * device : device to train on, cuda or cpu

            * model_dict : dict containing the model's "name" and "class"

            * weights_path : path to weights files if transfer learning is to be used

            * data_dicts : dict from :py:func:`Trainer.create_train_dataset_dict`

            * validation_percent : percentage of images to use as validation

            * max_epochs : the amout of epochs to train for

            * loss_function : the loss function to use for training

            * learning_rate : the learning rate of the optimizer

            * val_interval : the interval at which to perform validation (e.g. if 2 will validate once every 2 epochs.) Also determines frequency of saving, depending on whether the metric is better or not

            * batch_size : the batch size to use for training

            * results_path : the path to save results in

            * sampling : whether to extract patches from images or not

            * num_samples : the number of samples to extract from an image for training

            * sample_size : the size of the patches to extract when sampling

            * do_augmentation : whether to perform data augmentation or not

            * deterministic : dict with "use deterministic" : bool, whether to use deterministic training, "seed": seed for RNG


        """
        super().__init__(self.train)
        self._signals = LogSignal()
        self.log_signal = self._signals.log_signal
        self.warn_signal = self._signals.warn_signal
        self.error_signal = self._signals.error_signal

        self._weight_error = False
        #############################################
        self.config = worker_config

        self.train_files = []
        self.val_files = []
        #######################################
        self.downloader = WeightsDownloader()

    def set_download_log(self, widget):
        self.downloader.log_widget = widget

    def log(self, text):
        """Sends a signal that ``text`` should be logged

        Args:
            text (str): text to logged
        """
        self.log_signal.emit(text)

    def warn(self, warning):
        """Sends a warning to main thread"""
        self.warn_signal.emit(warning)

    def raise_error(self, exception, msg):
        """Sends an error to main thread"""
        logger.error(msg, exc_info=True)
        logger.error(exception, exc_info=True)
        self.error_signal.emit(exception, msg)
        self.errored.emit(exception)
        self.quit()

    def log_parameters(self):
        self.log("-" * 20)
        self.log("Parameters summary :\n")

        self.log(
            f"Percentage of dataset used for validation : {self.config.validation_percent * 100}%"
        )

        self.log("-" * 10)
        self.log("Training files :\n")
        [
            self.log(f"{Path(train_file['image']).name}\n")
            for train_file in self.train_files
        ]
        self.log("-" * 10)
        self.log("Validation files :\n")
        [
            self.log(f"{Path(val_file['image']).name}\n")
            for val_file in self.val_files
        ]
        self.log("-" * 10)

        if self.config.deterministic_config.enabled:
            self.log("Deterministic training is enabled")
            self.log(f"Seed is {self.config.deterministic_config.seed}")

        self.log(f"Training for {self.config.max_epochs} epochs")
        self.log(f"Loss function is : {str(self.config.loss_function)}")
        self.log(
            f"Validation is performed every {self.config.validation_interval} epochs"
        )
        self.log(f"Batch size is {self.config.batch_size}")
        self.log(f"Learning rate is {self.config.learning_rate}")

        if self.config.sampling:
            self.log(
                f"Extracting {self.config.num_samples} patches of size {self.config.sample_size}"
            )
        else:
            self.log("Using whole images as dataset")

        if self.config.do_augmentation:
            self.log("Data augmentation is enabled")

        if not self.config.weights_info.use_pretrained:
            self.log(f"Using weights from : {self.config.weights_info.path}")
            if self._weight_error:
                self.log(
                    ">>>>>>>>>>>>>>>>>\n"
                    "WARNING:\nChosen weights were incompatible with the model,\n"
                    "the model will be trained from random weights\n"
                    "<<<<<<<<<<<<<<<<<\n"
                )

        # self.log("\n")
        self.log("-" * 20)

    def train(self):
        """Trains the PyTorch model for the given number of epochs, with the selected model and data,
        using the chosen batch size, validation interval, loss function, and number of samples.
        Will perform validation once every :py:obj:`val_interval` and save results if the mean dice is better

        Requires:

        * device : device to train on, cuda or cpu

        * model_dict : dict containing the model's "name" and "class"

        * weights_path : path to weights files if transfer learning is to be used

        * data_dicts : dict from :py:func:`Trainer.create_train_dataset_dict`

        * validation_percent : percentage of images to use as validation

        * max_epochs : the amount of epochs to train for

        * loss_function : the loss function to use for training

        * learning rate : the learning rate of the optimizer

        * val_interval : the interval at which to perform validation (e.g. if 2 will validate once every 2 epochs.) Also determines frequency of saving, depending on whether the metric is better or not

        * batch_size : the batch size to use for training

        * results_path : the path to save results in

        * sampling : whether to extract patches from images or not

        * num_samples : the number of samples to extract from an image for training

        * sample_size : the size of the patches to extract when sampling

        * do_augmentation : whether to perform data augmentation or not

        * deterministic : dict with "use deterministic" : bool, whether to use deterministic training, "seed": seed for RNG
        """

        #########################
        # error_log = open(results_path +"/error_log.log" % multiprocessing.current_process().name, 'x')
        # faulthandler.enable(file=error_log, all_threads=True)
        #########################
        model_config = self.config.model_info
        weights_config = self.config.weights_info
        deterministic_config = self.config.deterministic_config

        start_time = time.time()

        try:
            if deterministic_config.enabled:
                set_determinism(
                    seed=deterministic_config.seed
                )  # use_deterministic_algorithms = True causes cuda error

            sys = platform.system()
            logger.debug(sys)
            if sys == "Darwin":  # required for macOS ?
                torch.set_num_threads(1)
                self.log("Number of threads has been set to 1 for macOS")

            self.log(f"config model : {self.config.model_info.name}")
            model_name = model_config.name
            model_class = model_config.get_model()

            if not self.config.sampling:
                data_check = LoadImaged(keys=["image"])(
                    self.config.train_data_dict[0]
                )
                check = data_check["image"].shape

            do_sampling = self.config.sampling

            size = self.config.sample_size if do_sampling else check

            PADDING = utils.get_padding_dim(size)
            model = model_class(  # FIXME check if correct
                input_img_size=PADDING, use_checkpoint=True
            )
            model = model.to(self.config.device)

            epoch_loss_values = []
            val_metric_values = []

            if len(self.config.train_data_dict) > 1:
                self.train_files, self.val_files = (
                    self.config.train_data_dict[
                        0 : int(
                            len(self.config.train_data_dict)
                            * self.config.validation_percent
                        )
                    ],
                    self.config.train_data_dict[
                        int(
                            len(self.config.train_data_dict)
                            * self.config.validation_percent
                        ) :
                    ],
                )
            else:
                self.train_files = self.val_files = self.config.train_data_dict
                msg = f"Only one image file was provided : {self.config.train_data_dict[0]['image']}.\n"

                logger.debug(f"SAMPLING is {self.config.sampling}")
                if not self.config.sampling:
                    msg += "Sampling is not in use, the only image provided will be used as the validation file."
                    self.warn(msg)
                else:
                    msg += "Samples for validation will be cropped for the same only volume that is being used for training"

                logger.warning(msg)

            logger.debug(
                f"Data dict from config is {self.config.train_data_dict}"
            )
            logger.debug(f"Train files : {self.train_files}")
            logger.debug(f"Val. files : {self.val_files}")

            if len(self.train_files) == 0:
                raise ValueError("Training dataset is empty")
            if len(self.val_files) == 0:
                raise ValueError("Validation dataset is empty")

            if self.config.do_augmentation:
                train_transforms = (
                    Compose(  # TODO : figure out which ones and values ?
                        [
                            RandShiftIntensityd(keys=["image"], offsets=0.7),
                            Rand3DElasticd(
                                keys=["image", "label"],
                                sigma_range=(0.3, 0.7),
                                magnitude_range=(0.3, 0.7),
                            ),
                            RandFlipd(keys=["image", "label"]),
                            RandRotate90d(keys=["image", "label"]),
                            RandAffined(
                                keys=["image", "label"],
                            ),
                            EnsureTyped(keys=["image", "label"]),
                        ]
                    )
                )
            else:
                train_transforms = EnsureTyped(keys=["image", "label"])

            val_transforms = Compose(
                [
                    # LoadImaged(keys=["image", "label"]),
                    # EnsureChannelFirstd(keys=["image", "label"]),
                    EnsureTyped(keys=["image", "label"]),
                ]
            )

            # self.log("Loading dataset...\n")
            def get_loader_func(num_samples):
                return Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        QuantileNormalizationd(keys=["image"]),
                        RandSpatialCropSamplesd(
                            keys=["image", "label"],
                            roi_size=(
                                self.config.sample_size
                            ),  # multiply by axis_stretch_factor if anisotropy
                            # max_roi_size=(120, 120, 120),
                            random_size=False,
                            num_samples=num_samples,
                        ),
                        Orientationd(keys=["image", "label"], axcodes="PLI"),
                        SpatialPadd(
                            keys=["image", "label"],
                            spatial_size=(
                                utils.get_padding_dim(self.config.sample_size)
                            ),
                        ),
                        EnsureTyped(keys=["image", "label"]),
                    ]
                )

            if do_sampling:
                # if there is only one volume, split samples
                # TODO(cyril) : maybe implement something in user config to toggle this behavior
                if len(self.config.train_data_dict) < 2:
                    num_train_samples = ceil(
                        self.config.num_samples
                        * self.config.validation_percent
                    )
                    num_val_samples = ceil(
                        self.config.num_samples
                        * (1 - self.config.validation_percent)
                    )
                    sample_loader_train = get_loader_func(num_train_samples)
                    sample_loader_eval = get_loader_func(num_val_samples)
                else:
                    num_train_samples = (
                        num_val_samples
                    ) = self.config.num_samples

                    sample_loader_train = get_loader_func(num_train_samples)
                    sample_loader_eval = get_loader_func(num_val_samples)

                logger.debug(f"AMOUNT of train samples : {num_train_samples}")
                logger.debug(
                    f"AMOUNT of validation samples : {num_val_samples}"
                )

                logger.debug("train_ds")
                train_ds = PatchDataset(
                    data=self.train_files,
                    transform=train_transforms,
                    patch_func=sample_loader_train,
                    samples_per_image=num_train_samples,
                )
                logger.debug("val_ds")
                val_ds = PatchDataset(
                    data=self.val_files,
                    transform=val_transforms,
                    patch_func=sample_loader_eval,
                    samples_per_image=num_val_samples,
                )

            else:
                load_whole_images = Compose(
                    [
                        LoadImaged(
                            keys=["image", "label"],
                            # image_only=True,
                            # reader=WSIReader(backend="tifffile")
                        ),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        Orientationd(keys=["image", "label"], axcodes="PLI"),
                        QuantileNormalizationd(keys=["image"]),
                        SpatialPadd(
                            keys=["image", "label"],
                            spatial_size=PADDING,
                        ),
                        EnsureTyped(keys=["image", "label"]),
                    ]
                )
                logger.debug("Cache dataset : train")
                train_ds = CacheDataset(
                    data=self.train_files,
                    transform=Compose(load_whole_images, train_transforms),
                )
                logger.debug("Cache dataset : val")
                val_ds = CacheDataset(
                    data=self.val_files, transform=load_whole_images
                )
            logger.debug("Dataloader")
            train_loader = DataLoader(
                train_ds,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=pad_list_data_collate,
            )

            val_loader = DataLoader(
                val_ds, batch_size=self.config.batch_size, num_workers=2
            )
            logger.info("\nDone")

            logger.debug("Optimizer")
            optimizer = torch.optim.Adam(
                model.parameters(), self.config.learning_rate
            )

            factor = self.config.scheduler_factor
            if factor >= 1.0:
                self.log(f"Warning : scheduler factor is {factor} >= 1.0")
                self.log("Setting it to 0.5")
                factor = 0.5

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=factor,
                patience=self.config.scheduler_patience,
                verbose=VERBOSE_SCHEDULER,
            )
            dice_metric = DiceMetric(
                include_background=False, reduction="mean"
            )

            best_metric = -1
            best_metric_epoch = -1

            # time = utils.get_date_time()
            logger.debug("Weights")

            if weights_config.custom:
                if weights_config.use_pretrained:
                    weights_file = model_class.weights_file
                    self.downloader.download_weights(model_name, weights_file)
                    weights = PRETRAINED_WEIGHTS_DIR / Path(weights_file)
                    weights_config.path = weights
                else:
                    weights = str(Path(weights_config.path))

                try:
                    model.load_state_dict(
                        torch.load(
                            weights,
                            map_location=self.config.device,
                        )
                    )
                except RuntimeError as e:
                    logger.error(f"Error when loading weights : {e}")
                    logger.exception(e)
                    warn = (
                        "WARNING:\nIt'd seem that the weights were incompatible with the model,\n"
                        "the model will be trained from random weights"
                    )
                    self.log(warn)
                    self.warn(warn)
                    self._weight_error = True

            if "cuda" in self.config.device:
                device_id = self.config.device.split(":")[-1]
                self.log("\nUsing GPU :")
                self.log(torch.cuda.get_device_name(int(device_id)))
            else:
                self.log("Using CPU")

            self.log_parameters()

            device = torch.device(self.config.device)

            # if model_name == "test":
            #     self.quit()
            #     yield TrainingReport(False)

            for epoch in range(self.config.max_epochs):
                # self.log("\n")
                self.log("-" * 10)
                self.log(f"Epoch {epoch + 1}/{self.config.max_epochs}")
                if device.type == "cuda":
                    self.log("Memory Usage:")
                    alloc_mem = round(
                        torch.cuda.memory_allocated(0) / 1024**3, 1
                    )
                    reserved_mem = round(
                        torch.cuda.memory_reserved(0) / 1024**3, 1
                    )
                    self.log(f"Allocated: {alloc_mem}GB")
                    self.log(f"Cached: {reserved_mem}GB")

                model.train()
                epoch_loss = 0
                step = 0
                for batch_data in train_loader:
                    step += 1
                    inputs, labels = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                    )

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    # self.log(f"Output dimensions : {outputs.shape}")
                    if outputs.shape[1] > 1:
                        outputs = outputs[
                            :, 1:, :, :
                        ]  # FIXME fix channel number
                        if len(outputs.shape) < 4:
                            outputs = outputs.unsqueeze(0)
                    loss = self.config.loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                    self.log(
                        f"* {step}/{len(train_ds) // train_loader.batch_size}, "
                        f"Train loss: {loss.detach().item():.4f}"
                    )
                    yield TrainingReport(
                        show_plot=False, weights=model.state_dict()
                    )

                # return

                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                self.log(f"Epoch: {epoch + 1}, Average loss: {epoch_loss:.4f}")

                self.log("Updating scheduler...")
                scheduler.step(epoch_loss)

                checkpoint_output = []
                eta = (
                    (time.time() - start_time)
                    * (self.config.max_epochs / (epoch + 1) - 1)
                    / 60
                )
                self.log("ETA: " + f"{eta:.2f}" + " minutes")

                if (
                    (epoch + 1) % self.config.validation_interval == 0
                    or epoch + 1 == self.config.max_epochs
                ):
                    model.eval()
                    self.log("Performing validation...")
                    with torch.no_grad():
                        for val_data in val_loader:
                            val_inputs, val_labels = (
                                val_data["image"].to(device),
                                val_data["label"].to(device),
                            )

                            try:
                                with torch.no_grad():
                                    val_outputs = sliding_window_inference(
                                        val_inputs,
                                        roi_size=size,
                                        sw_batch_size=self.config.batch_size,
                                        predictor=model,
                                        overlap=0.25,
                                        sw_device=self.config.device,
                                        device=self.config.device,
                                        progress=False,
                                    )
                            except Exception as e:
                                self.raise_error(e, "Error during validation")
                            logger.debug(
                                f"val_outputs shape : {val_outputs.shape}"
                            )
                            # val_outputs = model(val_inputs)

                            pred = decollate_batch(val_outputs)

                            labs = decollate_batch(val_labels)

                            # TODO : more parameters/flexibility
                            post_pred = Compose(
                                # AsDiscrete(threshold=0.6), # needed ?
                                EnsureType()
                            )  #
                            post_label = EnsureType()

                            val_outputs = [
                                post_pred(res_tensor) for res_tensor in pred
                            ]

                            val_labels = [
                                post_label(res_tensor) for res_tensor in labs
                            ]

                            # logger.debug(len(val_outputs))
                            # logger.debug(len(val_labels))

                            dice_metric(y_pred=val_outputs, y=val_labels)

                        checkpoint_output.append(
                            [
                                val_outputs[0].detach().cpu().numpy(),
                                val_inputs[0].detach().cpu().numpy(),
                                val_labels[0]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.uint16),
                            ]
                        )
                        [np.squeeze(vol) for vol in checkpoint_output]

                        metric = dice_metric.aggregate().detach().item()
                        dice_metric.reset()
                        val_metric_values.append(metric)

                        train_report = TrainingReport(
                            show_plot=True,
                            epoch=epoch,
                            loss_values=epoch_loss_values,
                            validation_metric=val_metric_values,
                            weights=model.state_dict(),
                            images=checkpoint_output,
                        )
                        self.log("Validation completed")
                        yield train_report

                        weights_filename = (
                            f"{model_name}_best_metric"
                            # + f"_epoch_{epoch + 1}" # avoid saving per epoch
                            + ".pth"
                        )

                        if metric > best_metric:
                            best_metric = metric
                            best_metric_epoch = epoch + 1
                            self.log("Saving best metric model")
                            torch.save(
                                model.state_dict(),
                                Path(self.config.results_path_folder)
                                / Path(
                                    weights_filename,
                                ),
                            )
                            self.log("Saving complete")
                        self.log(
                            f"Current epoch: {epoch + 1}, Current mean dice: {metric:.4f}"
                            f"\nBest mean dice: {best_metric:.4f} "
                            f"at epoch: {best_metric_epoch}"
                        )
            self.log("=" * 10)
            self.log(
                f"Train completed, best_metric: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
            # Save last checkpoint
            weights_filename = f"{model_name}_latest.pth"
            self.log("Saving last model")
            torch.save(
                model.state_dict(),
                Path(self.config.results_path_folder) / Path(weights_filename),
            )
            self.log("Saving complete, exiting")
            model.to("cpu")
            # clear (V)RAM
            # val_ds = None
            # train_ds = None
            # val_loader = None
            # train_loader = None
            # torch.cuda.empty_cache()

        except Exception as e:
            self.raise_error(e, "Error in training")
            self.quit()
        finally:
            self.quit()