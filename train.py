import argparse
from spimquant import CellSeg3DModelConfig


def get_config():
    pass


def train_model(config: CellSeg3DModelConfig):
    PROCESSED_SLICE_FOLDER = f'{config.scratch_folder}/cellseg3d_train_inputs'
    if os.path.exists(PROCESSED_SLICE_FOLDER):
        shutil.rmtree(PROCESSED_SLICE_FOLDER)
    os.mkdir(PROCESSED_SLICE_FOLDER)
    if os.path.exists(config.result_folder):
        shutil.rmtree(config.result_folder)
    os.mkdir(config.result_folder)

    clamp_min, clamp_max = config.input_brightness_range
    preprocess_to_3d_tiles(config.trainset, PROCESSED_SLICE_FOLDER,
                                 config.IM_TRAIN_CHANNEL,
                                 clamp_max, clamp_min)

    batch_size = 4
    learning_rate = 2e-5
    weight_decay = 0.01
    validation_frequency = 2
    intensity_sigma = 1.0
    spatial_sigma = 4.0
    ncuts_radius = 2

    train_data_folder = Path(PROCESSED_SLICE_FOLDER)

    def create_dataset(folder):
        images_filepaths = utils.get_all_matching_files(folder, pattern={'.npy', })
        data_dict = [{"image": str(image_name)} for image_name in images_filepaths]
        return data_dict

    WANDB_INSTALLED = False  # we don't use wandb

    global train_config, wandb_config
    train_config = WNetTrainingWorkerConfig(
        device=config.device,
        max_epochs=config.number_of_epochs,
        learning_rate=learning_rate,
        validation_interval=validation_frequency,
        batch_size=batch_size,
        num_workers=2,
        weights_info=WeightsInfo(),
        results_path_folder=config.result_folder,
        train_data_dict=create_dataset(train_data_folder),
        eval_volume_dict=None,
        # advanced
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_classes=config.num_classes,
        dropout=config.dropout,
        weight_decay=weight_decay,
        intensity_sigma=intensity_sigma,
        spatial_sigma=spatial_sigma,
        radius=ncuts_radius,
        reconstruction_loss=config.rec_loss,
        n_cuts_weight=config.n_cuts_weight,
        rec_loss_weight=config.rec_loss_weight,
        input_brightness_range=config.input_brightness_range,
    )
    wandb_config = WandBConfig(
        mode="disabled" if not WANDB_INSTALLED else "online",
        save_model_artifact=False,
    )

    worker = c.get_colab_worker(worker_config=train_config, wandb_config=wandb_config)
    for epoch_loss in worker.train():
        continue
    config.model_weight_path = f'{config.result_folder}/wnet.pth'
    return config


def train_channel(channel: int, trainset: DatasetReference, scratch_folder: str, result_folder: str, device: str,
                  nepoch=20, num_classes=5):
    if channel == 0:
        MIN_BRIGHTNESS = 0.
        MAX_BRIGHTNESS = 1000.
        rec_loss_weight = .005
    else:
        MIN_BRIGHTNESS = 300.
        MAX_BRIGHTNESS = 1000.
        rec_loss_weight = .005

    config = CellSeg3DModelConfig(
        trainset=trainset,
        scratch_folder=scratch_folder,
        result_folder=result_folder,
        IM_TRAIN_CHANNEL=channel,
        input_brightness_range=(MIN_BRIGHTNESS, MAX_BRIGHTNESS),
        number_of_epochs=nepoch,
        n_cuts_weight=.5,
        rec_loss_weight=rec_loss_weight,
        rec_loss='MSE',
        device=device,
        num_classes=num_classes,
    )
    train_model(config)
    return config
