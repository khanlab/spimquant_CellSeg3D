from napari_cellseg_annotator import plugin_model_training as train


def test_create_train_dataset_dict(make_napari_viewer):
    view = make_napari_viewer()
    widget = train.Trainer(view)

    widget.images_filepath = [""]
    widget.labels_filepaths = [""]

    assert not widget.check_ready()

    widget.images_filepath = ["C:/test/something.tif"]
    widget.labels_filepaths = ["C:/test/lab_something.tif"]

    assert widget.check_ready()


def test_update_loss_plot(make_napari_viewer):
    view = make_napari_viewer()
    widget = train.Trainer(view)

    widget.epoch_loss_values = [0, 1]
    widget.metric_values = [0.2]

    widget.update_loss_plot()

    assert widget.dice_metric_plot is None
    assert widget.train_loss_plot is None

    widget.epoch_loss_values = [0, 1, 0.5, 0.7]
    widget.metric_values = [0.2, 0.3]

    widget.update_loss_plot()

    assert widget.dice_metric_plot is not None
    assert widget.train_loss_plot is not None

    widget.epoch_loss_values = [0, 1, 0.5, 0.7, 0.5, 0.7]
    widget.metric_values = [0.2, 0.3, 0.5, 0.7]

    widget.update_loss_plot()

    assert widget.dice_metric_plot is not None
    assert widget.train_loss_plot is not None
