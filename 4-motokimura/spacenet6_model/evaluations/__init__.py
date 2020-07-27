import segmentation_models_pytorch as smp


def _iou_metrics(config, thresh=0.5):
    """
    """
    metrics = []

    # mean iou of all classes
    metric = smp.utils.metrics.IoU(threshold=thresh)
    metric.__name__ = 'iou/all'
    metrics.append(metric)

    # iou of each category
    classes = config.INPUT.CLASSES
    for target_class in classes:
        other_classes = classes.copy()
        other_classes.remove(target_class)
        ignore_channels = [
            classes.index(c) for c in other_classes
        ]

        metric = smp.utils.metrics.IoU(
            threshold=thresh,
            ignore_channels=ignore_channels
        )
        metric.__name__ = f'iou/{target_class}'

        metrics.append(metric)

    return metrics


def get_metrics(config):
    """
    """
    def _get_metrics(config, metric_name):
        """
        """
        # TODO: support multiple metrics
        if metric_name == 'iou':
            return _iou_metrics(config)
        else:
            raise ValueError()

    metrics = []
    for metric_name in config.EVAL.METRICS:
        metrics.extend(_get_metrics(config, metric_name))

    return metrics
