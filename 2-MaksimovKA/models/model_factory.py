import segmentation_models_pytorch as smp


def make_model(model_name='unet_resnet34',
               weights='imagenet',
               n_classes=2,
               input_channels=4):

    if model_name.split('_')[0] == 'unet':
        
        model = smp.Unet('_'.join(model_name.split('_')[1:]), 
                         classes=n_classes,
                         activation=None,
                         encoder_weights=weights,
                         in_channels=input_channels)

    elif model_name.split('_')[0] == 'fpn':
        model = smp.FPN('_'.join(model_name.split('_')[1:]),
                        classes=n_classes,
                        activation=None,
                        encoder_weights=weights,
                        in_channels=input_channels)

    elif model_name.split('_')[0] == 'linknet':
        model = smp.Linknet('_'.join(model_name.split('_')[1:]),
                            classes=n_classes,
                            activation=None,
                            encoder_weights=weights,
                            in_channels=input_channels)
    else:
        raise ValueError('Model not implemented')
    
    return model
