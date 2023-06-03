from .unet import UNetModel, CrossConvolutionDecoder


def get_channel_mult(image_size):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")
    return channel_mult


def create_decoder(
        image_size,
        num_channels,
        num_res_blocks,
        attention_resolutions,
):
    channel_mult = get_channel_mult(image_size)

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return CrossConvolutionDecoder(
        in_channels=None,
        model_channels=num_channels,
        out_channels=1,  # Binary Classification
        num_res_blocks=num_res_blocks,
        attention_resolutions=None,
        dropout=None,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=False,
        num_heads=None,
        num_heads_upsample=None,
        use_scale_shift_norm=False,
        original_H=image_size,
        original_W=image_size,
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_unet_and_segmentor(
        image_size,
        learn_sigma,
        num_channels,
        num_res_blocks,
        num_heads,
        num_heads_upsample,
        attention_resolutions,
        dropout,
        use_checkpoint,
        use_scale_shift_norm,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    decoder = create_decoder(
        image_size,
        num_channels,
        num_res_blocks,
        attention_resolutions=attention_resolutions,
    )
    return model, decoder


def create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
):
    channel_mult = get_channel_mult(image_size)

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )
