# training with captions
# XXX dropped option: hypernetwork training

import argparse
import math
import os
from multiprocessing import Value
import toml

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device
init_ipex()

from argparse import Namespace
from accelerate.utils import set_seed
from diffusers import DDPMScheduler

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.train_util as train_util
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    apply_debiased_estimation,
)


def train(
    v2=False,
    v_parameterization=False,
    pretrained_model_name_or_path=None,
    tokenizer_cache_dir=None,
    output_dir=None,
    output_name=None,
    huggingface_repo_id=None,
    huggingface_repo_type=None,
    huggingface_path_in_repo=None,
    huggingface_token=None,
    huggingface_repo_visibility=None,
    save_state_to_huggingface=False,
    resume_from_huggingface=False,
    async_upload=False,
    save_precision=None,
    save_every_n_epochs=None,
    save_every_n_steps=None,
    save_n_epoch_ratio=None,
    save_last_n_epochs=None,
    save_last_n_epochs_state=None,
    save_last_n_steps=None,
    save_last_n_steps_state=None,
    save_state=None,
    resume=None,
    train_batch_size=1,
    max_token_length=None,
    mem_eff_attn=False,
    torch_compile=False,
    dynamo_backend="inductor",
    xformers=False,
    sdpa=False,
    vae=None,
    max_train_epochs=None,
    max_data_loader_n_workers=8,
    seed=None,
    persistent_data_loader_workers=False,
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
    mixed_precision="no",
    full_fp16=False,
    full_bf16=False,
    fp8_base=False,
    ddp_timeout=None,
    ddp_gradient_as_bucket_view=False,
    ddp_static_graph=False,
    clip_skip=None,
    logging_dir=None,
    log_with=None,
    log_prefix=None,
    log_tracker_name=None,
    wandb_run_name=None,
    log_tracker_config=None,
    wandb_api_key=None,
    noise_offset=None,
    multires_noise_iterations=None,
    ip_noise_gamma=None,
    multires_noise_discount=0.3,
    adaptive_noise_scale=None,
    zero_terminal_snr=False,
    min_timestep=None,
    max_timestep=None,
    lowram=False,
    highvram=False,
    sample_every_n_steps=None,
    sample_at_first=False,
    sample_every_n_epochs=None,
    sample_prompts=None,
    sample_sampler="ddim",
    config_file=None,
    output_config=False,
    metadata_title=None,
    metadata_author=None,
    metadata_description=None,
    metadata_license=None,
    metadata_tags=None,
    prior_loss_weight=1.0,
    train_data_dir=None,
    shuffle_caption=False,
    caption_separator=",",
    caption_extension=".caption",
    caption_extention=None,
    keep_tokens=0,
    keep_tokens_separator="",
    caption_prefix=None,
    caption_suffix=None,
    color_aug=False,
    flip_aug=False,
    face_crop_aug_range=None,
    random_crop=False,
    debug_dataset=False,
    resolution=None,
    cache_latents=False,
    vae_batch_size=1,
    cache_latents_to_disk=False,
    enable_bucket=False,
    min_bucket_reso=256,
    max_bucket_reso=1024,
    bucket_reso_steps=64,
    bucket_no_upscale=False,
    token_warmup_min=1,
    token_warmup_step=0,
    dataset_class=None,
    caption_dropout_rate=0.0,
    caption_dropout_every_n_epochs=0,
    caption_tag_dropout_rate=0.0,
    reg_data_dir=None,
    in_json=None,
    dataset_repeats=1,

    save_model_as=None,
    use_safetensors=False,

    optimizer_type="",
    use_8bit_adam=False,
    use_lion_optimizer=False,
    learning_rate=2.0e-6,
    max_grad_norm=1.0,
    optimizer_args=None,
    lr_scheduler_type="",
    lr_scheduler_args=None,
    lr_scheduler="constant",
    lr_warmup_steps=0,
    lr_scheduler_num_cycles=1,
    lr_scheduler_power=1,

    dataset_config=None,

    min_snr_gamma=None,
    scale_v_pred_loss_like_noise_pred=False,
    v_pred_like_loss=None,
    debiased_estimation_loss=False,
    weighted_captions=False,

    diffusers_xformers=False,
    train_text_encoder=False,
    learning_rate_te=None,
    no_half_vae=False,

    console_log_level=None,
    console_log_file=None,
    console_log_simple=None,

    max_train_steps=None,
):
    args = Namespace(**locals())
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    setup_logging(args, reset=True)
    locals().update(vars(args))

    if seed is not None:
        set_seed(seed)  # 乱数系列を初期化する

    tokenizer = train_util.load_tokenizer(args)

    # データセットを準備する
    if dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(False, True, False, True))
        if dataset_config is not None:
            logger.info(f"Load dataset config from {dataset_config}")
            user_config = config_util.load_user_config(dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            user_config = {
                "datasets": [
                    {
                        "subsets": [
                            {
                                "image_dir": train_data_dir,
                                "metadata_file": in_json,
                            }
                        ]
                    }
                ]
            }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    if debug_dataset:
        train_util.debug_dataset(train_dataset_group)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # モデルを読み込む
    text_encoder, vae, unet, load_stable_diffusion_format = train_util.load_target_model(args, weight_dtype, accelerator)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = pretrained_model_name_or_path

    if save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = use_safetensors
    else:
        save_stable_diffusion_format = save_model_as.lower() == "ckpt" or save_model_as.lower() == "safetensors"
        use_safetensors = use_safetensors or ("safetensors" in save_model_as.lower())

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        #   model.set_use_memory_efficient_attention_xformers(valid)            # 次のリリースでなくなりそう
        # pipeが自動で再帰的にset_use_memory_efficient_attention_xformersを探すんだって(;´Д｀)
        # U-Netだけ使う時にはどうすればいいのか……仕方ないからコピって使うか
        # 0.10.2でなんか巻き戻って個別に指定するようになった(;^ω^)

        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if diffusers_xformers:
        accelerator.print("Use xformers by Diffusers")
        set_diffusers_xformers_flag(unet, True)
    else:
        # Windows版のxformersはfloatで学習できないのでxformersを使わない設定も可能にしておく必要がある
        accelerator.print("Disable Diffusers' xformers")
        set_diffusers_xformers_flag(unet, False)
        train_util.replace_unet_modules(unet, mem_eff_attn, xformers, sdpa)

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, vae_batch_size, cache_latents_to_disk, accelerator.is_main_process)
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # 学習を準備する：モデルを適切な状態にする
    training_models = []
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    training_models.append(unet)

    if train_text_encoder:
        accelerator.print("enable text encoder training")
        if gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
        training_models.append(text_encoder)
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)  # text encoderは学習しない
        if gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            text_encoder.train()  # required for gradient_checkpointing
        else:
            text_encoder.eval()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    for m in training_models:
        m.requires_grad_(True)

    trainable_params = []
    if learning_rate_te is None or not train_text_encoder:
        for m in training_models:
            trainable_params.extend(m.parameters())
    else:
        trainable_params = [
            {"params": list(unet.parameters()), "lr": learning_rate},
            {"params": list(text_encoder.parameters()), "lr": learning_rate_te},
        ]

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=trainable_params)

    # dataloaderを準備する
    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = min(max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if max_train_epochs is not None:
        max_train_steps = max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {max_train_epochs} epochs is / 指定エポックまでのステップ数: {max_train_steps}"
        )

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(max_train_steps)

    # lr schedulerを用意する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
    if full_fp16:
        assert (
            mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder.to(weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい
    if train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    if (save_n_epoch_ratio is not None) and (save_n_epoch_ratio > 0):
        save_every_n_epochs = math.floor(num_train_epochs / save_n_epoch_ratio) or 1

    # 学習する
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(f"  batch size per device / バッチサイズ: {train_batch_size}")
    accelerator.print(
        f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if wandb_run_name:
            init_kwargs["wandb"] = {"name": wandb_run_name}
        if log_tracker_config is not None:
            init_kwargs = toml.load(log_tracker_config)
        accelerator.init_trackers("finetuning" if log_tracker_name is None else log_tracker_name, init_kwargs=init_kwargs)

    # For --sample_at_first
    train_util.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

    loss_recorder = train_util.LossRecorder()
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(training_models[0]):  # 複数モデルに対応していない模様だがとりあえずこうしておく
                with torch.no_grad():
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device)  # .to(dtype=weight_dtype)
                    else:
                        # latentに変換
                        latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215
                b_size = latents.shape[0]

                with torch.set_grad_enabled(train_text_encoder):
                    # Get the text embedding for conditioning
                    if weighted_captions:
                        encoder_hidden_states = get_weighted_text_embeddings(
                            tokenizer,
                            text_encoder,
                            batch["captions"],
                            accelerator.device,
                            max_token_length // 75 if max_token_length else 1,
                            clip_skip=clip_skip,
                        )
                    else:
                        input_ids = batch["input_ids"].to(accelerator.device)
                        encoder_hidden_states = train_util.get_hidden_states(
                            args, input_ids, tokenizer, text_encoder, None if not full_fp16 else weight_dtype
                        )

                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

                # Predict the noise residual
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if v_parameterization:
                    # v-parameterization training
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                if min_snr_gamma or scale_v_pred_loss_like_noise_pred or debiased_estimation_loss:
                    # do not mean over batch dimension for snr weight or scale v-pred loss
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean([1, 2, 3])

                    if min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, min_snr_gamma, v_parameterization)
                    if scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler)

                    loss = loss.mean()  # mean over batch dimension
                else:
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients and max_grad_norm != 0.0:
                    params_to_clip = []
                    for m in training_models:
                        params_to_clip.extend(m.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                train_util.sample_images(
                    accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet
                )

                # 指定ステップごとにモデルを保存
                if save_every_n_steps is not None and global_step % save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder),
                            accelerator.unwrap_model(unet),
                            vae,
                        )

            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            if logging_dir is not None:
                logs = {"loss": current_loss}
                train_util.append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_unet=True)
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if logging_dir is not None:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder),
                    accelerator.unwrap_model(unet),
                    vae,
                )

        train_util.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

    is_main_process = accelerator.is_main_process
    if is_main_process:
        unet = accelerator.unwrap_model(unet)
        text_encoder = accelerator.unwrap_model(text_encoder)

    accelerator.end_training()

    if save_state and is_main_process:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        train_util.save_sd_model_on_train_end(
            args, src_path, save_stable_diffusion_format, use_safetensors, save_dtype, epoch, global_step, text_encoder, unet, vae
        )
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, False, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--learning_rate_te",
        type=float,
        default=None,
        help="learning rate for text encoder, default is same as unet / Text Encoderの学習率、デフォルトはunetと同じ",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    train(**vars(args))
