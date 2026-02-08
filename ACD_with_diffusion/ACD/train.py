# from __future__ import division
# from __future__ import print_function

# from collections import defaultdict

# import time
# import numpy as np
# import torch

# from modules import *
# import arg_parser
# import logger
# import data_loader
# import forward_pass_and_eval
# import utils_x as utils
# import model_loader
# import os
# from torchinfo import summary


# def train(start_time):
#     best_val_loss = np.inf
#     best_epoch = 0

#     for epoch in range(args.epochs):
#         probs_train = []
#         t_epoch = time.time()
#         train_losses = defaultdict(list)

#         for batch_idx, minibatch in enumerate(train_loader):

#             data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

#             optimizer.zero_grad()

#             losses, _, _, _, probs = forward_pass_and_eval.forward_pass_and_eval(
#                 args,
#                 encoder,
#                 decoder,
#                 data,
#                 relations,
#                 rel_rec,
#                 rel_send,
#                 args.hard,
#                 edge_probs=edge_probs,
#                 log_prior=log_prior,
#                 temperatures=temperatures,
#             )

#             loss = losses["loss"]

#             loss.backward()
#             optimizer.step()

#             train_losses = utils.append_losses(train_losses, losses)
#             probs_train.append(probs)
#         string = logs.result_string("train", epoch, train_losses, t=t_epoch)
#         logs.write_to_log_file(string)
#         logs.append_train_loss(train_losses)
#         scheduler.step()
#         # save probs
#         if args.save_probs:
#             np_probs = np.concatenate(probs_train)
#             probs_save_file = probs_folder + 'probs_' + str(epoch) + '.npy'
#             np.save(probs_save_file, np_probs)

#         if args.validate:
#             val_losses = val(epoch)
#             val_loss = np.mean(val_losses["loss"])
#             if val_loss < best_val_loss:
#                 print("Best model so far, saving...")
#                 logs.create_log(
#                     args,
#                     encoder=encoder,
#                     decoder=decoder,
#                     optimizer=optimizer,
#                     accuracy=np.mean(val_losses["acc"]),
#                 )
#                 best_val_loss = val_loss
#                 best_epoch = epoch
#         elif (epoch + 1) % 100 == 0:
#             logs.create_log(
#                 args,
#                 encoder=encoder,
#                 decoder=decoder,
#                 optimizer=optimizer,
#                 accuracy=np.mean(train_losses["acc"]),
#             )

#         logs.draw_loss_curves()
#         if args.b_walltime:
#             epoch_end_time = time.time()
#             epoch_time = epoch_end_time - t_epoch
#             if epoch_end_time - start_time < 171900 - epoch_time:
#                 continue
#             else:
#                 break

#     return best_epoch, epoch


# def val(epoch):
#     t_val = time.time()
#     val_losses = defaultdict(list)

#     if args.use_encoder:
#         encoder.eval()
#     decoder.eval()

#     for batch_idx, minibatch in enumerate(valid_loader):

#         data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

#         with torch.no_grad():
#             losses, _, _, _, _ = forward_pass_and_eval.forward_pass_and_eval(
#                 args,
#                 encoder,
#                 decoder,
#                 data,
#                 relations,
#                 rel_rec,
#                 rel_send,
#                 True,
#                 edge_probs=edge_probs,
#                 log_prior=log_prior,
#                 testing=True,
#                 temperatures=temperatures,
#             )

#         val_losses = utils.append_losses(val_losses, losses)

#     string = logs.result_string("validate", epoch, val_losses, t=t_val)
#     logs.write_to_log_file(string)
#     logs.append_val_loss(val_losses)

#     if args.use_encoder:
#         encoder.train()
#     decoder.train()

#     return val_losses


# def test(encoder, decoder, epoch):
#     args.shuffle_unobserved = False
#     # args.prediction_steps = 49
#     test_losses = defaultdict(list)
#     probs_list = list()

#     if args.load_folder == "":
#         ## load model that had the best validation performance during training
#         if args.use_encoder:
#             encoder.load_state_dict(torch.load(args.encoder_file))
#         decoder.load_state_dict(torch.load(args.decoder_file))

#     if args.use_encoder:
#         encoder.eval()
#     decoder.eval()

#     for batch_idx, minibatch in enumerate(test_loader):

#         data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

#         with torch.no_grad():
#             assert (data.size(2) - args.timesteps) >= args.timesteps

#             data_encoder = data[:, :, : args.timesteps, :].contiguous()
#             data_decoder = data[:, :, args.timesteps : -1, :].contiguous()

#             losses, _, _, _, probs = forward_pass_and_eval.forward_pass_and_eval(
#                 args,
#                 encoder,
#                 decoder,
#                 data,
#                 relations,
#                 rel_rec,
#                 rel_send,
#                 True,
#                 data_encoder=data_encoder,
#                 data_decoder=data_decoder,
#                 edge_probs=edge_probs,
#                 log_prior=log_prior,
#                 testing=True,
#                 temperatures=temperatures,
#             )
#         probs_list.append(probs)
#         test_losses = utils.append_losses(test_losses, losses)

#     string = logs.result_string("test", epoch, test_losses)
#     logs.write_to_log_file(string)
#     logs.append_test_loss(test_losses)

#     logs.create_log(
#         args,
#         decoder=decoder,
#         encoder=encoder,
#         optimizer=optimizer,
#         final_test=True,
#         test_losses=test_losses,
#     )
#     np.save(args.save_folder + 'results/edges_test.npy', np.concatenate(probs_list))
#     print("edges_test saved at: " + args.save_folder + 'results/edges_test.npy')

#     print("Finished.")
#     print("Dataset: ", args.suffix)
#     print("Ground truth graph locates at: ", args.data_path)
#     print("With portion: ", args.b_portion)
#     print("With ", args.b_time_steps, " time steps")


# if __name__ == "__main__":

#     t_begin = time.time()
#     args = arg_parser.parse_args()
#     logs = logger.Logger(args)
#     folder_path = logs.path

#     probs_folder = folder_path + '/probs/'
#     os.mkdir(probs_folder)
#     if args.GPU_to_use is not None:
#         logs.write_to_log_file("Using GPU #" + str(args.GPU_to_use))

#     # original:
#     # (
#     #     train_loader,
#     #     valid_loader,
#     #     test_loader,
#     #     loc_max,
#     #     loc_min,
#     #     vel_max,
#     #     vel_min,
#     # ) = data_loader.load_data(args)

#     (
#         train_loader,
#         valid_loader,
#         test_loader,
#         loc_max,
#         loc_min,
#         vel_max,
#         vel_min,
#     ) = data_loader.load_data_customized(args)

#     # original:
#     # rel_rec, rel_send = utils.create_rel_rec_send(args, args.num_atoms)

#     rel_rec, rel_send = utils.create_rel_rec_send_bi(args, args.num_atoms)

#     encoder, decoder, optimizer, scheduler, edge_probs = model_loader.load_model(
#         args, loc_max, loc_min, vel_max, vel_min
#     )

#     logs.write_to_log_file(encoder)
#     logs.write_to_log_file(decoder)

#     if args.prior != 1:
#         assert 0 <= args.prior <= 1, "args.prior not in the right range"
#         prior = np.array(
#             [args.prior]
#             + [
#                 (1 - args.prior) / (args.edge_types - 1)
#                 for _ in range(args.edge_types - 1)
#             ]
#         )
#         logs.write_to_log_file("Using prior")
#         logs.write_to_log_file(prior)
#         log_prior = torch.FloatTensor(np.log(prior))
#         log_prior = log_prior.unsqueeze(0).unsqueeze(0)

#         if args.cuda:
#             log_prior = log_prior.cuda()
#     else:
#         log_prior = None

#     print("Summary of Encoder: ")
#     summary(encoder,
#             input_size=[
#                 (args.batch_size, args.num_atoms, args.timesteps, args.dims),
#                 (rel_rec.size()),
#                 (rel_send.size()),
#                 # (adj.size())
#             ])

#     print("-" * 15)
#     print("Summary of Decoder: ")

#     # original:
#     # summary(decoder,
#     #         input_size=[
#     #             (args.batch_size, args.num_atoms, args.timesteps, args.dims),
#     #             (args.batch_size, args.num_atoms ** 2 - args.num_atoms, args.edge_types),
#     #             (rel_rec.size()),
#     #             (rel_send.size()),
#     #             # 1
#     #         ])

#     summary(decoder,
#             input_size=[
#                 (args.batch_size, args.num_atoms, args.timesteps, args.dims),
#                 (args.batch_size, args.num_atoms ** 2, args.edge_types),
#                 (rel_rec.size()),
#                 (rel_send.size()),
#                 # 1
#             ])

#     if args.global_temp:
#         args.categorical_temperature_prior = utils.get_categorical_temperature_prior(
#             args.alpha, args.num_cats, to_cuda=args.cuda
#         )

#     ##Train model
#     try:
#         if args.test_time_adapt:
#             raise KeyboardInterrupt

#         best_epoch, epoch = train(t_begin)

#     except KeyboardInterrupt:
#         best_epoch, epoch = -1, -1

#     print("Optimization Finished!")
#     logs.write_to_log_file("Best Epoch: {:04d}".format(best_epoch))

#     if args.test:
#         test(encoder, decoder, epoch)

from __future__ import division
from __future__ import print_function
import inspect
from collections import defaultdict
import time
import os
import sys
import pprint
import numpy as np
import torch
from torchinfo import summary

from modules import *
import arg_parser
import logger
import data_loader
import forward_pass_and_eval
import utils_x as utils
import model_loader


# -----------------------------
# Helpers: logging + aggregation
# -----------------------------
LOG_ALL_PATH = None   # will be set in __main__
LOG_BEST_PATH = None  # will be set in __main__


def _append_line(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def log_all(text: str):
    """Write to log_all.txt"""
    if LOG_ALL_PATH is None:
        return
    _append_line(LOG_ALL_PATH, text)


def log_best(text: str):
    """Write to log.txt (best-only)"""
    if LOG_BEST_PATH is None:
        return
    _append_line(LOG_BEST_PATH, text)


def write_log_all_header(args):
    """Write header (cmd + args) to log_all.txt once."""
    if LOG_ALL_PATH is None:
        return
    need = (not os.path.exists(LOG_ALL_PATH)) or (os.path.getsize(LOG_ALL_PATH) == 0)
    if not need:
        return

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cmd = " ".join(sys.argv)
    args_dict = vars(args)

    log_all("=" * 100)
    log_all(f"Start time: {now}")
    log_all(f"Command: {cmd}")
    log_all("Parsed args:")
    log_all(pprint.pformat(args_dict, sort_dicts=True))
    log_all("=" * 100)
    log_all("")


def mean_losses(loss_dict):
    """defaultdict(list) -> dict(mean)"""
    out = {}
    for k, v in loss_dict.items():
        try:
            out[k] = float(np.mean(v))
        except Exception:
            pass
    return out


def format_losses(losses: dict) -> str:
    """dict -> 'k=v, k=v'"""
    if losses is None:
        return ""
    parts = []
    for k in sorted(losses.keys()):
        v = losses[k]
        try:
            parts.append(f"{k}={float(v):.6g}")
        except Exception:
            parts.append(f"{k}={v}")
    return ", ".join(parts)

def call_forward_pass_and_eval(
    args,
    encoder,
    decoder,
    data,
    relations,
    rel_rec,
    rel_send,
    hard,
    data_encoder=None,
    data_decoder=None,
    edge_probs=None,
    testing=False,
    log_prior=None,
    temperatures=None,
    diffusion_refiner=None,
):
    fn = forward_pass_and_eval.forward_pass_and_eval
    sig = inspect.signature(fn)

    kwargs = dict(
        data_encoder=data_encoder,
        data_decoder=data_decoder,
        edge_probs=edge_probs,
        testing=testing,
        log_prior=log_prior,
        temperatures=temperatures,
    )

    # 只有当 forward_pass_and_eval.py 未来加入该参数时才传，保证“现在不改 forward_pass 也兼容”
    if diffusion_refiner is not None:
        if "diffusion_refiner" in sig.parameters:
            kwargs["diffusion_refiner"] = diffusion_refiner
        elif "refiner" in sig.parameters:
            kwargs["refiner"] = diffusion_refiner

    return fn(
        args,
        encoder,
        decoder,
        data,
        relations,
        rel_rec,
        rel_send,
        hard,
        **kwargs,
    )

def log_epoch_all(epoch: int, split: str, losses_mean: dict, elapsed_s: float = None):
    """One line per epoch per split in log_all.txt"""
    if elapsed_s is not None:
        log_all(f"[{split}] epoch={epoch:04d}  t={elapsed_s:.2f}s  {format_losses(losses_mean)}")
    else:
        log_all(f"[{split}] epoch={epoch:04d}  {format_losses(losses_mean)}")


def log_epoch_best(epoch: int, metric_name: str, metric_value: float,
                   train_mean: dict = None, val_mean: dict = None, test_mean: dict = None):
    """Write only when best improves."""
    log_best("=" * 80)
    log_best(f"BEST UPDATE @ epoch={epoch:04d}  {metric_name}={metric_value:.6g}")
    if train_mean is not None:
        log_best(f"[train] {format_losses(train_mean)}")
    if val_mean is not None:
        log_best(f"[val]   {format_losses(val_mean)}")
    if test_mean is not None:
        log_best(f"[test]  {format_losses(test_mean)}")
    log_best("")


# -----------------------------
# Train / Val / Test
# -----------------------------
def train(start_time):
    # best metric: default same as你现在代码：验证集 mean(losses["loss"]) 越小越好
    best_val_loss = np.inf
    best_epoch = 0

    # 如果不开 validate，就用训练集 best（避免每100轮保存）
    best_train_loss = np.inf

    for epoch in range(args.epochs):
        probs_train = []
        t_epoch = time.time()
        train_losses = defaultdict(list)
        # NRI v2 aligned: always joint training (no diffusion_phase)
        if args.use_encoder:
            encoder.train()
        decoder.train()
        if args.use_diffusion and diffusion_refiner is not None:
            diffusion_refiner.train()


        for batch_idx, minibatch in enumerate(train_loader):
            data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

            optimizer.zero_grad()

            losses, _, _, _, probs = call_forward_pass_and_eval(
                args,
                encoder,
                decoder,
                data,
                relations,
                rel_rec,
                rel_send,
                args.hard,
                edge_probs=edge_probs,
                log_prior=log_prior,
                temperatures=temperatures,
                diffusion_refiner=diffusion_refiner,
            )


            loss = losses["loss"]
            loss.backward()
            optimizer.step()

            train_losses = utils.append_losses(train_losses, losses)
            probs_train.append(probs)

        # ---- epoch end: train logging (ALL) ----
        train_mean = mean_losses(train_losses)
        epoch_elapsed = time.time() - t_epoch
        log_epoch_all(epoch, "train", train_mean, elapsed_s=epoch_elapsed)

        # keep your existing curves bookkeeping
        logs.append_train_loss(train_losses)

        scheduler.step()

        # save probs (optional)
        # if args.save_probs and ((epoch % max(int(args.save_probs_distance), 1)) == 0):
        #     np_probs = np.concatenate(probs_train)
        #     probs_save_file = os.path.join(probs_folder, f"probs_{epoch}.npy")
        #     np.save(probs_save_file, np_probs)

        # ---- validation ----
        if args.validate:
            val_losses = val(epoch)
            val_mean = mean_losses(val_losses)
            log_epoch_all(epoch, "val", val_mean)

            val_loss = float(val_mean.get("loss", np.mean(val_losses["loss"])))

            if val_loss < best_val_loss:
                print("Best model so far, saving...")
                # 保存 best checkpoint（保持原逻辑）
                logs.create_log(
                    args,
                    encoder=encoder,
                    decoder=decoder,
                    refiner=diffusion_refiner,
                    optimizer=optimizer,
                    accuracy=float(val_mean.get("acc", np.mean(val_losses["acc"]))),
                )

                best_val_loss = val_loss
                best_epoch = epoch

                # ---- best-only log (log.txt) ----
                log_epoch_best(
                    epoch=epoch,
                    metric_name="val_loss",
                    metric_value=best_val_loss,
                    train_mean=train_mean,
                    val_mean=val_mean,
                    test_mean=None,
                )
        else:
            # 不validate时：用训练loss作为 best 标准，只在变好时保存&写 log.txt
            train_loss = float(train_mean.get("loss", np.mean(train_losses["loss"])))
            if train_loss < best_train_loss:
                print("Best train model so far, saving...")
                logs.create_log(
                    args,
                    encoder=encoder,
                    decoder=decoder,
                    refiner=diffusion_refiner,
                    optimizer=optimizer,
                    accuracy=float(train_mean.get("acc", np.mean(train_losses["acc"]))),
                )
                best_train_loss = train_loss
                best_epoch = epoch

                log_epoch_best(
                    epoch=epoch,
                    metric_name="train_loss",
                    metric_value=best_train_loss,
                    train_mean=train_mean,
                    val_mean=None,
                    test_mean=None,
                )

        # optional: draw curves every epoch (your original behavior)
        logs.draw_loss_curves()

        # walltime logic
        if args.b_walltime:
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - t_epoch
            if epoch_end_time - start_time < 171900 - epoch_time:
                continue
            else:
                break

    return best_epoch, epoch


def val(epoch):
    t_val = time.time()
    val_losses = defaultdict(list)

    if args.use_encoder:
        encoder.eval()
    decoder.eval()
    if args.use_diffusion and diffusion_refiner is not None:
        diffusion_refiner.eval()

    for batch_idx, minibatch in enumerate(valid_loader):
        data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

        with torch.no_grad():
            losses, _, _, _, _ = call_forward_pass_and_eval(
    args,
    encoder,
    decoder,
    data,
    relations,
    rel_rec,
    rel_send,
    True,
    edge_probs=edge_probs,
    log_prior=log_prior,
    testing=True,
    temperatures=temperatures,
    diffusion_refiner=diffusion_refiner,
)


        val_losses = utils.append_losses(val_losses, losses)

    # restore
    if args.use_encoder:
        encoder.train()
    decoder.train()
    if args.use_diffusion and diffusion_refiner is not None:
        diffusion_refiner.train()

    # 这里不写 log.txt，只返回给 train() 做 best 判断
    # 若你想记录 val 总耗时，也可以写到 log_all
    val_elapsed = time.time() - t_val
    val_mean = mean_losses(val_losses)
    log_all(f"[val_time] epoch={epoch:04d}  t={val_elapsed:.2f}s")
    return val_losses


def test(encoder, decoder, epoch):
    args.shuffle_unobserved = False
    test_losses = defaultdict(list)
    probs_list = []

    if args.load_folder == "":
        # load best validation model
        if args.use_encoder:
            encoder.load_state_dict(torch.load(args.encoder_file))
        decoder.load_state_dict(torch.load(args.decoder_file))

        if args.use_diffusion and diffusion_refiner is not None:
            dp_path = getattr(args, "diffusion_prior_file", None)
            if dp_path is None:
                dp_path = getattr(args, "refiner_file", None)
            if dp_path is not None and os.path.exists(dp_path):
                diffusion_refiner.load_state_dict(torch.load(dp_path, map_location=args.device))
            else:
                print(f"[WARN] use_diffusion=True but cannot find diffusion prior checkpoint at: {dp_path}")

    if args.use_encoder:
        encoder.eval()
    decoder.eval()
    if args.use_diffusion and diffusion_refiner is not None:
        diffusion_refiner.eval()

    for batch_idx, minibatch in enumerate(test_loader):
        data, relations, temperatures = data_loader.unpack_batches(args, minibatch)

        with torch.no_grad():
            assert (data.size(2) - args.timesteps) >= args.timesteps

            data_encoder = data[:, :, : args.timesteps, :].contiguous()
            data_decoder = data[:, :, args.timesteps : -1, :].contiguous()

            losses, _, _, _, probs = call_forward_pass_and_eval(
                args,
                encoder,
                decoder,
                data,
                relations,
                rel_rec,
                rel_send,
                True,
                data_encoder=data_encoder,
                data_decoder=data_decoder,
                edge_probs=edge_probs,
                log_prior=log_prior,
                testing=True,
                temperatures=temperatures,
                diffusion_refiner=diffusion_refiner,
            )


        probs_list.append(probs)
        test_losses = utils.append_losses(test_losses, losses)

    test_mean = mean_losses(test_losses)
    log_epoch_all(epoch, "test", test_mean)

    logs.append_test_loss(test_losses)

    # keep original final_test saving behavior
    logs.create_log(
        args,
        decoder=decoder,
        encoder=encoder,
        refiner=diffusion_refiner,
        optimizer=optimizer,
        final_test=True,
        test_losses=test_losses,
    )

    # edges_out = os.path.join(args.save_folder, "results", "edges_test.npy")
    # np.save(edges_out, np.concatenate(probs_list))
    # print("edges_test saved at: " + edges_out)

    print("Finished.")
    print("Dataset: ", args.suffix)
    print("Ground truth graph locates at: ", args.data_path)
    print("With portion: ", args.b_portion)
    print("With ", args.b_time_steps, " time steps")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    t_begin = time.time()
    args = arg_parser.parse_args()

    # original logger object (used for saving models/curves)
    logs = logger.Logger(args)
    folder_path = logs.path

    # our two log files in the same folder
    LOG_ALL_PATH = os.path.join(folder_path, "log_all.txt")
    LOG_BEST_PATH = os.path.join(folder_path, "log.txt")
    write_log_all_header(args)

    # probs folder
    probs_folder = os.path.join(folder_path, "probs")
    os.makedirs(probs_folder, exist_ok=True)

    if args.GPU_to_use is not None:
        log_all("Using GPU #" + str(args.GPU_to_use))

    # load data
    (
        train_loader,
        valid_loader,
        test_loader,
        loc_max,
        loc_min,
        vel_max,
        vel_min,
    ) = data_loader.load_data_customized(args)

    # relation receiver/sender matrices
    rel_rec, rel_send = utils.create_rel_rec_send_bi(args, args.num_atoms)

    # load model
    encoder, decoder, diffusion_refiner, optimizer, scheduler, edge_probs = model_loader.load_model(
        args, loc_max, loc_min, vel_max, vel_min
    )



    # write model repr to log_all
    log_all("Encoder:\n" + str(encoder))
    log_all("Decoder:\n" + str(decoder))
    log_all("DiffusionRefiner:\n" + str(diffusion_refiner))

    # prior
    if args.prior != 1:
        assert 0 <= args.prior <= 1, "args.prior not in the right range"
        prior = np.array(
            [args.prior]
            + [
                (1 - args.prior) / (args.edge_types - 1)
                for _ in range(args.edge_types - 1)
            ]
        )
        log_all("Using prior")
        log_all(str(prior))

        log_prior = torch.FloatTensor(np.log(prior)).unsqueeze(0).unsqueeze(0)
        if args.cuda:
            log_prior = log_prior.cuda()
    else:
        log_prior = None

    # summaries (optional; may be large)
    print("Summary of Encoder: ")
    summary(
        encoder,
        input_size=[
            (args.batch_size, args.num_atoms, args.timesteps, args.dims),
            (rel_rec.size()),
            (rel_send.size()),
        ],
    )

    print("-" * 15)
    print("Summary of Decoder: ")
    summary(
        decoder,
        input_size=[
            (args.batch_size, args.num_atoms, args.timesteps, args.dims),
            (args.batch_size, args.num_atoms ** 2, args.edge_types),
            (rel_rec.size()),
            (rel_send.size()),
        ],
    )

    if args.global_temp:
        args.categorical_temperature_prior = utils.get_categorical_temperature_prior(
            args.alpha, args.num_cats, to_cuda=args.cuda
        )

    # Train model
    try:
        if args.test_time_adapt:
            raise KeyboardInterrupt
        best_epoch, epoch = train(t_begin)
    except KeyboardInterrupt:
        best_epoch, epoch = -1, -1

    print("Optimization Finished!")
    log_all("Optimization Finished!")
    log_all("Best Epoch: {:04d}".format(best_epoch))

    if args.test:
        test(encoder, decoder, epoch)
