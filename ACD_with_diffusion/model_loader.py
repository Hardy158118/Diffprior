# import os
# import torch
# import torch.optim as optim
# from torch.optim import lr_scheduler

# from modules import *
# from MLPEncoder import MLPEncoder
# from CNNEncoder import CNNEncoder
# from MLPEncoderUnobserved import MLPEncoderUnobserved
# from EncoderGlobalTemp import CNNEncoderGlobalTemp

# from MLPDecoder import MLPDecoder
# from RNNDecoder import RNNDecoder
# from SimulationDecoder import SimulationDecoder
# from DecoderGlobalTemp import MLPDecoderGlobalTemp, SimulationDecoderGlobalTemp

# import utils_x as utils
from diffusion_prior2 import DiffusionPrior


# def load_distribution(args):
#     edge_probs = torch.randn(
#         torch.Size([args.num_atoms ** 2 - args.num_atoms, args.edge_types]),
#         device=args.device.type,
#         requires_grad=True,
#     )
#     return edge_probs


# def load_encoder(args):
#     if args.global_temp:
#         encoder = CNNEncoderGlobalTemp(
#             args,
#             args.dims,
#             args.encoder_hidden,
#             args.edge_types,
#             args.encoder_dropout,
#             args.factor,
#         )
#     elif args.unobserved > 0 and args.model_unobserved == 0:
#         encoder = MLPEncoderUnobserved(
#             args,
#             args.timesteps * args.dims,
#             args.encoder_hidden,
#             args.edge_types,
#             do_prob=args.encoder_dropout,
#             factor=args.factor,
#         )
#     else:
#         if args.encoder == "mlp":
#             encoder = MLPEncoder(
#                 args,
#                 args.timesteps * args.dims,
#                 args.encoder_hidden,
#                 args.edge_types,
#                 do_prob=args.encoder_dropout,
#                 factor=args.factor,
#             )
#         elif args.encoder == "cnn":
#             encoder = CNNEncoder(
#                 args,
#                 args.dims,
#                 args.encoder_hidden,
#                 args.edge_types,
#                 args.encoder_dropout,
#                 args.factor,
#             )

#     encoder, num_GPU = utils.distribute_over_GPUs(args, encoder, num_GPU=args.num_GPU)
#     if args.load_folder:
#         print("Loading model file")
#         args.encoder_file = os.path.join(args.load_folder, "encoder.pt")
#         encoder.load_state_dict(torch.load(args.encoder_file, map_location=args.device))

#     return encoder


# def load_decoder(args, loc_max, loc_min, vel_max, vel_min):
#     if args.global_temp:
#         if args.decoder == "mlp":
#             decoder = MLPDecoderGlobalTemp(
#                 n_in_node=args.dims,
#                 edge_types=args.edge_types,
#                 msg_hid=args.decoder_hidden,
#                 msg_out=args.decoder_hidden,
#                 n_hid=args.decoder_hidden,
#                 do_prob=args.decoder_dropout,
#                 skip_first=args.skip_first,
#                 latent_dim=args.latent_dim,
#             )
#         elif args.decoder == "sim":
#             decoder = SimulationDecoderGlobalTemp(
#                 loc_max, loc_min, vel_max, vel_min, args.suffix
#             )
#     else:
#         if args.decoder == "mlp":
#             decoder = MLPDecoder(
#                 args,
#                 n_in_node=args.dims,
#                 edge_types=args.edge_types,
#                 msg_hid=args.decoder_hidden,
#                 msg_out=args.decoder_hidden,
#                 n_hid=args.decoder_hidden,
#                 do_prob=args.decoder_dropout,
#                 skip_first=args.skip_first,
#             )
#         elif args.decoder == "rnn":
#             decoder = RNNDecoder(
#                 n_in_node=args.dims,
#                 edge_types=args.edge_types,
#                 n_hid=args.decoder_hidden,
#                 do_prob=args.decoder_dropout,
#                 skip_first=args.skip_first,
#             )
#         elif args.decoder == "sim":
#             decoder = SimulationDecoder(loc_max, loc_min, vel_max, vel_min, args.suffix)

#     decoder, num_GPU = utils.distribute_over_GPUs(args, decoder, num_GPU=args.num_GPU)
#     # print("Let's use", num_GPU, "GPUs!")

#     if args.load_folder:
#         print("Loading model file")
#         args.decoder_file = os.path.join(args.load_folder, "decoder.pt")
#         decoder.load_state_dict(torch.load(args.decoder_file, map_location=args.device))
#         args.save_folder = False

#     return decoder


# def load_model(args, loc_max, loc_min, vel_max, vel_min):

#     decoder = load_decoder(args, loc_max, loc_min, vel_max, vel_min)

#     if args.use_encoder:
#         encoder = load_encoder(args)
#         edge_probs = None
#         optimizer = optim.Adam(
#             list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr,
#         )
#     else:
#         encoder = None
#         edge_probs = load_distribution(args)
#         optimizer = optim.Adam(
#             [{"params": edge_probs, "lr": args.lr_z}]
#             + [{"params": decoder.parameters(), "lr": args.lr}]
#         )

#     scheduler = lr_scheduler.StepLR(
#         optimizer, step_size=args.lr_decay, gamma=args.gamma
#     )

#     return (
#         encoder,
#         decoder,
#         optimizer,
#         scheduler,
#         edge_probs,
#     )


import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from modules import *
from MLPEncoder import MLPEncoder
from CNNEncoder import CNNEncoder
from MLPEncoderUnobserved import MLPEncoderUnobserved
from EncoderGlobalTemp import CNNEncoderGlobalTemp


from MLPDecoder import MLPDecoder
from RNNDecoder import RNNDecoder
from SimulationDecoder import SimulationDecoder
from DecoderGlobalTemp import MLPDecoderGlobalTemp, SimulationDecoderGlobalTemp

import utils_x as utils
from diffusion_prior2 import DiffusionPrior


def load_distribution(args):
    edge_probs = torch.randn(
        torch.Size([args.num_atoms ** 2 - args.num_atoms, args.edge_types]),
        device=args.device.type,
        requires_grad=True,
    )
    return edge_probs


def load_encoder(args):
    if args.global_temp:
        encoder = CNNEncoderGlobalTemp(
            args,
            args.dims,
            args.encoder_hidden,
            args.edge_types,
            args.encoder_dropout,
            args.factor,
        )
    elif args.unobserved > 0 and args.model_unobserved == 0:
        encoder = MLPEncoderUnobserved(
            args,
            args.timesteps * args.dims,
            args.encoder_hidden,
            args.edge_types,
            do_prob=args.encoder_dropout,
            factor=args.factor,
        )
    else:
        if args.encoder == "mlp":
            encoder = MLPEncoder(
                args,
                args.timesteps * args.dims,
                args.encoder_hidden,
                args.edge_types,
                do_prob=args.encoder_dropout,
                factor=args.factor,
            )
        elif args.encoder == "cnn":
            encoder = CNNEncoder(
                args,
                args.dims,
                args.encoder_hidden,
                args.edge_types,
                args.encoder_dropout,
                args.factor,
            )

    encoder, num_GPU = utils.distribute_over_GPUs(args, encoder, num_GPU=args.num_GPU)
    if args.load_folder:
        print("Loading model file")
        args.encoder_file = os.path.join(args.load_folder, "encoder.pt")
        encoder.load_state_dict(torch.load(args.encoder_file, map_location=args.device))

    return encoder


def load_decoder(args, loc_max, loc_min, vel_max, vel_min):
    if args.global_temp:
        if args.decoder == "mlp":
            decoder = MLPDecoderGlobalTemp(
                n_in_node=args.dims,
                edge_types=args.edge_types,
                msg_hid=args.decoder_hidden,
                msg_out=args.decoder_hidden,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout,
                skip_first=args.skip_first,
                latent_dim=args.latent_dim,
            )
        elif args.decoder == "sim":
            decoder = SimulationDecoderGlobalTemp(
                loc_max, loc_min, vel_max, vel_min, args.suffix
            )
    else:
        if args.decoder == "mlp":
            decoder = MLPDecoder(
                args,
                n_in_node=args.dims,
                edge_types=args.edge_types,
                msg_hid=args.decoder_hidden,
                msg_out=args.decoder_hidden,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout,
                skip_first=args.skip_first,
            )
        elif args.decoder == "rnn":
            decoder = RNNDecoder(
                n_in_node=args.dims,
                edge_types=args.edge_types,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout,
                skip_first=args.skip_first,
            )
        elif args.decoder == "sim":
            decoder = SimulationDecoder(loc_max, loc_min, vel_max, vel_min, args.suffix)

    decoder, num_GPU = utils.distribute_over_GPUs(args, decoder, num_GPU=args.num_GPU)
    # print("Let's use", num_GPU, "GPUs!")

    if args.load_folder:
        print("Loading model file")
        args.decoder_file = os.path.join(args.load_folder, "decoder.pt")
        decoder.load_state_dict(torch.load(args.decoder_file, map_location=args.device))
        args.save_folder = False

    return decoder


def load_model(args, loc_max, loc_min, vel_max, vel_min):
    """Create encoder/decoder and (optionally) DiffusionPrior (NRI v2).
    Returns (encoder, decoder, diffusion_refiner, optimizer, scheduler, edge_probs).
    """
    decoder = load_decoder(args, loc_max, loc_min, vel_max, vel_min)

    # -------------------- Diffusion prior (NRI v2) --------------------
    diffusion_refiner = None
    if getattr(args, "use_diffusion", False):
        diffusion_refiner = DiffusionPrior(
            latent_dim=args.edge_types,
            timesteps=args.diff_T,
            schedule=args.diff_schedule,
            time_emb_dim=args.diff_time_emb_dim,
            hidden_dim=args.diff_hidden,
            dropout=args.diff_dropout,
            lambda_ent=getattr(args, "lambda_ent", 0.0),
            scale_by_timesteps=getattr(args, "diff_scale_by_T", False),
            variance_type=getattr(args, "diff_variance_type", "beta"),
            train_t_max=getattr(args, "diff_train_t_max", None),
            train_num_t=getattr(args, "diff_train_k", 1),
        ).to(args.device)

        # load diffusion prior weights if requested
        if args.load_folder:
            dp_file = os.path.join(args.load_folder, "diffusion_prior.pt")
            if os.path.exists(dp_file):
                diffusion_refiner.load_state_dict(torch.load(dp_file, map_location=args.device))
            else:
                # backward-compat: older name
                legacy = os.path.join(args.load_folder, "refiner.pt")
                if os.path.exists(legacy):
                    diffusion_refiner.load_state_dict(torch.load(legacy, map_location=args.device))
                else:
                    print(f"[WARN] use_diffusion=True but no diffusion_prior.pt in load_folder: {args.load_folder}")

    # -------------------- Encoder / edge distribution --------------------
    if args.use_encoder:
        encoder = load_encoder(args)
        edge_probs = None

        # joint training: encoder + decoder (+ diffusion prior if enabled)
        optim_params = list(encoder.parameters()) + list(decoder.parameters())
        if diffusion_refiner is not None:
            optim_params += list(diffusion_refiner.parameters())
        optimizer = optim.Adam(optim_params, lr=args.lr)
    else:
        encoder = None
        edge_probs = load_distribution(args)

        param_groups = (
            [{"params": edge_probs, "lr": args.lr_z}]
            + [{"params": decoder.parameters(), "lr": args.lr}]
        )
        if diffusion_refiner is not None:
            param_groups += [{"params": diffusion_refiner.parameters(), "lr": args.lr}]
        optimizer = optim.Adam(param_groups)

    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay, gamma=args.gamma
    )

    return (
        encoder,
        decoder,
        diffusion_refiner,
        optimizer,
        scheduler,
        edge_probs,
    )
