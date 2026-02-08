


from __future__ import division
from __future__ import print_function

from collections import defaultdict
import time
import torch

import numpy as np

from modules import *
import utils_x as utils
import utils_unobserved


def test_time_adapt(
    args,
    logits,
    decoder,
    data_encoder,
    rel_rec,
    rel_send,
    predicted_atoms,
    log_prior,
    diffusion_refiner=None,
):
    """Test-time adaptation on logits.

    NRI v2 aligned:
      - decoder uses refined logits (one-step refinement) if use_diffusion
      - prior term uses diffusion loss (lambda_diff * loss_diff) instead of categorical KL
    """
    # prepare refiner core (DiffusionPrior)
    refiner_core = None
    if getattr(args, "use_diffusion", False):
        if diffusion_refiner is None:
            raise ValueError("args.use_diffusion=True but diffusion_refiner is None in test_time_adapt().")
        refiner_core = diffusion_refiner.module if hasattr(diffusion_refiner, "module") else diffusion_refiner

    with torch.enable_grad():
        tta_data_decoder = data_encoder.detach()

        if args.use_encoder:
            # initialize q(z) with q(z|x)
            tta_logits = logits.detach()
            tta_logits.requires_grad = True
        else:
            # initialize q(z) randomly
            tta_logits = torch.randn_like(logits, device=args.device.type, requires_grad=True)

        tta_optimizer = torch.optim.Adam([{"params": tta_logits, "lr": args.lr_logits}])
        tta_target = data_encoder[:, :, 1:, :].detach()

        ploss = 0.0
        first_loss = None

        for i in range(args.num_tta_steps):
            tta_optimizer.zero_grad()

            # decoder input: one-step refinement (Scheme B)
            if refiner_core is not None:
                tta_logits_for_dec = refiner_core.refine_one_step(
                    tta_logits,
                    t_ref=args.diff_t_ref,
                    gamma=args.diff_refine_gamma,
                    noise_mode=args.diff_refine_noise,
                    noise_seed=args.diff_refine_noise_seed,
                    clip_denoised=args.diff_refine_clip,
                    use_eval=True,
                )
            else:
                tta_logits_for_dec = tta_logits

            tta_edges = utils.gumbel_softmax(tta_logits_for_dec, tau=args.temp, hard=False)
            tta_output = decoder(tta_data_decoder, tta_edges, rel_rec, rel_send, args.prediction_steps)

            loss = utils.nll_gaussian(tta_output, tta_target, args.var)

            # prior term: diffusion prior replaces KL
            if refiner_core is not None:
                loss_diff, _stats = refiner_core.loss(
                    tta_logits,
                    t_max=args.diff_train_t_max,
                    num_t_samples=args.diff_train_k,
                )
                loss = loss + args.lambda_diff * loss_diff
            else:
                prob = utils.my_softmax(tta_logits, -1)
                if args.prior != 1:
                    loss = loss + utils.kl_categorical(prob, log_prior, predicted_atoms)
                else:
                    loss = loss + utils.kl_categorical_uniform(prob, predicted_atoms, args.edge_types)

            loss.backward()
            tta_optimizer.step()

            ploss += float(loss.detach().cpu())
            if first_loss is None:
                first_loss = float(loss.detach().cpu())

            if (i + 1) % 10 == 0:
                print(i, ": ", ploss / 10.0)
                ploss = 0.0

    if first_loss is not None:
        print("Fine-tuning improvement: ", first_loss - float(loss.detach().cpu()))

    return tta_logits

def forward_pass_and_eval(
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
    start = time.time()
    losses = defaultdict(lambda: torch.zeros((), device=args.device.type))

    #################### INPUT DATA ####################
    diff_data_enc_dec = False
    if data_encoder is not None and data_decoder is not None:
        diff_data_enc_dec = True

    if data_encoder is None:
        data_encoder = data
    if data_decoder is None:
        data_decoder = data

    #################### DATA WITH UNOBSERVED TIME-SERIES ####################
    predicted_atoms = args.num_atoms
    if args.unobserved > 0:
        if args.shuffle_unobserved:
            mask_idx = np.random.randint(0, args.num_atoms)
        else:
            mask_idx = args.num_atoms - 1

        ### baselines ###
        if args.model_unobserved == 1:
            (
                data_encoder,
                data_decoder,
                predicted_atoms,
                relations,
            ) = utils_unobserved.baseline_remove_unobserved(
                args, data_encoder, data_decoder, mask_idx, relations, predicted_atoms
            )
            unobserved = 0
        if args.model_unobserved == 2:
            (
                data_encoder,
                unobserved,
                losses["mse_unobserved"],
            ) = utils_unobserved.baseline_mean_imputation(args, data_encoder, mask_idx)
            data_decoder = utils_unobserved.add_unobserved_to_data(
                args, data_decoder, unobserved, mask_idx, diff_data_enc_dec
            )
    else:
        mask_idx = 0
        unobserved = 0

    #################### TEMPERATURE INFERENCE ####################
    if args.global_temp:
        ctp = args.categorical_temperature_prior
        cmax = ctp[-1]
        uniform_prior_mean = cmax
        uniform_prior_width = cmax 

    #################### ENCODER ####################
    if args.use_encoder:
        if args.unobserved > 0 and args.model_unobserved == 0:
            ## model unobserved time-series
            (
                logits,
                unobserved,
                losses["mse_unobserved"],
            ) = encoder(data_encoder, rel_rec, rel_send, mask_idx=mask_idx)
            data_decoder = utils_unobserved.add_unobserved_to_data(
                args, data_decoder, unobserved, mask_idx, diff_data_enc_dec
            )
        elif args.global_temp:
            (logits, temperature_samples, 
                    inferred_mean, inferred_width) = encoder(
                            data_encoder, rel_rec, rel_send)
            temperature_samples *= 2 * cmax
            inferred_mean *= 2 * cmax 
            inferred_width *= 2 * cmax
        else:
            ## model only the edges
            logits = encoder(data_encoder, rel_rec, rel_send)
    else:
        logits = edge_probs.unsqueeze(0).repeat(data_encoder.shape[0], 1, 1)

    if args.test_time_adapt and args.num_tta_steps > 0 and testing:
        assert args.unobserved == 0, "No implementation for test-time adaptation when there are unobserved time-series."
        logits = test_time_adapt(
            args,
            logits,
            decoder,
            data_encoder,
            rel_rec,
            rel_send,
            predicted_atoms,
            log_prior,
        )
    # print("Logits: ", logits.size())
    #################### DIFFUSION PRIOR (NRI v2) ####################
    # logits_raw: encoder output (deterministic continuous latent)
    logits_raw = logits

    # ensure keys exist for logging
    losses["loss_diffusion"] = torch.zeros((), device=args.device.type)

    # decoder input: one-step refinement (Scheme B, fixed t_ref)
    if getattr(args, "use_diffusion", False):
        if diffusion_refiner is None:
            raise ValueError(
                "args.use_diffusion=True but diffusion_refiner is None. "
                "Check model_loader.py and train.py wiring."
            )
        refiner_core = diffusion_refiner.module if hasattr(diffusion_refiner, "module") else diffusion_refiner

        logits_refined = refiner_core.refine_one_step(
            logits_raw,
            t_ref=args.diff_t_ref,
            gamma=args.diff_refine_gamma,
            noise_mode=args.diff_refine_noise,
            noise_seed=args.diff_refine_noise_seed,
            clip_denoised=args.diff_refine_clip,
            use_eval=True,  # IMPORTANT: stable decoder input (disable dropout) even during training
        )
    else:
        refiner_core = None
        logits_refined = logits_raw

    # sample edges / compute prob from refined logits (NRI v2)
    edges = utils.gumbel_softmax(logits_refined, tau=args.temp, hard=hard)
    # print("edges in passing: ", edges.size())
    prob = utils.my_softmax(logits_refined, -1)
    target = data_decoder[:, :, 1:, :]

    #################### DECODER ####################
    if args.decoder == "rnn":
        output = decoder(
            data_decoder,
            edges,
            rel_rec,
            rel_send,
            pred_steps=args.prediction_steps,
            burn_in=True,
            burn_in_steps=args.timesteps - args.prediction_steps,
        )
    else:
        if args.global_temp:
            output = decoder(
                data_decoder, 
                edges, 
                temperature_samples, 
                rel_rec, 
                rel_send, 
                args.prediction_steps
            )
        else:
            output = decoder(
                data_decoder,
                edges,
                rel_rec,
                rel_send,
                args.prediction_steps,
            )

    #################### LOSSES ####################
    if args.unobserved > 0:
        if args.model_unobserved != 1:
            losses["mse_observed"] = utils_unobserved.calc_mse_observed(
                args, output, target, mask_idx
            )

            if not args.shuffle_unobserved:
                losses["observed_acc"] = utils.edge_accuracy_observed(
                    logits, relations, num_atoms=args.num_atoms
                )
                losses["observed_auroc"] = utils.calc_auroc_observed(
                    prob, relations, num_atoms=args.num_atoms
                )
                losses["observed_auprc"] = utils.calc_auprc_observed(
                    prob, relations, num_atoms=args.num_atoms
                )
                losses["observed_jac"] = utils.calc_jac_observed(
                    prob, relations, num_atoms=args.num_atoms
                )

    if args.global_temp:
        losses['loss_kl_temp'] = utils.kl_uniform(inferred_width, uniform_prior_width)
        losses['temp_logprob'] = utils.get_uniform_logprobs(
                inferred_mean.flatten(), inferred_width.flatten(), temperatures)
        targets = torch.eq(torch.reshape(ctp, [1, -1]), torch.reshape(temperatures, [-1, 1])).double()
        preds = utils.get_preds_from_uniform(inferred_mean, inferred_width, ctp)

        losses['temp_precision'] = torch.sum(targets * preds) / torch.sum(preds)
        losses['temp_recall'] = torch.sum(targets * preds) / torch.sum(targets)
        losses['temp_corr'] = utils.get_correlation(inferred_mean.flatten(), temperatures)

    ## calculate performance based on how many particles are influenced by unobserved one/last one
    if not args.shuffle_unobserved and args.unobserved > 0:
        losses = utils_unobserved.calc_performance_per_num_influenced(
            args,
            relations,
            output,
            target,
            logits_refined,
            prob,
            mask_idx,
            losses
        )
    # print("prob")
    # print(prob.size())
    # print(prob[0])
    # print("relations")
    # print(relations.size())
    # print(relations[0])
    #################### MAIN LOSSES ####################
    ### latent losses ###
    # latent regularizer: diffusion prior replaces KL (NRI v2)
    if getattr(args, "use_diffusion", False):
        if refiner_core is None:
            raise ValueError("use_diffusion=True but refiner_core is None (unexpected).")

        logits_for_diff = logits_raw.detach() if getattr(args, "diff_detach_encoder_in_diff", False) else logits_raw

        loss_diff, diff_stats = refiner_core.loss(
            logits_for_diff,
            t_max=args.diff_train_t_max,
            num_t_samples=args.diff_train_k,
        )
        losses["loss_diffusion"] = loss_diff

        # log stats from diffusion loss if provided
        if isinstance(diff_stats, dict):
            for k, v in diff_stats.items():
                losses[k] = v

        losses["loss_kl"] = args.lambda_diff * loss_diff
    else:
        losses["loss_kl"] = utils.kl_latent(args, prob, log_prior, predicted_atoms)
    losses["acc"] = utils.edge_accuracy(logits_refined, relations)
    losses["auroc"] = utils.calc_auroc(prob, relations)
    losses["auprc"] = utils.calc_auprc(prob, relations)
    losses["jac"] = utils.calc_jac(prob, relations)


    ### output losses ###
    losses["loss_nll"] = utils.nll_gaussian(
        output, target, args.var
    ) 

    losses["loss_mse"] = F.mse_loss(output, target)
    if getattr(args, "use_diffusion", False):
        total_loss = losses["loss_nll"] + losses["loss_kl"]
    else:
        total_loss = losses["loss_nll"] + losses["loss_kl"] if args.kl else losses["loss_nll"]
    total_loss += args.teacher_forcing * losses["mse_unobserved"]
    if args.global_temp:
        total_loss += losses['loss_kl_temp']

    losses["loss"] = total_loss

    losses["inference time"] = time.time() - start

    return losses, output, unobserved, edges, prob.detach().cpu()[:, :, 1].numpy()
