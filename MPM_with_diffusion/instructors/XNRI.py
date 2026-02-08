import torch
import config as cfg
from torch.nn.functional import mse_loss
from utils.torch_extension import edge_accuracy, asym_rate, transpose
from instructors.base import Instructor
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch import Tensor, optim
from utils.metric import cross_entropy, kl_divergence, nll_gaussian
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
import time


class XNRIIns(Instructor):
    """
    Training and testing for the neural relational inference task.
    """
    def __init__(self, model: torch.nn.DataParallel, data: dict, es: np.ndarray, cmd):
        """
        Args:
            model: an auto-encoder
            data: train / val /test set
            es: edge list
            cmd: command line parameters
        """
        super(XNRIIns, self).__init__(cmd)
        self.model = model
        self.data = {key: TensorDataset(value[0], value[1])
                     for key, value in data.items()}
        self.es = torch.LongTensor(es)
        # number of nodes
        self.size = cmd.size
        self.batch_size = cmd.batch
        self.time_steps = cmd.b_time_steps
        self.b_walltime = cmd.b_walltime
        # optimizer
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)
        # learning rate scheduler, same as in NRI
        self.scheduler = StepLR(self.opt, step_size=cfg.lr_decay, gamma=cfg.gamma)

    def train(self, save_folder, start_time):
        # use the loss as the metric for model selection, default: +\infty
        val_best = np.inf
        # path to save the current best model

        # original:
        # prefix = '/'.join(cfg.log.split('/')[:-1])

        # customized:
        prefix = save_folder
        name = '{}/best.pth'.format(prefix)
        for epoch in range(1, 1 + self.cmd.epochs):
            t_epoch_start = time.time()
            self.model.train()
            # diffusion logging buffers (reset per-epoch)
            self._diff_train = []
            self._diff_ddpm_mse_train = []

            # shuffle the data at each epoch
            data = self.load_data(self.data['train'], self.batch_size)
            loss_a = 0.
            N = 0.
            for _, states in data:
                if cfg.gpu:
                    states = states.cuda()
                scale = len(states) / self.batch_size
                # N: number of samples, equal to the batch size with possible exception for the last batch
                N += scale
                loss_a += scale * self.train_nri(states, self.time_steps)
            loss_a /= N 
            self.log.info('epoch {:03d} loss {:.3e}'.format(epoch, loss_a))
            if getattr(self.cmd, 'use_diff_prior', False) and len(getattr(self, '_diff_train', [])) > 0:
                self.log.info('epoch {:03d} diff {:.3e} diff_ddpm_mse {:.3e}'.format(
                    epoch,
                    float(np.mean(self._diff_train)),
                    float(np.mean(self._diff_ddpm_mse_train)) if len(self._diff_ddpm_mse_train) else float('nan'),
                ))
            if self.time_steps == 49:
                losses = self.report('val', [cfg.M])
            else:
                losses = self.report('val', [1])

            val_cur = losses[0]
            if val_cur < val_best:
                # update the current best model when approaching a lower loss
                self.log.info('epoch {:03d} metric {:.3e}'.format(epoch, val_cur))
                val_best = val_cur
                torch.save(self.model.module.state_dict(), name)

            # learning rate scheduling
            self.scheduler.step()
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - t_epoch_start
            if self.b_walltime:
                if epoch_end_time - start_time < 171900 - epoch_time:
                    continue
                else:
                    break
        if self.cmd.epochs > 0:
            self.model.module.load_state_dict(torch.load(name))
        print("------Now Testing!------")
        self.test('test', 20, save_folder)

    def report(self, name: str, Ms: list) -> list:
        """
        Evaluate the loss.

        Args:
            name: 'train' / 'val' / 'test'
            Ms: [...], each element is a number of steps to predict
        
        Return:
            losses: [...], each element is an average loss
        """
        losses = []
        for M in Ms:
            loss, mse, acc, rate, ratio, sparse = self.evalate(self.data[name], M)
            losses.append(loss)
            self.log.info('{} M {:02d} mse {:.3e} acc {:.4f} _acc {:.4f} rate {:.4f} ratio {:.4f} sparse {:.4f}'.format(
                name, M, mse, acc, 1 - acc, rate, ratio, sparse))
        return losses

    # ---------------- Diffusion prior helpers ----------------
    def _diff_prior_module(self):
        """Return DiffusionPrior module if enabled, else None."""
        if (not getattr(self.cmd, "use_diff_prior", False)):
            return None
        # model is DataParallel; diffusion prior is stored inside the wrapped module
        if hasattr(self.model, "module") and hasattr(self.model.module, "diff_prior"):
            return self.model.module.diff_prior
        return None

    def _compute_diff_kl_term(self, logits_for_diff: Tensor):
        """
        Compute diffusion loss term that replaces the uniform KL term.

        Args:
            logits_for_diff: z0 logits from encoder, shape [B, E, K] (deterministic)

        Return:
            loss_kl: scalar tensor (already multiplied by lambda_diff)
            stats: dict for logging (may be empty)
        """
        diff_prior = self._diff_prior_module()
        if diff_prior is None:
            return None, {}

        z0 = logits_for_diff.detach() if getattr(self.cmd, "diff_detach_encoder_in_diff", False) else logits_for_diff
        out = diff_prior.loss(
            z0,
            t_max=getattr(self.cmd, "diff_train_t_max", None),
            num_t_samples=getattr(self.cmd, "diff_train_k", None),
        )
        if isinstance(out, tuple):
            loss_diff, stats = out
        else:
            loss_diff, stats = out, {}

        # In train_diff2.py: loss_kl := lambda_diff * loss_diff
        loss_kl = getattr(self.cmd, "lambda_diff", 1.0) * loss_diff
        return loss_kl, stats
    # --------------------------------------------------------

    def train_nri(self, states: Tensor, time_steps) -> Tensor:
        """
        Args:
            states: [batch, step, node, dim], all node states, including historical states and the states to predict
        """
        use_diff = bool(getattr(self.cmd, "use_diff_prior", False))
        # compute the relation distribution (prob) and predict future node states (output)
        if time_steps < 49:
            out = self.model(states, states, p=True, M=1, tosym=cfg.sym, return_logits_for_diff=use_diff)
        else:
            out = self.model(states, states, p=True, M=cfg.M, tosym=cfg.sym, return_logits_for_diff=use_diff)

        if use_diff:
            output, prob, logits_for_diff = out
        else:
            output, prob = out
            logits_for_diff = None

        # prob is [B, E, K] from model; transpose to [E, B, K] for existing metrics/utilities
        prob = prob.transpose(0, 1).contiguous()

        # reconstruction loss
        loss_nll = nll_gaussian(output, states[:, 1:], 5e-5)

        # KL term (replaced by diffusion loss when enabled)
        if use_diff and logits_for_diff is not None:
            loss_kl, stats = self._compute_diff_kl_term(logits_for_diff)
            if loss_kl is None:
                # fallback (shouldn't happen if DiffusionPrior is correctly attached)
                loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
            else:
                # log diffusion stats (store unscaled diffusion loss)
                lam = float(getattr(self.cmd, "lambda_diff", 1.0))
                if hasattr(self, "_diff_train"):
                    diff_loss_val = float((loss_kl.detach().cpu().item() / lam) if lam != 0.0 else float("nan"))
                    self._diff_train.append(diff_loss_val)
                if hasattr(self, "_diff_ddpm_mse_train") and isinstance(stats, dict) and ("diff_ddpm_mse" in stats):
                    self._diff_ddpm_mse_train.append(float(stats["diff_ddpm_mse"].detach().cpu().item()))
        else:
            loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)

        loss = loss_nll + loss_kl

        # impose the soft symmetric constraint by adding a regularization term
        if self.cmd.reg > 0:
            prob_hat = transpose(prob, self.size)
            loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
            loss = loss + loss_sym * self.cmd.reg

        self.optimize(self.opt, loss * cfg.scale)

        # choice for the evaluation metric, adding the regularization term or not
        if self.cmd.no_reg:
            loss = loss_nll + loss_kl
        return loss

    def test(self, name: str, M: int, save_folder: str):
        """
        Evaluate related metrics to measure the model performance.
        The biggest difference between this function and evalute() is that, the mses are evaluated at each step.

        Args:
            name: 'train' / 'val' / 'test'
            M: number of steps to predict

        Return:
            mse_multi: mse at each step
        """
        """
        acc: accuracy of relation reconstruction
        mses: mean square error over all steps
        rate: rate of assymmetry
        ratio: relative root mean squared error
        sparse: rate of sparsity in terms of the first type of edge
        losses: loss_nll + loss_kl (+ loss_reg)
        mse_multi: mse at each step
        """
        acc, mses, rate, ratio, sparse, losses, mse_multi = [], [], [], [], [], [], []
        auroc_test = []
        probs_test = []
        data = self.load_data(self.data[name], self.batch_size)
        N = 0.
        use_diff = bool(getattr(self.cmd, "use_diff_prior", False))

        # evaluation should disable dropout (important for diffusion prior with dropout)
        self.model.eval()

        with torch.no_grad():
            for adj, states in data:
                if cfg.gpu:
                    adj = adj.cuda()
                    states = states.cuda()
                states_enc = states[:, :cfg.train_steps, :, :]
                states_dec = states[:, -cfg.train_steps:, :, :]
                target = states_dec[:, 1:]

                if self.time_steps < 49:
                    out = self.model(states_enc, states_dec, hard=True, p=True, M=1, tosym=cfg.sym, return_logits_for_diff=use_diff)
                else:
                    out = self.model(states_enc, states_dec, hard=True, p=True, M=M, tosym=cfg.sym, return_logits_for_diff=use_diff)

                if use_diff:
                    output, prob, logits_for_diff = out
                else:
                    output, prob = out
                    logits_for_diff = None

                prob = prob.transpose(0, 1).contiguous()

                # for auroc calculation
                prob_np = prob.detach().cpu().numpy()
                n_batches = adj.size()[0]
                relations_np = adj.view(n_batches, -1).detach().cpu().numpy()
                probs_test.append(prob_np.transpose(1, 0, 2))

                scale = len(states) / self.batch_size
                N += scale

                # use loss as the validation metric
                loss_nll = nll_gaussian(target, output, 5e-5)

                if use_diff and logits_for_diff is not None:
                    loss_kl, _stats = self._compute_diff_kl_term(logits_for_diff)
                    if loss_kl is None:
                        loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
                else:
                    loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)

                loss = loss_nll + loss_kl
                if self.cmd.reg > 0 and (not self.cmd.no_reg):
                    prob_hat = transpose(prob, self.size)
                    loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
                    loss = loss + loss_sym * self.cmd.reg

                # scale all metrics to match the batch size
                loss = loss * scale
                losses.append(loss)

                mses.append(scale * mse_loss(output, target).data)
                ratio.append(scale * (((output - target) ** 2).sum(-1).sqrt() / (target ** 2).sum(-1).sqrt()).mean())
                acc.append(scale * edge_accuracy(prob, adj))
                _, p = prob.max(-1)
                rate.append(scale * asym_rate(p.t(), self.size))
                sparse.append(prob.max(-1)[1].float().mean() * scale)

                # auroc at step 1
                preds_np = prob_np[:, :, 1].transpose(1, 0)
                for i in range(len(preds_np)):
                    auroc = roc_auc_score(relations_np[i], preds_np[i], average=None)
                    auroc_test.append(auroc)

                # recompute mse_multi at each step (kept from original)
                out2 = self.model(states_enc, states_dec, hard=True, p=True, M=M, tosym=cfg.sym, return_logits_for_diff=False)
                output2, prob2 = out2
                prob2 = prob2.transpose(0, 1).contiguous()
                mse_step = ((output2 - target) ** 2).mean(dim=(0, 2, -1))
                mse_step *= scale
                mse_multi.append(mse_step)

        loss = sum(losses) / N
        mses = sum(mses) / N
        mse_multi = sum(mse_multi) / N
        acc = sum(acc) / N
        rate = sum(rate) / N
        ratio = sum(ratio) / N
        sparse = sum(sparse) / N
        auroc_res = np.mean(auroc_test)

        np.save(str(save_folder) + '/results/edges_test.npy', np.vstack(probs_test))
        print("edges_test saved at: " + save_folder + '/results/edges_test.npy')
        print("auroc: {:.4f}".format(auroc_res))

        self.log.info('{} M {:02d} mse {:.3e} acc {:.4f} _acc {:.4f} rate {:.4f} ratio {:.4f} sparse {:.4f} auroc {:.4f}'.format(
                name, M, mses, acc, 1 - acc, rate, ratio, sparse, auroc_res))
        msteps = ','.join(['{:.3e}'.format(i) for i in mse_multi])
        self.log.info(msteps)
        return mse_multi


    def evalate(self, test, M: int):
        """
        Evaluate related metrics to monitor the training process.

        Args:
            test: data set to be evaluted
            M: number of steps to predict

        Return:
            loss: loss_nll + loss_kl (+ loss_reg)
            mse: mean square error over all steps
            acc: accuracy of relation reconstruction
            rate: rate of assymmetry
            ratio: relative root mean squared error
            sparse: rate of sparsity in terms of the first type of edge
        """
        acc, mse, rate, ratio, sparse, losses = [], [], [], [], [], []
        data = self.load_data(test, self.batch_size)
        N = 0.
        use_diff = bool(getattr(self.cmd, "use_diff_prior", False))

        # evaluation should disable dropout (important for diffusion prior with dropout)
        self.model.eval()

        with torch.no_grad():
            for adj, states in data:
                if cfg.gpu:
                    adj = adj.cuda()
                    states = states.cuda()
                states_enc = states[:, :cfg.train_steps, :, :]
                states_dec = states[:, -cfg.train_steps:, :, :]
                target = states_dec[:, 1:]

                out = self.model(
                    states_enc,
                    states_dec,
                    hard=True,
                    p=True,
                    M=M,
                    tosym=cfg.sym,
                    return_logits_for_diff=use_diff,
                )
                if use_diff:
                    output, prob, logits_for_diff = out
                else:
                    output, prob = out
                    logits_for_diff = None

                prob = prob.transpose(0, 1).contiguous()

                scale = len(states) / self.batch_size
                N += scale

                # use loss as the validation metric
                loss_nll = nll_gaussian(target, output, 5e-5)

                if use_diff and logits_for_diff is not None:
                    loss_kl, _stats = self._compute_diff_kl_term(logits_for_diff)
                    if loss_kl is None:
                        loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
                else:
                    loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)

                loss = loss_nll + loss_kl
                if self.cmd.reg > 0 and (not self.cmd.no_reg):
                    prob_hat = transpose(prob, self.size)
                    loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
                    loss = loss + loss_sym * self.cmd.reg

                # scale all metrics to match the batch size
                loss = loss * scale
                losses.append(loss)

                mse.append(scale * mse_loss(output, target).data)
                ratio.append(scale * (((output - target) ** 2).sum(-1).sqrt() / (target ** 2).sum(-1).sqrt()).mean())
                acc.append(scale * edge_accuracy(prob, adj))
                _, p = prob.max(-1)
                rate.append(scale * asym_rate(p.t(), self.size))
                sparse.append(prob.max(-1)[1].float().mean() * scale)

        loss = sum(losses) / N
        mse = sum(mse) / N
        acc = sum(acc) / N
        rate = sum(rate) / N
        ratio = sum(ratio) / N
        sparse = sum(sparse) / N
        return loss, mse, acc, rate, ratio, sparse

