# import time
# import os
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import pandas as pd
# from collections import defaultdict
# import itertools


# class Logger:
#     def __init__(self, args):
#         self.args = args

#         self.train_losses = pd.DataFrame()
#         self.train_losses_idx = 0

#         self.test_losses = pd.DataFrame()
#         self.test_losses_idx = 0

#         if args.validate:
#             self.val_losses = pd.DataFrame()
#             self.val_losses_idx = 0
#         else:
#             self.val_losses = None

#         self.num_models_to_keep = 1
#         assert self.num_models_to_keep > 0, "Dont delete all models!!!"

#         self.create_log_path(args)

#         self.path = self.path_return(args)

#     def create_log_path(self, args, add_path_var=""):
#         name = args.suffix + '_' + str(args.time)
#         args.log_path = os.path.join(args.save_folder, add_path_var, name)

#         if not os.path.exists(args.log_path):
#             os.makedirs(args.log_path)

#         if args.expername != "":
#             sympath = os.path.join(args.sym_save_folder, args.expername)
#             if os.path.islink(sympath):
#                 os.remove(sympath)
#             ## check whether args.log_path is absolute path and if not concatenate with current working directory
#             if os.path.isabs(args.log_path):
#                 log_link = args.log_path
#             else:
#                 log_link = os.path.join(os.getcwd(), args.log_path)
#             os.symlink(log_link, sympath)

#         self.log_file = os.path.join(args.log_path, "log.txt")
#         self.write_to_log_file(args)

#         args.encoder_file = os.path.join(args.log_path, "encoder.pt")
#         args.decoder_file = os.path.join(args.log_path, "decoder.pt")
#         args.optimizer_file = os.path.join(args.log_path, "optimizer.pt")

#         args.plotdir = os.path.join(args.log_path, "plots")
#         if not os.path.exists(args.plotdir):
#             os.makedirs(args.plotdir)

#     def path_return(self, args):
#         return args.log_path

#     def save_checkpoint(self, args, encoder, decoder, optimizer, specifier=""):
#         args.encoder_file = os.path.join(args.log_path, "encoder" + specifier + ".pt")
#         args.decoder_file = os.path.join(args.log_path, "decoder" + specifier + ".pt")
#         args.optimizer_file = os.path.join(
#             args.log_path, "optimizer" + specifier + ".pt"
#         )

#         if encoder is not None:
#             torch.save(encoder.state_dict(), args.encoder_file)
#         if decoder is not None:
#             torch.save(decoder.state_dict(), args.decoder_file)
#         if optimizer is not None:
#             torch.save(optimizer.state_dict(), args.optimizer_file)

#     def write_to_log_file(self, string):
#         """
#         Write given string in log-file and print as terminal output
#         """
#         print(string)
#         cur_file = open(self.log_file, "a")
#         print(string, file=cur_file)
#         cur_file.close()

#     def create_log(
#         self,
#         args,
#         encoder=None,
#         decoder=None,
#         accuracy=None,
#         optimizer=None,
#         final_test=False,
#         test_losses=None,
#     ):

#         print("Saving model and log-file to " + args.log_path)

#         # Save losses throughout training and plot
#         self.train_losses.to_pickle(os.path.join(self.args.log_path, "train_loss"))

#         if self.val_losses is not None:
#             self.val_losses.to_pickle(os.path.join(self.args.log_path, "val_loss"))

#         if accuracy is not None:
#             np.save(os.path.join(self.args.log_path, "accuracy"), accuracy)

#         specifier = ""
#         if final_test:
#             pd_test_losses = pd.DataFrame(
#                 [
#                     [k] + [np.mean(v)]
#                     for k, v in test_losses.items()
#                     if type(v) != defaultdict
#                 ],
#                 columns=["loss", "score"],
#             )
#             pd_test_losses.to_pickle(os.path.join(self.args.log_path, "test_loss"))

#             pd_test_losses_per_influenced = pd.DataFrame(
#                 list(
#                     itertools.chain(
#                         *[
#                             [
#                                 [k]
#                                 + [idx]
#                                 + [np.mean(list(itertools.chain.from_iterable(elem)))]
#                                 for idx, elem in sorted(v.items())
#                             ]
#                             for k, v in test_losses.items()
#                             if type(v) == defaultdict
#                         ]
#                     )
#                 ),
#                 columns=["loss", "num_influenced", "score"],
#             )
#             pd_test_losses_per_influenced.to_pickle(
#                 os.path.join(self.args.log_path, "test_loss_per_influenced")
#             )

#             specifier = "final"

#         # Save the model checkpoint
#         self.save_checkpoint(args, encoder, decoder, optimizer, specifier=specifier)

#     def draw_loss_curves(self):
#         for i in self.train_losses.columns:
#             plt.figure()
#             plt.plot(self.train_losses[i], "-b", label="train " + i)

#             if self.val_losses is not None and i in self.val_losses:
#                 plt.plot(self.val_losses[i], "-r", label="val " + i)

#             plt.xlabel("epoch")
#             plt.ylabel("loss")
#             plt.legend(loc="upper right")

#             # save image
#             plt.savefig(os.path.join(self.args.log_path, i + ".png"))
#             plt.close()

#     def append_train_loss(self, loss):
#         for k, v in loss.items():
#             self.train_losses.at[str(self.train_losses_idx), k] = np.mean(v)
#         self.train_losses_idx += 1

#     def append_val_loss(self, val_loss):
#         for k, v in val_loss.items():
#             self.val_losses.at[str(self.val_losses_idx), k] = np.mean(v)
#         self.val_losses_idx += 1

#     def append_test_loss(self, test_loss):
#         for k, v in test_loss.items():
#             if type(v) != defaultdict:
#                 self.test_losses.at[str(self.test_losses_idx), k] = np.mean(v)
#         self.test_losses_idx += 1

#     def result_string(self, trainvaltest, epoch, losses, t=None):
#         string = ""
#         if trainvaltest == "test":
#             string += (
#                 "-------------------------------- \n"
#                 "--------Testing----------------- \n"
#                 "-------------------------------- \n"
#             )
#         else:
#             string += str(epoch) + " " + trainvaltest + "\t \t"

#         for loss, value in losses.items():
#             if type(value) == defaultdict:
#                 string += loss + " "
#                 for idx, elem in sorted(value.items()):
#                     string += str(idx) + ": {:.10f} \t".format(
#                         np.mean(list(itertools.chain.from_iterable(elem)))
#                     )
#             elif np.mean(value) != 0 and not math.isnan(np.mean(value)):
#                 string += loss + " {:.10f} \t".format(np.mean(value))

#         if t is not None:
#             string += "time: {:.4f}s \t".format(time.time() - t)

#         return string

import os
import sys
import time
import math
import pprint
import datetime
import itertools
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Logger:
    def __init__(self, args):
        self.args = args

        self.train_losses = pd.DataFrame()
        self.train_losses_idx = 0

        self.test_losses = pd.DataFrame()
        self.test_losses_idx = 0

        if args.validate:
            self.val_losses = pd.DataFrame()
            self.val_losses_idx = 0
        else:
            self.val_losses = None

        self.num_models_to_keep = 1
        assert self.num_models_to_keep > 0, "Dont delete all models!!!"

        # Prepare folders + paths
        self.create_log_path(args)

        # Public attribute used by your train.py
        self.path = self.path_return(args)

    # -------------------------
    # New: two-log-file behavior
    # -------------------------
    def _maybe_write_header(self, args):
        """
        Write command + parsed args to log_all.txt once (when file doesn't exist or is empty).
        """
        if not hasattr(self, "log_all_path") or self.log_all_path is None:
            return

        need_header = (not os.path.exists(self.log_all_path)) or (os.path.getsize(self.log_all_path) == 0)
        if not need_header:
            return

        now = datetime.datetime.now().isoformat()
        cmd = " ".join(sys.argv)
        args_dict = vars(args)

        with open(self.log_all_path, "a", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write(f"Start time: {now}\n")
            f.write(f"Command: {cmd}\n")
            f.write("Parsed args:\n")
            f.write(pprint.pformat(args_dict, sort_dicts=True) + "\n")
            f.write("=" * 100 + "\n\n")

    def _format_losses(self, losses: dict) -> str:
        """
        losses is usually a dict of scalars (already averaged), e.g. {"loss":0.1, "acc":0.9}
        """
        keys = sorted(losses.keys())
        parts = []
        for k in keys:
            v = losses[k]
            try:
                parts.append(f"{k}={float(v):.6g}")
            except Exception:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    def log_epoch_all(self, epoch: int, split: str, losses: dict):
        """
        Append one line to log_all.txt each epoch.
        """
        with open(self.log_all_path, "a", encoding="utf-8") as f:
            f.write(f"[{split}] epoch={epoch:04d}  {self._format_losses(losses)}\n")

    def log_epoch_best(
        self,
        epoch: int,
        metric_name: str,
        metric_value,
        train_losses=None,
        val_losses=None,
        test_losses=None,
    ):
        """
        Append only when best improves (to log.txt).
        """
        with open(self.log_best_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            try:
                mv = float(metric_value)
                f.write(f"BEST UPDATE @ epoch={epoch:04d}  {metric_name}={mv:.6g}\n")
            except Exception:
                f.write(f"BEST UPDATE @ epoch={epoch:04d}  {metric_name}={metric_value}\n")

            if train_losses is not None:
                f.write(f"[train] {self._format_losses(train_losses)}\n")
            if val_losses is not None:
                f.write(f"[val]   {self._format_losses(val_losses)}\n")
            if test_losses is not None:
                f.write(f"[test]  {self._format_losses(test_losses)}\n")
            f.write("\n")

    def write_to_log_all_file(self, string: str):
        """
        Write given string in log_all.txt and print as terminal output.
        """
        print(string)
        with open(self.log_all_path, "a", encoding="utf-8") as f:
            print(string, file=f)

    def write_to_best_log_file(self, string: str):
        """
        Write given string in log.txt (best-only) and print as terminal output.
        Use this ONLY when best improves (or final summary).
        """
        print(string)
        with open(self.log_best_path, "a", encoding="utf-8") as f:
            print(string, file=f)

    # -------------------------
    # Paths / folders
    # -------------------------
    def create_log_path(self, args, add_path_var=""):
        # fixed folder name
        name = "outcome"
        args.log_path = os.path.join(args.save_folder, add_path_var, name)
        os.makedirs(args.log_path, exist_ok=True)

        # optional symlink behavior (keep original)
        if args.expername != "":
            sympath = os.path.join(args.sym_save_folder, args.expername)
            if os.path.islink(sympath):
                os.remove(sympath)

            # check whether args.log_path is absolute path and if not concatenate with cwd
            if os.path.isabs(args.log_path):
                log_link = args.log_path
            else:
                log_link = os.path.join(os.getcwd(), args.log_path)

            os.symlink(log_link, sympath)

        # two log files
        self.log_all_path = os.path.join(args.log_path, "log_all.txt")
        self.log_best_path = os.path.join(args.log_path, "log.txt")

        # write header into log_all
        self._maybe_write_header(args)

        # model checkpoint files
        args.encoder_file = os.path.join(args.log_path, "encoder.pt")
        args.decoder_file = os.path.join(args.log_path, "decoder.pt")
        args.diffusion_prior_file = os.path.join(args.log_path, "diffusion_prior.pt")
        args.refiner_file = args.diffusion_prior_file  # backward-compatible alias

        args.optimizer_file = os.path.join(args.log_path, "optimizer.pt")

        # plots
        args.plotdir = os.path.join(args.log_path, "plots")
        os.makedirs(args.plotdir, exist_ok=True)

        # (Optional) write a short startup line to log_all (NOT to log.txt)
        self.write_to_log_all_file(f"Logging to: {args.log_path}")

    def path_return(self, args):
        return args.log_path

    # -------------------------
    # Original checkpoint + plots
    # -------------------------
    def save_checkpoint(self, args, encoder, decoder, optimizer, refiner=None, specifier=""):

        args.encoder_file = os.path.join(args.log_path, "encoder" + specifier + ".pt")
        args.decoder_file = os.path.join(args.log_path, "decoder" + specifier + ".pt")
        args.diffusion_prior_file = os.path.join(args.log_path, "diffusion_prior" + specifier + ".pt")
        args.refiner_file = args.diffusion_prior_file  # alias

        args.optimizer_file = os.path.join(args.log_path, "optimizer" + specifier + ".pt")
        if refiner is not None:
            torch.save(refiner.state_dict(), args.diffusion_prior_file)

        if encoder is not None:
            torch.save(encoder.state_dict(), args.encoder_file)
        if decoder is not None:
            torch.save(decoder.state_dict(), args.decoder_file)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), args.optimizer_file)

    # Keep original method name for backward compatibility
    def write_to_log_file(self, string):
        """
        Backward compatible: previously wrote to log.txt.
        Now redirect it to log_all.txt to avoid polluting best-only log.
        """
        self.write_to_log_all_file(string)

    def create_log(
        self,
        args,
        encoder=None,
        decoder=None,
        refiner=None,
        accuracy=None,
        optimizer=None,
        final_test=False,
        test_losses=None,
    ):
        """
        Save models and loss curves/metrics.
        NOTE: This should be called when best improves (or at final_test).
        """
        print("Saving model and log-file to " + args.log_path)

        # Save losses throughout training and plot
        self.train_losses.to_pickle(os.path.join(self.args.log_path, "train_loss"))

        if self.val_losses is not None:
            self.val_losses.to_pickle(os.path.join(self.args.log_path, "val_loss"))

        if accuracy is not None:
            np.save(os.path.join(self.args.log_path, "accuracy"), accuracy)

        specifier = ""
        if final_test:
            pd_test_losses = pd.DataFrame(
                [
                    [k] + [np.mean(v)]
                    for k, v in test_losses.items()
                    if type(v) != defaultdict
                ],
                columns=["loss", "score"],
            )
            pd_test_losses.to_pickle(os.path.join(self.args.log_path, "test_loss"))

            pd_test_losses_per_influenced = pd.DataFrame(
                list(
                    itertools.chain(
                        *[
                            [
                                [k] + [idx] + [np.mean(list(itertools.chain.from_iterable(elem)))]
                                for idx, elem in sorted(v.items())
                            ]
                            for k, v in test_losses.items()
                            if type(v) == defaultdict
                        ]
                    )
                ),
                columns=["loss", "num_influenced", "score"],
            )
            pd_test_losses_per_influenced.to_pickle(
                os.path.join(self.args.log_path, "test_loss_per_influenced")
            )

            specifier = "final"

        # Save the model checkpoint
        self.save_checkpoint(args, encoder, decoder, optimizer, refiner=refiner, specifier=specifier)


    def draw_loss_curves(self):
        for i in self.train_losses.columns:
            plt.figure()
            plt.plot(self.train_losses[i], "-b", label="train " + i)

            if self.val_losses is not None and i in self.val_losses:
                plt.plot(self.val_losses[i], "-r", label="val " + i)

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(loc="upper right")

            # save image
            plt.savefig(os.path.join(self.args.log_path, i + ".png"))
            plt.close()

    def append_train_loss(self, loss):
        for k, v in loss.items():
            self.train_losses.at[str(self.train_losses_idx), k] = np.mean(v)
        self.train_losses_idx += 1

    def append_val_loss(self, val_loss):
        for k, v in val_loss.items():
            self.val_losses.at[str(self.val_losses_idx), k] = np.mean(v)
        self.val_losses_idx += 1

    def append_test_loss(self, test_loss):
        for k, v in test_loss.items():
            if type(v) != defaultdict:
                self.test_losses.at[str(self.test_losses_idx), k] = np.mean(v)
        self.test_losses_idx += 1

    def result_string(self, trainvaltest, epoch, losses, t=None):
        string = ""
        if trainvaltest == "test":
            string += (
                "-------------------------------- \n"
                "--------Testing----------------- \n"
                "-------------------------------- \n"
            )
        else:
            string += str(epoch) + " " + trainvaltest + "\t \t"

        for loss, value in losses.items():
            if type(value) == defaultdict:
                string += loss + " "
                for idx, elem in sorted(value.items()):
                    string += str(idx) + ": {:.10f} \t".format(
                        np.mean(list(itertools.chain.from_iterable(elem)))
                    )
            elif np.mean(value) != 0 and not math.isnan(np.mean(value)):
                string += loss + " {:.10f} \t".format(np.mean(value))

        if t is not None:
            string += "time: {:.4f}s \t".format(time.time() - t)

        return string
