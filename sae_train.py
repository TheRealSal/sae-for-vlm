from dictionary_learning import ActivationBuffer, AutoEncoder, JumpReluAutoEncoder
from dictionary_learning.trainers import *
from dictionary_learning.training import trainSAE
from torch.utils.data import DataLoader
from datasets.activations import ActivationsDataset
import torch
from pathlib import Path
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser("Train Sparse Autoencoder", add_help=False)
    parser.add_argument("--sae_model", default="jumprelu", type=str)
    parser.add_argument("--activations_dir", required=True, type=str)
    parser.add_argument("--val_activations_dir", required=True, type=str)
    parser.add_argument("--checkpoints_dir", default="./output_dir", type=str)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expansion_factor", type=int, default=1)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save_steps", type=int, default=1_000)
    parser.add_argument("--log_steps", type=int, default=50)
    # JumpRelu
    parser.add_argument("--bandwidth", type=float, default=0.001)
    parser.add_argument("--sparsity_penalty", type=float, default=0.1)
    # Standard
    parser.add_argument("--l1_penalty", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--resample_steps", type=int, default=None)
    # TopK + Batch TopK
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--auxk_alpha", type=float, default=1/32)
    parser.add_argument("--decay_start", type=int, default=1_000_000)
    # Batch TopK
    parser.add_argument("--threshold_beta", type=float, default=0.999)
    parser.add_argument("--threshold_start_step", type=int, default=1000)
    # MatryoshkaBatchTopK
    parser.add_argument("--group_fractions", type=float, nargs="+")
    # LinearIDOL
    parser.add_argument("--tau", type=int, default=20)
    parser.add_argument("--w", type=float, default=0.5)
    parser.add_argument("--noise_mode", type=str, default="lap")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--l_mse_Zt", type=float, default=0.0)
    parser.add_argument("--l_ind", type=float, default=0.1)
    parser.add_argument("--l_spB", type=float, default=0.01)
    parser.add_argument("--l_spM", type=float, default=0.01)
    parser.add_argument("--l_spZ", type=float, default=0.01)

    return parser

def train_sae(args):
    dataset = ActivationsDataset(args.activations_dir, device=torch.device(args.device))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = ActivationsDataset(args.val_activations_dir, device=torch.device(args.device))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    sample = next(iter(dataloader))
    print(sample.shape)

    activation_dim = sample.shape[1]
    dictionary_size = args.expansion_factor * activation_dim

    trainers = {
        'jumprelu': JumpReluTrainer,
        'standard': StandardTrainer,
        'batch_top_k': BatchTopKTrainer,
        'top_k': TopKTrainer,
        'matroyshka_batch_top_k': MatryoshkaBatchTopKTrainer,
        'linear_idol': LinearIDOLTrainer,
    }

    # autoencoders = {
    #     'jumprelu': JumpReluAutoEncoder,
    #     'standard': AutoEncoder,
    #     'batch_top_k': BatchTopKSAE,
    #     'top_k': AutoEncoderTopK,
    # }

    trainer_cfg = {
        "trainer": trainers[args.sae_model],
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": args.lr,
        "device": args.device,
        "steps": args.steps,
        "layer": "",
        "lm_name": "",
        "submodule_name": ""
    }

    if args.sae_model == "jumprelu":
        trainer_cfg["bandwidth"] = args.bandwidth
        trainer_cfg["sparsity_penalty"] = args.sparsity_penalty
    if args.sae_model == "standard":
        trainer_cfg["l1_penalty"] = args.l1_penalty
        trainer_cfg["warmup_steps"] = args.warmup_steps
        trainer_cfg["resample_steps"] = args.resample_steps
    if args.sae_model == "top_k" or args.sae_model == "batch_top_k" or args.sae_model == "matroyshka_batch_top_k":
        trainer_cfg["k"] = args.k
        trainer_cfg["auxk_alpha"] = args.auxk_alpha
        trainer_cfg["decay_start"] = args.decay_start
    if args.sae_model == "batch_top_k" or args.sae_model == "matroyshka_batch_top_k":
        trainer_cfg["threshold_beta"] = args.threshold_beta
        trainer_cfg["threshold_start_step"] = args.threshold_start_step
    if args.sae_model == "matroyshka_batch_top_k":
        trainer_cfg["group_fractions"] = args.group_fractions
    if args.sae_model == "linear_idol":
        trainer_cfg["tau"] = args.tau
        trainer_cfg["w"] = args.w
        trainer_cfg["noise_mode"] = args.noise_mode
        trainer_cfg["topk_sparsity"] = args.k
        trainer_cfg["mode"] = "instantaneous"
        trainer_cfg["wd"] = args.wd
        trainer_cfg["warmup_steps"] = args.warmup_steps
        trainer_cfg["decay_start"] = args.decay_start
        trainer_cfg["l_mse_Zt"] = args.l_mse_Zt
        trainer_cfg["l_ind"] = args.l_ind
        trainer_cfg["l_spB"] = args.l_spB
        trainer_cfg["l_spM"] = args.l_spM
        trainer_cfg["l_spZ"] = args.l_spZ

    dataset_name = Path(args.activations_dir).name
    if args.sae_model == "linear_idol":
        run_suffix = f"tau{args.tau}_k{args.k}"
    else:
        run_suffix = str(args.k)
    save_dir = Path(args.checkpoints_dir) / f"{dataset_name}_{args.sae_model}_{run_suffix}_x{args.expansion_factor}"
    save_dir.mkdir(parents=True, exist_ok=True)

    ae = trainSAE(
        data=dataloader,
        val_data=val_dataloader,
        trainer_configs=[trainer_cfg],
        use_wandb=True,
        wandb_entity="mateuszpach",
        wandb_project="Clip SAE",
        steps=args.steps,
        save_steps=[x for x in range(0, args.steps, args.save_steps)],
        save_dir=save_dir,
        log_steps=args.log_steps,
    )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    train_sae(args)