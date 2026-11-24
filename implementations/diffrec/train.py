"""Train DiffRec variants from scratch with the published data splits."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
import torch

from data import add_random_noise, load_data
from diffusion_loss import training_loss
from prepare import DATASETS, REVISION, ROOT, UPSTREAM, verify_upstream
from presets import METRICS, MODELS
from ranking import evaluate

UPSTREAM_MODELS = {
    "diffrec": "DiffRec", "l-diffrec": "L-DiffRec",
    "t-diffrec": "T-DiffRec", "lt-diffrec": "LT-DiffRec",
}


def run(args, config):
    verify_upstream()
    sys.path.insert(0, str(UPSTREAM / UPSTREAM_MODELS[args.model]))
    from models.DNN import DNN
    from models.gaussian_diffusion import GaussianDiffusion, ModelMeanType

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA indisponível. Confira o PyTorch e o driver ou use --device cpu.")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    data = load_data(args.dataset, config.time_min)
    if args.random_noise:
        data = add_random_noise(data, args.random_noise, args.seed)
    n_users, n_items = data.train.shape
    autoencoder = None
    dimensions = [*config.hidden, n_items]
    if config.ae_lr is not None:
        from models.Autoencoder import AutoEncoder, compute_loss

        embeddings_path = DATASETS / args.dataset / "item_emb.npy"
        if not embeddings_path.is_file():
            raise ValueError(f"Embedding oficial ausente: {embeddings_path}. Consulte o README.")
        embeddings = np.load(embeddings_path, allow_pickle=False)
        if embeddings.ndim != 2 or embeddings.shape[0] != n_items or not np.isfinite(embeddings).all():
            raise ValueError(f"Embedding incompatível com os {n_items} itens: {embeddings_path}.")
        autoencoder = AutoEncoder(torch.from_numpy(embeddings), args.categories, [300], [], device, "tanh").to(device)
        if args.categories > 1:
            autoencoder.category_idx = [indices.to(device) for indices in autoencoder.category_idx]
            original_order = torch.argsort(autoencoder.category_map).to(device)
        dimensions[-1] = 300

    model = DNN(dimensions[::-1], dimensions, 10, time_type="cat").to(device)
    predict_noise = args.target == "eps"
    mean_type = ModelMeanType.EPSILON if predict_noise else ModelMeanType.START_X
    diffusion = GaussianDiffusion(mean_type, "linear-var", config.scale, config.noise_min,
                                  config.noise_max, config.steps, device).to(device)
    groups = [{"params": model.parameters(), "lr": config.lr}]
    if autoencoder is not None:
        groups.append({"params": autoencoder.parameters(), "lr": config.ae_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)

    def dense(batch):
        return torch.from_numpy(batch.toarray()).to(device)

    @torch.no_grad()
    def predict(batch):
        inputs = dense(batch)
        if autoencoder is None:
            return diffusion.p_sample(model, inputs, args.sampling_steps, args.sampling_noise)
        _, latent, _ = autoencoder.Encode(inputs)
        latent = diffusion.p_sample(model, latent, args.sampling_steps, args.sampling_noise)
        scores = autoencoder.Decode(latent)
        return scores[:, original_order] if args.categories > 1 else scores

    name = f"{args.model}-{args.dataset}-seed{args.seed}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    output = args.output or ROOT / "runs" / name
    output.mkdir(parents=True, exist_ok=False)
    print(f"{args.model} | {args.dataset} | {n_users} usuários, {n_items} itens | {device} | seed={args.seed}")
    print(f"Saída: {output}", flush=True)
    history = []
    best_valid = None
    best_epoch = 0
    updates = 0
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if autoencoder is not None:
            autoencoder.train()
        epoch_start = perf_counter()
        epoch_loss = torch.zeros((), device=device, dtype=torch.float64)
        users = torch.randperm(n_users).numpy()
        for offset in range(0, n_users, config.batch_size):
            batch = dense(data.inputs[users[offset:offset + config.batch_size]])
            optimizer.zero_grad(set_to_none=True)
            if autoencoder is None:
                loss, _ = training_loss(diffusion, model, batch, config.reweight, predict_noise)
            else:
                target, latent, kl = autoencoder.Encode(batch)
                elbo, reconstruction = training_loss(diffusion, model, latent, config.reweight, predict_noise)
                anneal = min(0.3, updates / 200)
                lamda = max((1 - (epoch - 1) / 500) * config.lamda, 0.005)
                vae = compute_loss(autoencoder.Decode(reconstruction), target) + anneal * kl
                loss = lamda * elbo + vae if config.reweight else elbo + lamda * vae
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * len(batch)
            updates += 1

        loss_value = epoch_loss.item() / n_users
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss não finita na época {epoch}.")
        entry = {"epoch": epoch, "loss": loss_value}
        print(f"Época {epoch:03d} | loss={loss_value:.6f} | {perf_counter() - epoch_start:.1f}s", end="")
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            if autoencoder is not None:
                autoencoder.eval()
            valid = evaluate(predict, data.inputs, data.valid, data.train, config.batch_size)
            entry["valid"] = valid
            print(f" | validação R@20={valid['recall@20']:.4f}", end="")
            if best_valid is None or valid["recall@20"] > best_valid["recall@20"]:
                best_valid, best_epoch = valid, epoch
                checkpoint = {"model": model.state_dict()}
                if autoencoder is not None:
                    checkpoint["autoencoder"] = autoencoder.state_dict()
                    if args.categories > 1:
                        checkpoint["category_map"] = autoencoder.category_map
                temp = output / "best.pt.tmp"
                torch.save(checkpoint, temp)
                temp.replace(output / "best.pt")
        print(flush=True)
        history.append(entry)
        # Upstream stops before the epoch at distance 20 from the best validation.
        if best_epoch and epoch - best_epoch >= 19:
            break

    parameters = sum(param.numel() for group in optimizer.param_groups for param in group["params"])
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if autoencoder is not None:
        autoencoder.load_state_dict(checkpoint["autoencoder"])
        autoencoder.eval()
    test_inputs = data.inputs + data.valid if args.test_with_validation else data.inputs
    test = evaluate(predict, test_inputs, data.test, data.train + data.valid, config.batch_size)
    reference = None if args.random_noise else dict(zip(METRICS, config.reference))
    difference = {key: round(test[key] - reference[key], 4) for key in METRICS} if reference else None
    result = {
        "model": args.model, "dataset": args.dataset, "seed": args.seed,
        "revision": REVISION, "device": str(device), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "users": n_users, "items": n_items,
        "interactions": {split: getattr(data, split).nnz for split in ("train", "valid", "test")},
        "parameters": parameters,
        "cuda_peak_allocated_mib": (
            round(torch.cuda.max_memory_allocated(device) / 2**20, 2) if device.type == "cuda" else None
        ),
        "elapsed_seconds": round(perf_counter() - start, 2),
        "settings": {**asdict(config), "epochs": args.epochs, "target": args.target,
                     "categories": args.categories, "sampling_steps": args.sampling_steps,
                     "sampling_noise": args.sampling_noise, "threads": args.threads,
                     "test_with_validation": args.test_with_validation, "random_noise": args.random_noise},
        "best_epoch": best_epoch, "valid": best_valid, "test": test,
        "paper_table": config.table if reference else None, "paper_test": reference,
        "difference": difference,
        "history": history,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Melhor época: {best_epoch} | Resultados de teste")
    for metric in METRICS:
        comparison = f"  artigo {reference[metric]:.4f}  delta {difference[metric]:+.4f}" if reference else ""
        print(f"{metric:9}  local {test[metric]:.4f}{comparison}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="diffrec")
    parser.add_argument("--dataset", choices=MODELS["diffrec"], default="ml-1m_clean")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--categories", type=int, default=2)
    parser.add_argument("--target", choices=("x0", "eps"), default="x0")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--noise-scale", type=float)
    parser.add_argument("--random-noise", type=float, default=0.0,
                        help="proporção de itens aleatórios adicionados ao treino e à validação")
    parser.add_argument("--sampling-steps", type=int, default=0)
    parser.add_argument("--sampling-noise", action="store_true")
    parser.add_argument("--test-with-validation", action="store_true",
                        help="inclui interações da validação na entrada do teste (tst_w_val dos autores)")
    parser.add_argument("--reweight", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output", type=Path, help="novo diretório para métricas e checkpoint")
    args = parser.parse_args()
    config = MODELS[args.model].get(args.dataset)
    if config is None:
        parser.error(f"O preset {args.model}/{args.dataset} não foi publicado pelos autores.")
    overrides = {"batch_size": args.batch_size, "steps": args.steps, "scale": args.noise_scale, "reweight": args.reweight}
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    if min(args.epochs, args.threads, config.batch_size) < 1:
        parser.error("Épocas, threads e tamanho do lote devem ser positivos.")
    if config.steps < 2 or not 0 <= args.sampling_steps <= config.steps:
        parser.error("Use pelo menos 2 passos de difusão e 0 a T passos de ruído na inferência.")
    if not 0 <= config.scale <= 1 or (config.scale == 0 and (args.target == "eps" or args.sampling_steps)):
        parser.error("Use ruído entre 0 e 1. O caso zero exige alvo x0 e sampling-steps=0.")
    if not 1 <= args.categories <= 5 or not 0 <= args.seed < 2**32:
        parser.error("Use de 1 a 5 categorias e uma semente entre 0 e 2**32-1.")
    if not 0 <= args.random_noise <= 0.5:
        parser.error("Use uma proporção de ruído aleatório entre 0 e 0,5.")
    if args.random_noise and (args.model != "diffrec" or not args.dataset.endswith("_clean")):
        parser.error("O ruído aleatório usa DiffRec com uma partição clean.")
    try:
        run(args, config)
    except torch.cuda.OutOfMemoryError as exc:
        parser.exit(1, f"Memória de GPU insuficiente. Use --device cpu ou reduza --batch-size.\n{exc}\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Falha ao executar DiffRec: {exc}\n")


if __name__ == "__main__":
    main()
