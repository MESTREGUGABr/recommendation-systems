"""Training loss using the reference diffusion schedule and timestep sampler."""

import torch


@torch.no_grad()
def update_history(diffusion, timesteps, losses):
    # Keep the most recent observations for each timestep in minibatch order.
    # Vectorizing the original FIFO avoids GPU synchronization per user.
    history = diffusion.Lt_history
    width = history.shape[1]
    counts = torch.zeros_like(diffusion.Lt_count)
    counts.scatter_add_(0, timesteps, torch.ones_like(timesteps))
    ends = counts.cumsum(0)
    offsets = torch.arange(width, device=timesteps.device)
    new_indices = ends[:, None] - width + offsets
    has_new_value = new_indices >= (ends - counts)[:, None]
    ordered = losses.detach()[timesteps.argsort(stable=True)]
    retained = history.gather(1, (counts[:, None] + offsets).clamp(max=width - 1))
    history.copy_(torch.where(has_new_value, ordered[new_indices.clamp(min=0)], retained))
    diffusion.Lt_count.add_(counts).clamp_(max=width)


def training_loss(diffusion, model, clean, reweight, predict_noise=False):
    steps, probabilities = diffusion.sample_timesteps(len(clean), clean.device, "importance")
    noise = torch.randn_like(clean)
    corrupted = diffusion.q_sample(clean, steps, noise) if diffusion.noise_scale else clean
    output = model(corrupted, steps)
    if predict_noise:
        reconstructed = diffusion._predict_xstart_from_eps(corrupted, steps, output)
        losses = (noise - output).square().mean(dim=1)
        if reweight:
            weight = (1 - diffusion.alphas_cumprod[steps]) / (
                (1 - diffusion.alphas_cumprod_prev[steps]).square() * (1 - diffusion.betas[steps])
            )
            reconstruction = (clean - reconstructed).square().mean(dim=1) / 2
            losses = torch.where(steps == 0, reconstruction, losses * weight)
    else:
        reconstructed = output
        losses = (clean - output).square().mean(dim=1)
        if reweight and diffusion.noise_scale:
            weight = diffusion.SNR(steps - 1) - diffusion.SNR(steps)
            losses = losses * torch.where(steps == 0, 1.0, weight)
    update_history(diffusion, steps, losses)
    return (losses / probabilities).mean(), reconstructed
