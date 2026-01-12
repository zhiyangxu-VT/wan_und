import torch


class FlowMatchScheduler:
    def __init__(self):
        self.num_train_timesteps = 1000
        self.sigmas = None
        self.timesteps = None
        self.training = False
        self.linear_timesteps_weights = None

    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=5):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_training_weight(self):
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        weights = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            weights = weights * (len(self.timesteps) / steps)
            weights = weights + weights[1]
        self.linear_timesteps_weights = weights

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        self.sigmas, self.timesteps = self.set_timesteps_wan(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        return self.linear_timesteps_weights[timestep_id]
