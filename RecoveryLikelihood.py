import argparse

import torch
import torch.nn as nn
import numpy as np


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    betas = np.append(betas, 1.)
    assert betas.shape == (num_diffusion_timesteps + 1,)
    return betas


def get_sigma_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    """
    Get the noise level schedule
    :param beta_start: begin noise level
    :param beta_end: end noise level
    :param num_diffusion_timesteps: number of timesteps
    :return:
    -- sigmas: sigma_{t+1}, scaling parameter of epsilon_{t+1}
    -- a_s: sqrt(1 - sigma_{t+1}^2), scaling parameter of x_t
    """
    betas = np.linspace(beta_start, beta_end, 1000, dtype=np.float64)
    betas = np.append(betas, 1.)
    assert isinstance(betas, np.ndarray)
    betas = betas.astype(np.float64)
    assert (betas > 0).all() and (betas <= 1).all()
    sqrt_alphas = np.sqrt(1. - betas)
    temp = np.concatenate([np.arange(num_diffusion_timesteps) * (1000 // ((num_diffusion_timesteps - 1) * 2)), [999]])
    idx = temp.astype(np.int32)
    a_s = np.concatenate(
        [[np.prod(sqrt_alphas[: idx[0] + 1])],
         np.asarray([np.prod(sqrt_alphas[idx[i - 1] + 1: idx[i] + 1]) for i in np.arange(1, len(idx))])])
    sigma = np.sqrt(1 - a_s ** 2)

    return sigma, a_s


def unsorted_segment_mean(values, index, num_segments):
    ones = torch.ones_like(values)
    sums = torch.zeros(num_segments, device=values.device).scatter_add_(0, index, values)
    counts = torch.zeros(num_segments, device=values.device).scatter_add_(0, index, ones)
    return sums / counts


class RecoveryLikelihood(nn.Module):
    def __init__(self, model, args):
        super(RecoveryLikelihood, self).__init__()
        self.args = args
        self.num_timesteps = args.num_diffusion_timesteps

        sigmas, a_s = get_sigma_schedule(beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=self.num_timesteps)
        self.sigmas = torch.FloatTensor(sigmas).to(args.device)
        self.a_s = torch.FloatTensor(a_s).to(args.device)

        self.a_s_cum = torch.FloatTensor(np.cumprod(a_s)).to(args.device)
        self.sigmas_cum = torch.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        self.is_recovery = torch.ones(self.num_timesteps + 1).to(args.device)
        self.is_recovery[-1] = 0
        self.device = args.device

        # self.net = net_res_temb2(name='net', ch=128, ch_mult=ch_mult, num_res_blocks=self.args.num_res_blocks, attn_resolutions=(16,))
        # self.net = Wide_ResNet(28, 10, norm='batch', dropout_rate=0).to(args.device)
        self.net = model

    @staticmethod
    def _extract(a, t, x_shape, device):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        if isinstance(t, int) or len(t.shape) == 0:
            t = torch.ones(x_shape[0], dtype=torch.int64, device=device) * t
        bs, = t.shape
        assert x_shape[0] == bs
        out = a[t]
        # out = tf.gather(tf.convert_to_tensor(a, dtype=tf.float32), t)
        # print(out.shape, t.shape, bs)
        assert list(out.shape) == [bs]
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        x_t = self._extract(self.a_s_cum, t, x_start.shape, self.args.device) * x_start + \
              self._extract(self.sigmas_cum, t, x_start.shape, self.args.device) * noise

        return x_t

    def q_sample_pairs(self, x_start, t):
        """
        Generate a pair of disturbed images for training
        :param x_start: x_0
        :param t: time step t
        :return: x_t, x_{t+1}
        """
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t)
        x_t_plus_one = self._extract(self.a_s, t + 1, x_start.shape, self.args.device) * x_t + \
                       self._extract(self.sigmas, t + 1, x_start.shape, self.args.device) * noise

        return x_t, x_t_plus_one

    def q_sample_progressive(self, x_0):
        """
        Generate a full sequence of disturbed images
        """
        x_preds = []
        for t in range(self.num_timesteps + 1):
            t_now = torch.ones([x_0.shape[0]], dtype=torch.int32, device=self.args.device) * t
            x = self.q_sample(x_0, t_now)
            x_preds.append(x)
        x_preds = torch.stack(x_preds, axis=0)

        return x_preds

    # === Training loss ===
    def training_losses(self, x_pos, x_neg, t):
        """
        Training loss calculation
        """
        a_s = self._extract(self.a_s_prev, t + 1, x_pos.shape, self.args.device)
        y_pos = a_s * x_pos
        y_neg = a_s * x_neg
        pos_f = self.net(y_pos, t).sum(dim=1)
        neg_f = self.net(y_neg, t).sum(dim=1)
        loss = - (pos_f - neg_f)

        loss_scale = 1.0 / (self.sigmas[t + 1] / self.sigmas[1])
        loss = loss_scale * loss

        # loss_ts = torch.math.unsorted_segment_mean(torch.abs(loss), t, self.num_timesteps)
        loss_ts = unsorted_segment_mean(torch.abs(loss), t, self.num_timesteps).detach()
        f_ts = unsorted_segment_mean(pos_f, t, self.num_timesteps).detach()

        return loss.mean(), loss_ts, f_ts

    def log_prob(self, y, t, tilde_x, b0, sigma, is_recovery):
        logits = self.net(y, t)

        return logits.sum(dim=1) / torch.reshape(b0, [-1]) - torch.sum((y - tilde_x) ** 2 / 2 / sigma ** 2 * is_recovery, dim=[1, 2, 3])

    def grad_f(self, y, t, tilde_x, b0, sigma, is_recovery):
        log_p_y = self.log_prob(y, t, tilde_x, b0, sigma, is_recovery)
        grad_y = torch.autograd.grad(log_p_y.sum(), [y], retain_graph=True)[0]
        # grad_y = torch.clamp(grad_y, -1, 1)
        return grad_y, log_p_y

    # === Sampling ===
    def p_sample_langevin(self, tilde_x, t):
        """
        Langevin sampling function
        """
        sigma = self._extract(self.sigmas, t + 1, tilde_x.shape, self.args.device)
        sigma_cum = self._extract(self.sigmas_cum, t, tilde_x.shape, self.args.device)
        is_recovery = self._extract(self.is_recovery, t + 1, tilde_x.shape, self.args.device)
        a_s = self._extract(self.a_s_prev, t + 1, tilde_x.shape, self.args.device)

        c_t_square = sigma_cum / self.sigmas_cum[0]
        step_size_square = c_t_square * self.args.mcmc_step_size_b_square * sigma ** 2

        # y = torch.identity(tilde_x)
        y = torch.autograd.Variable(tilde_x, requires_grad=True).to(self.args.device)
        is_accepted_summary = torch.zeros(y.shape[0], dtype=torch.float32, device=self.args.device)

        grad_y, log_p_y = self.grad_f(y, t, tilde_x, step_size_square, sigma, is_recovery)

        for _ in range(self.args.mcmc_num_steps):
            noise = torch.randn_like(y)
            y_new = y + 0.5 * step_size_square * grad_y + torch.sqrt(step_size_square) * noise * self.args.noise_scale

            grad_y_new, log_p_y_new = self.grad_f(y_new, t, tilde_x, step_size_square, sigma, is_recovery)
            y, grad_y, log_p_y = y_new, grad_y_new, log_p_y_new

        is_accepted_summary = 1.0 * is_accepted_summary / self.args.mcmc_num_steps
        is_accepted_summary = torch.mean(is_accepted_summary)

        x = y / a_s

        values = torch.norm(torch.reshape(x, [x.shape[0], -1]) - torch.reshape(tilde_x, [tilde_x.shape[0], -1]), dim=1)
        disp = unsorted_segment_mean(values, t, self.num_timesteps)
        return x, disp, is_accepted_summary

    def p_sample_progressive(self, noise):
        """
        Sample a sequence of images with the sequence of noise levels
        """
        num = noise.shape[0]
        x_neg_t = noise
        x_neg = torch.zeros([self.args.num_diffusion_timesteps, num, 3, self.args.img_sz, self.args.img_sz], device=self.device)
        x_neg = torch.cat([x_neg, torch.unsqueeze(noise, axis=0)], dim=0)
        is_accepted_summary = 0.

        for t in range(self.args.num_diffusion_timesteps - 1, -1, -1):
            t_v = torch.tensor(t).to(self.device)

            x_neg_t, _, is_accepted = self.p_sample_langevin(x_neg_t, t_v)
            is_accepted_summary = is_accepted_summary + is_accepted
            x_neg_t = torch.reshape(x_neg_t, [num, 3, self.args.img_sz, self.args.img_sz])
            insert_mask = t == torch.arange(self.args.num_diffusion_timesteps + 1, device=self.device)
            insert_mask = torch.reshape(insert_mask, [-1, *([1] * len(noise.shape))])
            x_neg = insert_mask * torch.unsqueeze(x_neg_t, axis=0) + (~ insert_mask) * x_neg
        is_accepted_summary = is_accepted_summary / self.args.num_diffusion_timesteps * 1.0
        return x_neg, is_accepted_summary


if __name__ == '__main__':
    # sigmas, a_s = get_sigma_schedule(beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=6)
    # print(sigmas.shape)
    # print(a_s.shape)

    args = argparse.Namespace()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = t.device('cpu')
    args.device = device

    args.img_sz = 32
    args.num_diffusion_timesteps = 6
    args.opt = 'sgd'
    args.ma_decay = 0.999
    args.noise_scale = 1.0
    args.mcmc_num_steps = 30
    args.mcmc_step_size_b_square = 2e-4

    x_p_d = torch.randn([64, 3, 32, 32])
    x_p_d = x_p_d.to(device)

    from models.wideresnet_te import Wide_ResNet as wrn
    model = wrn(28, 10, norm='batch').to(args.device)
    diffusion = RecoveryLikelihood(model, args)
    t = torch.randint(size=[64], high=6, device=device)
    k = diffusion.q_sample_pairs(x_p_d, t)
    print(k[0].shape, k[1].shape)
    # m = diffusion.p_sample_langevin(x_p_d, t)
    t = torch.randint(size=[64], high=6, device=device)
    x_pos, x_neg = diffusion.q_sample_pairs(x_p_d, t)
    x_neg, disp, is_accepted = diffusion.p_sample_langevin(x_neg, t)
    loss, _, _ = diffusion.training_losses(x_pos, x_neg, t)
    print(loss)

    noise = torch.randn_like(x_p_d)
    x_neg_seq = diffusion.p_sample_progressive(noise)[0][:, :64]
    print(x_neg_seq.shape)
