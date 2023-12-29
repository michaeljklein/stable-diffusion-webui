import torch
import tqdm
import k_diffusion.sampling


@torch.no_grad()
def restart_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., restart_list=None):
    """Implements restart sampling in Restart Sampling for Improving Generative Processes (2023)
    Restart_list format: {min_sigma: [ restart_steps, restart_times, max_sigma]}
    If restart_list is None: will choose restart_list automatically, otherwise will use the given restart_list
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    step_id = 0
    from k_diffusion.sampling import to_d, get_sigmas_karras

    def heun_step(x, old_sigma, new_sigma, second_order=True):
        nonlocal step_id
        denoised = model(x, old_sigma * s_in, **extra_args)
        d = to_d(x, old_sigma, denoised)
        if callback is not None:
            callback({'x': x, 'i': step_id, 'sigma': new_sigma, 'sigma_hat': old_sigma, 'denoised': denoised})
        dt = new_sigma - old_sigma
        if new_sigma == 0 or not second_order:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, new_sigma * s_in, **extra_args)
            d_2 = to_d(x_2, new_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        step_id += 1
        return x

    steps = sigmas.shape[0] - 1
    if restart_list is None:
        if steps >= 20:
            restart_steps = 9
            restart_times = 1
            if steps >= 36:
                restart_steps = steps // 4
                restart_times = 2
            sigmas = get_sigmas_karras(steps - restart_steps * restart_times, sigmas[-2].item(), sigmas[0].item(), device=sigmas.device)
            restart_list = {0.1: [restart_steps + 1, restart_times, 2]}
        else:
            restart_list = {}

    restart_list = {int(torch.argmin(abs(sigmas - key), dim=0)): value for key, value in restart_list.items()}

    step_list = []
    for i in range(len(sigmas) - 1):
        step_list.append((sigmas[i], sigmas[i + 1]))
        if i + 1 in restart_list:
            restart_steps, restart_times, restart_max = restart_list[i + 1]
            min_idx = i + 1
            max_idx = int(torch.argmin(abs(sigmas - restart_max), dim=0))
            if max_idx < min_idx:
                sigma_restart = get_sigmas_karras(restart_steps, sigmas[min_idx].item(), sigmas[max_idx].item(), device=sigmas.device)[:-1]
                while restart_times > 0:
                    restart_times -= 1
                    step_list.extend(zip(sigma_restart[:-1], sigma_restart[1:]))

    last_sigma = None
    for old_sigma, new_sigma in tqdm.tqdm(step_list, disable=disable):
        if last_sigma is None:
            last_sigma = old_sigma
        elif last_sigma < old_sigma:
            x = x + k_diffusion.sampling.torch.randn_like(x) * s_noise * (old_sigma ** 2 - last_sigma ** 2) ** 0.5
        x = heun_step(x, old_sigma, new_sigma)
        last_sigma = new_sigma

    return x


@torch.no_grad()
def sample_dpmpp_2m_hyperbatch(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M) - Hyperbatch.

    Copyright (c) 2022 Katherine Crowson

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

    https://github.com/crowsonkb/k-diffusion
    """
    extra_args = {} if extra_args is None else extra_args

    # BEGIN PATCH
    batch_size = x.size(dim=0)
    x = x[0:1]
    current_batch_size = x.size(dim=0)
    # TODO: remove assert
    assert current_batch_size == 1
    num_steps = len(sigmas)
    current_batch_size_log2 = 0
    batch_size_log2 = batch_size.bit_length()
    batch_size_leftover = 0
    if 2 ** batch_size_log2 != batch_size:
        batch_size_log2 -= 1
        batch_size_leftover = batch_size - 2 ** batch_size_log2

    # number of steps between doubling
    num_sub_steps = batch_size_log2 + 1
    sub_step_size = num_steps // num_sub_steps
    if num_steps % num_sub_steps:
        sub_step_final = sub_step_size * num_sub_steps
    else:
        sub_step_final = sub_step_size * batch_size_log2

    # MulticondLearnedConditioning
    if 'cond' in extra_args:
        extra_args['cond'].shape = (1,)
        # list
        extra_args['cond'].batch = extra_args['cond'].batch[0:1]

    # tensor
    if 'image_cond' in extra_args:
        extra_args['image_cond'] = extra_args['image_cond'][0:1]

    # list
    if 'uncond' in extra_args:
        extra_args['uncond'] = extra_args['uncond'][0:1]
    # END PATCH

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in tqdm.auto.trange(len(sigmas) - 1, disable=disable):
        if i != 0 and i % sub_step_size == 0:
            if i == sub_step_final:
                appended_size = batch_size_leftover
            else:
                appended_size = None

            current_batch_size_log2 += 1
            x = torch.cat([x, x[0:appended_size]], dim=0)
            s_in = x.new_ones([x.shape[0]])
            old_denoised = torch.cat([old_denoised, old_denoised[0:appended_size]], dim=0)
            current_batch_size = x.size(dim=0)

            # MulticondLearnedConditioning
            if 'cond' in extra_args:
                extra_args['cond'].shape = (current_batch_size,)
                extra_args['cond'].batch = extra_args['cond'].batch + extra_args['cond'].batch[0:appended_size]

            if 'image_cond' in extra_args:
                extra_args['image_cond'] = torch.cat([extra_args['image_cond'], extra_args['image_cond'][0:appended_size]], dim=0)

            # list
            if 'uncond' in extra_args:
                extra_args['uncond'] = extra_args['uncond'] + extra_args['uncond'][0:appended_size]

        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x



@torch.no_grad()
def sample_dpmpp_sde_hyperbatch(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    """DPM-Solver++ (stochastic) - Hyperbatch.

    Copyright (c) 2022 Katherine Crowson

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

    https://github.com/crowsonkb/k-diffusion
    """
    from k_diffusion.sampling import to_d, get_ancestral_step, BrownianTreeNoiseSampler

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args

    # BEGIN PATCH
    batch_size = x.size(dim=0)
    x = x[0:1]
    current_batch_size = x.size(dim=0)
    # TODO: remove assert
    assert current_batch_size == 1
    num_steps = len(sigmas)
    current_batch_size_log2 = 0
    batch_size_log2 = batch_size.bit_length()
    batch_size_leftover = 0
    if 2 ** batch_size_log2 != batch_size:
        batch_size_log2 -= 1
        batch_size_leftover = batch_size - 2 ** batch_size_log2

    # number of steps between doubling
    num_sub_steps = batch_size_log2 + 1
    sub_step_size = num_steps // num_sub_steps
    if num_steps % num_sub_steps:
        sub_step_final = sub_step_size * num_sub_steps
    else:
        sub_step_final = sub_step_size * batch_size_log2

    # MulticondLearnedConditioning
    if 'cond' in extra_args:
        extra_args['cond'].shape = (1,)
        # list
        extra_args['cond'].batch = extra_args['cond'].batch[0:1]

    # tensor
    if 'image_cond' in extra_args:
        extra_args['image_cond'] = extra_args['image_cond'][0:1]

    # list
    if 'uncond' in extra_args:
        extra_args['uncond'] = extra_args['uncond'][0:1]
    # END PATCH

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in tqdm.trange(len(sigmas) - 1, disable=disable):
        if i != 0 and i % sub_step_size == 0:
            if i == sub_step_final:
                appended_size = batch_size_leftover
            else:
                appended_size = None

            current_batch_size_log2 += 1
            x = torch.cat([x, x[0:appended_size]], dim=0)
            s_in = x.new_ones([x.shape[0]])
            current_batch_size = x.size(dim=0)

            # MulticondLearnedConditioning
            if 'cond' in extra_args:
                extra_args['cond'].shape = (current_batch_size,)
                extra_args['cond'].batch = extra_args['cond'].batch + extra_args['cond'].batch[0:appended_size]

            if 'image_cond' in extra_args:
                extra_args['image_cond'] = torch.cat([extra_args['image_cond'], extra_args['image_cond'][0:appended_size]], dim=0)

            # list
            if 'uncond' in extra_args:
                extra_args['uncond'] = extra_args['uncond'] + extra_args['uncond'][0:appended_size]

        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            noise_sample = noise_sampler(sigma_fn(t), sigma_fn(s))[0:current_batch_size]
            x_2 = x_2 + noise_sample * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            noise_sample = noise_sampler(sigma_fn(t), sigma_fn(t_next))[0:current_batch_size]
            x = x + noise_sample * s_noise * su
    return x
