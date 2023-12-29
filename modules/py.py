
def current_batch_size_list(batch_size, num_steps):
    current_batch_size = 1
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

    current_batch_sizes = []
    for i in range(num_steps - 1):
        if i != 0 and i % sub_step_size == 0:
            if i == sub_step_final:
                appended_size = batch_size_leftover
            else:
                appended_size = None

            current_batch_size_log2 += 1
            if appended_size is None:
                current_batch_size = 2 * current_batch_size
            else:
                current_batch_size = current_batch_size + appended_size

        current_batch_sizes.append(current_batch_size)

    return current_batch_sizes


# print(get_hyberbatch_traces(7, 20))
# [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 0, 0], [3, 3, 1, 0], [4, 0, 0, 0], [5, 1, 1, 0], [6, 2, 0, 0]]
def get_hyberbatch_traces(batch_size, num_steps):
    trace_list = [[0]]
    current_batch_size = 1
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

    for i in range(num_steps - 1):
        if i != 0 and i % sub_step_size == 0:
            if i == sub_step_final:
                appended_size = batch_size_leftover
            else:
                appended_size = None

            current_batch_size_log2 += 1
            trace_list = list(map(lambda i_trace: [i_trace[0]] + i_trace[1], enumerate(trace_list + trace_list[0:appended_size])))
            if appended_size is None:
                current_batch_size = 2 * current_batch_size
            else:
                current_batch_size = current_batch_size + appended_size

    return trace_list

import torch
import tqdm
import k_diffusion.sampling


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
    from k_diffusion.sampling import to_d, get_ancestral_step

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




############

import torch
import inspect
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_extra, sd_samplers_cfg_denoiser
from modules.sd_samplers_cfg_denoiser import CFGDenoiser  # noqa: F401
from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback

from modules.shared import opts
import modules.shared as shared

samplers_k_diffusion = [
    ('DPM++ 2M Karras', 'sample_dpmpp_2m', ['k_dpmpp_2m_ka'], {'scheduler': 'karras'}),
    ('DPM++ SDE Karras', 'sample_dpmpp_sde', ['k_dpmpp_sde_ka'], {'scheduler': 'karras', "second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE Exponential', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_exp'], {'scheduler': 'exponential', "brownian_noise": True}),
    ('DPM++ 2M SDE Karras', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_ka'], {'scheduler': 'karras', "brownian_noise": True}),
    ('Euler a', 'sample_euler_ancestral', ['k_euler_a', 'k_euler_ancestral'], {"uses_ensd": True}),
    ('Euler', 'sample_euler', ['k_euler'], {}),
    ('LMS', 'sample_lms', ['k_lms'], {}),
    ('Heun', 'sample_heun', ['k_heun'], {"second_order": True}),
    ('DPM2', 'sample_dpm_2', ['k_dpm_2'], {'discard_next_to_last_sigma': True, "second_order": True}),
    ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM++ 2S a', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a'], {"uses_ensd": True, "second_order": True}),
    ('DPM++ 2M', 'sample_dpmpp_2m', ['k_dpmpp_2m'], {}),
    ('DPM++ SDE', 'sample_dpmpp_sde', ['k_dpmpp_sde'], {"second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_ka'], {"brownian_noise": True}),
    ('DPM++ 2M SDE Heun', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun'], {"brownian_noise": True, "solver_type": "heun"}),
    ('DPM++ 2M SDE Heun Karras', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun_ka'], {'scheduler': 'karras', "brownian_noise": True, "solver_type": "heun"}),
    ('DPM++ 2M SDE Heun Exponential', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun_exp'], {'scheduler': 'exponential', "brownian_noise": True, "solver_type": "heun"}),
    ('DPM++ 3M SDE', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde'], {'discard_next_to_last_sigma': True, "brownian_noise": True}),
    ('DPM++ 3M SDE Karras', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde_ka'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
    ('DPM++ 3M SDE Exponential', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde_exp'], {'scheduler': 'exponential', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
    ('DPM fast', 'sample_dpm_fast', ['k_dpm_fast'], {"uses_ensd": True}),
    ('DPM adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {"uses_ensd": True}),
    ('LMS Karras', 'sample_lms', ['k_lms_ka'], {'scheduler': 'karras'}),
    ('DPM2 Karras', 'sample_dpm_2', ['k_dpm_2_ka'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM2 a Karras', 'sample_dpm_2_ancestral', ['k_dpm_2_a_ka'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM++ 2S a Karras', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a_ka'], {'scheduler': 'karras', "uses_ensd": True, "second_order": True}),
    ('Restart', sd_samplers_extra.restart_sampler, ['restart'], {'scheduler': 'karras', "second_order": True}),
    ('DPM++ 2M Karras - Hyperbatch', sd_samplers_extra.sample_dpmpp_2m_hyperbatch, ['k_dpmpp_2m_ka_hyperbatch'], {'scheduler': 'karras'}),
    ('DPM++ SDE Karras - Hyperbatch', sd_samplers_extra.sample_dpmpp_sde_hyperbatch, ['k_dpmpp_sde_ka_hyperbatch'], {'scheduler': 'karras', "second_order": True, "brownian_noise": True}),
    ('DPM++ SDE - Hyperbatch', sd_samplers_extra.sample_dpmpp_sde_hyperbatch, ['k_dpmpp_sde_hyperbatch'], {"second_order": True, "brownian_noise": True}),
]


samplers_data_k_diffusion = [
    sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
    for label, funcname, aliases, options in samplers_k_diffusion
    if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
]

sampler_extra_params = {
    'sample_euler': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_heun': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_dpm_2': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_dpm_fast': ['s_noise'],
    'sample_dpm_2_ancestral': ['s_noise'],
    'sample_dpmpp_2s_ancestral': ['s_noise'],
    'sample_dpmpp_sde': ['s_noise'],
    'sample_dpmpp_2m_sde': ['s_noise'],
    'sample_dpmpp_3m_sde': ['s_noise'],
    sd_samplers_extra.sample_dpmpp_sde_hyperbatch: ['s_noise'],
}

k_diffusion_samplers_map = {x.name: x for x in samplers_data_k_diffusion}
k_diffusion_scheduler = {
    'Automatic': None,
    'karras': k_diffusion.sampling.get_sigmas_karras,
    'exponential': k_diffusion.sampling.get_sigmas_exponential,
    'polyexponential': k_diffusion.sampling.get_sigmas_polyexponential
}


class CFGDenoiserKDiffusion(sd_samplers_cfg_denoiser.CFGDenoiser):
    @property
    def inner_model(self):
        if self.model_wrap is None:
            denoiser = k_diffusion.external.CompVisVDenoiser if shared.sd_model.parameterization == "v" else k_diffusion.external.CompVisDenoiser
            self.model_wrap = denoiser(shared.sd_model, quantize=shared.opts.enable_quantization)

        return self.model_wrap


class KDiffusionSampler(sd_samplers_common.Sampler):
    def __init__(self, funcname, sd_model, options=None):
        super().__init__(funcname)

        self.extra_params = sampler_extra_params.get(funcname, [])

        self.options = options or {}
        self.func = funcname if callable(funcname) else getattr(k_diffusion.sampling, self.funcname)

        self.model_wrap_cfg = CFGDenoiserKDiffusion(self)
        self.model_wrap = self.model_wrap_cfg.inner_model

    def get_sigmas(self, p, steps):
        discard_next_to_last_sigma = self.config is not None and self.config.options.get('discard_next_to_last_sigma', False)
        if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
            discard_next_to_last_sigma = True
            p.extra_generation_params["Discard penultimate sigma"] = True

        steps += 1 if discard_next_to_last_sigma else 0

        if p.sampler_noise_scheduler_override:
            sigmas = p.sampler_noise_scheduler_override(steps)
        elif opts.k_sched_type != "Automatic":
            m_sigma_min, m_sigma_max = (self.model_wrap.sigmas[0].item(), self.model_wrap.sigmas[-1].item())
            sigma_min, sigma_max = (0.1, 10) if opts.use_old_karras_scheduler_sigmas else (m_sigma_min, m_sigma_max)
            sigmas_kwargs = {
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
            }

            sigmas_func = k_diffusion_scheduler[opts.k_sched_type]
            p.extra_generation_params["Schedule type"] = opts.k_sched_type

            if opts.sigma_min != m_sigma_min and opts.sigma_min != 0:
                sigmas_kwargs['sigma_min'] = opts.sigma_min
                p.extra_generation_params["Schedule min sigma"] = opts.sigma_min
            if opts.sigma_max != m_sigma_max and opts.sigma_max != 0:
                sigmas_kwargs['sigma_max'] = opts.sigma_max
                p.extra_generation_params["Schedule max sigma"] = opts.sigma_max

            default_rho = 1. if opts.k_sched_type == "polyexponential" else 7.

            if opts.k_sched_type != 'exponential' and opts.rho != 0 and opts.rho != default_rho:
                sigmas_kwargs['rho'] = opts.rho
                p.extra_generation_params["Schedule rho"] = opts.rho

            sigmas = sigmas_func(n=steps, **sigmas_kwargs, device=shared.device)
        elif self.config is not None and self.config.options.get('scheduler', None) == 'karras':
            sigma_min, sigma_max = (0.1, 10) if opts.use_old_karras_scheduler_sigmas else (self.model_wrap.sigmas[0].item(), self.model_wrap.sigmas[-1].item())

            sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=shared.device)
        elif self.config is not None and self.config.options.get('scheduler', None) == 'exponential':
            m_sigma_min, m_sigma_max = (self.model_wrap.sigmas[0].item(), self.model_wrap.sigmas[-1].item())
            sigmas = k_diffusion.sampling.get_sigmas_exponential(n=steps, sigma_min=m_sigma_min, sigma_max=m_sigma_max, device=shared.device)
        else:
            sigmas = self.model_wrap.get_sigmas(steps)

        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)

        sigmas = self.get_sigmas(p, steps)
        sigma_sched = sigmas[steps - t_enc - 1:]

        xi = x + noise * sigma_sched[0]

        if opts.img2img_extra_noise > 0:
            p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
            extra_noise_params = ExtraNoiseParams(noise, x, xi)
            extra_noise_callback(extra_noise_params)
            noise = extra_noise_params.noise
            xi += noise * opts.img2img_extra_noise

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        if 'sigma_min' in parameters:
            ## last sigma is zero which isn't allowed by DPM Fast & Adaptive so taking value before last
            extra_params_kwargs['sigma_min'] = sigma_sched[-2]
        if 'sigma_max' in parameters:
            extra_params_kwargs['sigma_max'] = sigma_sched[0]
        if 'n' in parameters:
            extra_params_kwargs['n'] = len(sigma_sched) - 1
        if 'sigma_sched' in parameters:
            extra_params_kwargs['sigma_sched'] = sigma_sched
        if 'sigmas' in parameters:
            extra_params_kwargs['sigmas'] = sigma_sched

        if self.config.options.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs['noise_sampler'] = noise_sampler

        if self.config.options.get('solver_type', None) == 'heun':
            extra_params_kwargs['solver_type'] = 'heun'

        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }

        samples = self.launch_sampling(t_enc + 1, lambda: self.func(self.model_wrap_cfg, xi, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))

        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        return samples

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        steps = steps or p.steps

        sigmas = self.get_sigmas(p, steps)

        if opts.sgm_noise_multiplier:
            p.extra_generation_params["SGM noise multiplier"] = True
            x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        else:
            x = x * sigmas[0]

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        if 'n' in parameters:
            extra_params_kwargs['n'] = steps

        if 'sigma_min' in parameters:
            extra_params_kwargs['sigma_min'] = self.model_wrap.sigmas[0].item()
            extra_params_kwargs['sigma_max'] = self.model_wrap.sigmas[-1].item()

        if 'sigmas' in parameters:
            extra_params_kwargs['sigmas'] = sigmas

        if self.config.options.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs['noise_sampler'] = noise_sampler

        if self.config.options.get('solver_type', None) == 'heun':
            extra_params_kwargs['solver_type'] = 'heun'

        self.last_latent = x
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }

        samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, x, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))

        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        return samples



