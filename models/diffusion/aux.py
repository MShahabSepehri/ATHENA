import torch
import numpy as np
from functools import partial
from abc import ABC, abstractmethod


NUMBER_DICT = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
               6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
               11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 
               15: 'fifteen', 16: 'sixteen'}


class StopRun(Exception):
    pass

class Backend(ABC):
    def __init__(self):
        self.PROMPT_KEYS = None

    @staticmethod
    @abstractmethod
    def get_sigmas(pipe, total_steps):
        pass

    @abstractmethod
    def continue_generation(pipe, latents, current_step, stop_step, sigmas, **kwargs):
        pass

    @abstractmethod
    def go_forward(pipe, stop_step, **kwargs):
        pass

    @staticmethod
    def get_model(pipe):
        return pipe.transformer

    @staticmethod
    def get_generator(seed):
        return torch.Generator("cpu").manual_seed(seed)

    @staticmethod
    def get_step(pipe):
        step = pipe.scheduler.step_index
        if step is None:
            step = 0
        step += 1
        return step
    
    @staticmethod
    def normalize(latents, new_latents):
        original_norm = torch.norm(latents.flatten(1), p=2, dim=1, keepdim=True)
        new_norm = torch.norm(new_latents.flatten(1), p=2, dim=1, keepdim=True)
        renorm_factor = original_norm / (new_norm + 1e-6)
        renorm_factor = renorm_factor.view(latents.size(0), 1, 1, 1)
        new_latents = new_latents * renorm_factor
        return new_latents
    

    def estimage_check(self, pipe, count_func, estimate_step, total_steps, target, seed, **kwargs):
        kwargs['generator'] = self.get_generator(seed)
        sigmas = self.get_sigmas(pipe, total_steps)

        latents, timesteps, sigmas_s = self.go_forward(pipe, 0, **kwargs)
        image, updated_latents = self.continue_generation(pipe, latents.clone(), 0, estimate_step, sigmas, **kwargs)
        count = count_func(image)

        if count == target:
            image, _ = self.continue_generation(pipe, updated_latents.clone(), estimate_step - 1, total_steps, sigmas, **kwargs)
            return None, image
        
        return count, None


    def format_forward_output(self, noise_pred):
        return (noise_pred)

    def athena_forward(self, *args, pipe, original_forward, steering_step,
                      null_prompt_args, factor, merge_func,
                      beta, **kwargs):
            step = self.get_step(pipe)
            noise_pred = original_forward(*args, **kwargs)[0]
            if step <= steering_step:
                orig_dict = {k: kwargs[k] for k in self.PROMPT_KEYS}
                for k in self.PROMPT_KEYS:
                    kwargs[k] = null_prompt_args[k]
                p2 = original_forward(*args, **kwargs)[0]

                tmp = noise_pred + factor * merge_func(noise_pred, p2) * (beta ** (step - 1))
                noise_pred = self.normalize(noise_pred, tmp)
                for k in self.PROMPT_KEYS:
                    kwargs[k] = orig_dict[k]

            return self.format_forward_output(noise_pred)


    def athena(self, pipe, null_prompt_args, steering_step, factor, beta, seed, merge_func=None, **kwargs):
        if merge_func is None:
            merge_func = lambda x, y: x - y
        original_forward = self.get_model(pipe).forward

        wrapped_forward = partial(self.athena_forward, 
                                pipe=pipe, 
                                original_forward=original_forward, 
                                steering_step=steering_step,
                                null_prompt_args=null_prompt_args, 
                                factor=factor, 
                                merge_func=merge_func,
                                beta=beta)

        self.get_model(pipe).forward = wrapped_forward
        kwargs['generator'] = self.get_generator(seed)
        image = pipe(**kwargs).images[0]
        self.get_model(pipe).forward = original_forward
        return image


    def capture_args(self, pipe, null_prompt, **kwargs):
        def ca(*args, **kwargs):
            raise StopRun(kwargs)
        original_forward = self.get_model(pipe).forward
        
        try:
            prompt = kwargs.get('prompt', '')
            kwargs['prompt'] = null_prompt
            self.get_model(pipe).forward = ca
            _ = pipe(**kwargs).images[0]
        except StopRun as e:
            kwargs['prompt'] = prompt
            self.get_model(pipe).forward = original_forward
            tmp = e.args[0]
            if self.PROMPT_KEYS is None:
                return tmp
            return {k: tmp.get(k) for k in self.PROMPT_KEYS}


    def athena_static(self, pipe, factor, steering_step, seed, target, beta=1, replacement='', progress_bar=False, **kwargs):
        pipe.set_progress_bar_config(disable=(not progress_bar))
        
        prompt = kwargs.get('prompt', '')
        null_prompt = prompt.replace(NUMBER_DICT.get(target), replacement)

        args_dict = self.capture_args(pipe, null_prompt, **kwargs)
        image = self.athena(pipe, args_dict, steering_step, factor, beta, seed, **kwargs)
        return image


    def athena_feedback(self, pipe, count_func, target, total_steps, factor, steering_step, seed, estimate_step=15, beta=1, progress_bar=False, **kwargs):
        pipe.set_progress_bar_config(disable=(not progress_bar))
        count, image = self.estimage_check(pipe, count_func, estimate_step, total_steps, target, seed, **kwargs)
        if image is not None:
            return image
        tmp = NUMBER_DICT.get(min(max(count, 1), 16))
        
        prompt = kwargs.get('prompt', '')
        null_prompt = prompt.replace(NUMBER_DICT.get(target), tmp)

        args_dict = self.capture_args(pipe, null_prompt, **kwargs)
        image = self.athena(pipe, args_dict, steering_step, factor, beta, seed, **kwargs)
        return image


    def athena_adaptive(self, pipe, count_func, target, total_steps, factor, steering_step, seed, estimate_step=15, beta=1, r=2, max_try=2, progress_bar=False, **kwargs):
        pipe.set_progress_bar_config(disable=(not progress_bar))
        count, image = self.estimage_check(pipe, count_func, estimate_step, total_steps, target, seed, **kwargs)
        if image is not None:
            return image
        tmp = NUMBER_DICT.get(min(max(count, 1), 16))
        
        prompt = kwargs.get('prompt', '')
        null_prompt = prompt.replace(NUMBER_DICT.get(target), tmp)

        args_dict = self.capture_args(pipe, null_prompt, **kwargs)
        
        merge_func = lambda x, y: x - y
        original_forward = self.get_model(pipe).forward

        wrapped_forward = partial(self.athena_forward, 
                                  pipe=pipe, 
                                  original_forward=original_forward, 
                                  steering_step=steering_step,
                                  null_prompt_args=args_dict, 
                                  factor=factor, 
                                  merge_func=merge_func,
                                  beta=beta)

        self.get_model(pipe).forward = wrapped_forward
        prev_count = count
        for i in range(max_try - 1):
            kwargs['generator'] = self.get_generator(seed)
            count, image = self.estimage_check(pipe, count_func, estimate_step, total_steps, target, seed, **kwargs)
            if image is not None:
                self.get_model(pipe).forward = original_forward
                return image
            c = (prev_count - target) * (count - target)
            tmp = factor
            if c < 0:
                factor /= r * (0.9 ** i)
            else:
                factor *= r * (0.9 ** i)
            prev_count = count
            wrapped_forward = partial(self.athena_forward, 
                                      pipe=pipe, 
                                      original_forward=original_forward, 
                                      steering_step=steering_step,
                                      null_prompt_args=args_dict, 
                                      factor=factor, 
                                      merge_func=merge_func,
                                      beta=beta)
            self.get_model(pipe).forward = wrapped_forward
        image = pipe(**kwargs).images[0]
        self.get_model(pipe).forward = original_forward
        return image


class FluxBackend(Backend):

    def __init__(self):
        super().__init__()
        self.PROMPT_KEYS = ['pooled_projections', 'encoder_hidden_states', 'txt_ids']

    @staticmethod
    def get_sigmas(pipe, total_steps):
        sigmas = np.linspace(1.0, 1 / total_steps, total_steps)
        return sigmas

    def continue_generation(self, pipe, latents, current_step, 
                            stop_step, sigmas, **kwargs):

        def wrapped_step(*args, **kwargs):
            return original_step(args[0], args[1], latents, *args[3: ], **kwargs)

        def wrapped_forward(*args, **kwargs):
            step = self.get_step(pipe)
            if step == 1:
                if latents is not None:
                    kwargs['hidden_states'] = latents
                    pipe.scheduler.step = wrapped_step
                pipe.scheduler.sigmas[-1] = 0.0
            else:
                pipe.scheduler.step = original_step
            if step == num_steps:
                return_dict['latents'] = kwargs['hidden_states'].clone()
            noise_pred = original_forward(*args, **kwargs)
            return noise_pred
        
        return_dict = {}
        tmp_sigmas = sigmas[current_step: stop_step].copy()
        original_steps = kwargs['num_inference_steps']
        original_sigmas = kwargs.get('sigmas')
        kwargs['sigmas'] = tmp_sigmas
        num_steps = stop_step - current_step
        kwargs['num_inference_steps'] = num_steps

        original_forward = self.get_model(pipe).forward
        original_step = pipe.scheduler.step
        self.get_model(pipe).forward = wrapped_forward

        image = pipe(**kwargs).images[0]
        kwargs['sigmas'] = original_sigmas
        kwargs['num_inference_steps'] = original_steps
        self.get_model(pipe).forward = original_forward
        pipe.scheduler.step = original_step
        return image, return_dict.get('latents', None)


    def go_forward(self, pipe, stop_step, **kwargs):
        def wrapped_forward(*args, **kwargs):
            step = self.get_step(pipe)
            if step == stop_step + 1:
                timesteps = pipe.scheduler.timesteps
                sigmas = pipe.scheduler.sigmas
                latents = kwargs.get('hidden_states')
                raise StopRun((latents, timesteps.cpu().numpy(), sigmas.cpu().numpy()))
            return original_forward(*args, **kwargs)
        
        original_forward = self.get_model(pipe).forward
        self.get_model(pipe).forward = wrapped_forward

        try:
            _ = pipe(**kwargs).images[0]
            self.get_model(pipe).forward = original_forward
        except StopRun as e:
            self.get_model(pipe).forward = original_forward
            return e.args[0]


class SDBackend(Backend):

    def __init__(self):
        super().__init__()
        self.PROMPT_KEYS = ['pooled_projections', 'encoder_hidden_states']

    def format_forward_output(self, noise_pred):
        return (noise_pred, )

    @staticmethod
    def get_sigmas(pipe, total_steps):
        timesteps = np.linspace(pipe.scheduler._sigma_to_t(pipe.scheduler.sigma_max), 
                                pipe.scheduler._sigma_to_t(pipe.scheduler.sigma_min), 
                                total_steps)
        sigmas = timesteps / pipe.scheduler.config.num_train_timesteps
        return sigmas

    def continue_generation(self, pipe, latents, current_step, 
                            stop_step, sigmas, **kwargs):

        def wrapped_step(*args, **kwargs):
            return original_step(args[0], args[1], latents, *args[3: ], **kwargs)

        def wrapped_forward(*args, **kwargs):
            step = self.get_step(pipe)
            if step == 1:
                if latents is not None:
                    kwargs['hidden_states'] = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
                    pipe.scheduler.step = wrapped_step
                pipe.scheduler.sigmas[-1] = 0.0
            else:
                pipe.scheduler.step = original_step
            if step == num_steps:
                tmp = kwargs['hidden_states'].clone()
                if pipe.do_classifier_free_guidance:
                    tmp = tmp[: tmp.shape[0] // 2, ...]
                return_dict['latents'] = tmp
            noise_pred = original_forward(*args, **kwargs)
            return noise_pred
        
        return_dict = {}
        tmp_sigmas = sigmas[current_step: stop_step].copy()
        original_steps = kwargs['num_inference_steps']
        original_sigmas = kwargs.get('sigmas')
        kwargs['sigmas'] = tmp_sigmas
        num_steps = stop_step - current_step
        kwargs['num_inference_steps'] = num_steps

        original_forward = self.get_model(pipe).forward
        original_step = pipe.scheduler.step
        self.get_model(pipe).forward = wrapped_forward

        image = pipe(**kwargs).images[0]
        kwargs['sigmas'] = original_sigmas
        kwargs['num_inference_steps'] = original_steps
        self.get_model(pipe).forward = original_forward
        pipe.scheduler.step = original_step
        return image, return_dict.get('latents', None)


    def go_forward(self, pipe, stop_step, **kwargs):
        def wrapped_forward(*args, **kwargs):
            step = self.get_step(pipe)
            if step == stop_step + 1:
                timesteps = pipe.scheduler.timesteps
                sigmas = pipe.scheduler.sigmas
                latents = kwargs.get('hidden_states')
                if pipe.do_classifier_free_guidance:
                    latents = latents[: latents.shape[0] // 2, ...]
                raise StopRun((latents, timesteps.cpu().numpy(), sigmas.cpu().numpy()))
            return original_forward(*args, **kwargs)
        
        original_forward = self.get_model(pipe).forward
        self.get_model(pipe).forward = wrapped_forward

        try:
            _ = pipe(**kwargs).images[0]
            self.get_model(pipe).forward = original_forward
        except StopRun as e:
            self.get_model(pipe).forward = original_forward
            return e.args[0]
        

class SDXLBackend(Backend):

    def __init__(self):
        super().__init__()
        self.PROMPT_KEYS = ['added_cond_kwargs', 'encoder_hidden_states']

    @staticmethod
    def get_model(pipe):
        return pipe.unet
    
    def format_forward_output(self, noise_pred):
        return (noise_pred, )

    @staticmethod
    def get_sigmas(pipe, total_steps):
        sch = pipe.scheduler
        if sch.config.timestep_spacing == "linspace":
            timesteps = np.linspace(
                0, sch.config.num_train_timesteps - 1, total_steps, dtype=np.float32
            )[::-1].copy()
        elif sch.config.timestep_spacing == "leading":
            step_ratio = sch.config.num_train_timesteps // total_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (
                (np.arange(0, total_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
            )
            timesteps += sch.config.steps_offset
        elif sch.config.timestep_spacing == "trailing":
            step_ratio = sch.config.num_train_timesteps / total_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (
                (np.arange(sch.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
            )
            timesteps -= 1
        sigmas = np.array(((1 - sch.alphas_cumprod) / sch.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)
        if sch.config.interpolation_type == "linear":
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        elif sch.config.interpolation_type == "log_linear":
            sigmas = torch.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), total_steps + 1).exp().numpy()

        if sch.config.final_sigmas_type == "sigma_min":
                sigma_last = ((1 - sch.alphas_cumprod[0]) / sch.alphas_cumprod[0]) ** 0.5
        elif sch.config.final_sigmas_type == "zero":
            sigma_last = 0

        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        return sigmas
    
    @staticmethod
    def get_scales(pipe):
        scale = 1.0
        if pipe.scheduler.is_scale_input_called:
            sigma = pipe.scheduler.sigmas[pipe.scheduler.step_index]
            scale = (sigma**2 + 1) ** 0.5
        return scale
    

    @staticmethod
    def scale_latents(pipe, latents):
        scale = SDXLBackend.get_scales(pipe)
        return latents / scale
    
    @staticmethod
    def descale_latents(pipe, latents):
        scale = SDXLBackend.get_scales(pipe)
        return latents * scale


    def continue_generation(self, pipe, latents, current_step, 
                            stop_step, sigmas, **kwargs):

        def wrapped_step(*args, **kwargs):
            return original_step(args[0], args[1], latents, *args[3: ], **kwargs)

        def wrapped_forward(*args, **kwargs):
            step = self.get_step(pipe)
            if step == 1:
                if latents is not None:
                    args = list(args)
                    tmp = self.scale_latents(pipe, latents)
                    args[0] = torch.cat([tmp] * 2) if pipe.do_classifier_free_guidance else tmp
                    args = tuple(args)
                    pipe.scheduler.step = wrapped_step
                pipe.scheduler.sigmas[-1] = 0.0
            else:
                pipe.scheduler.step = original_step
            if step == num_steps:
                tmp = args[0].clone()
                if pipe.do_classifier_free_guidance:
                    tmp = tmp[: tmp.shape[0] // 2, ...]
                return_dict['latents'] = self.descale_latents(pipe, tmp)
            noise_pred = original_forward(*args, **kwargs)
            return noise_pred
        
        return_dict = {}
        tmp_sigmas = sigmas[current_step: stop_step + 1].copy()
        original_steps = kwargs['num_inference_steps']
        original_sigmas = kwargs.get('sigmas')
        kwargs['sigmas'] = tmp_sigmas
        num_steps = stop_step - current_step
        kwargs['num_inference_steps'] = num_steps

        original_forward = self.get_model(pipe).forward
        original_step = pipe.scheduler.step
        self.get_model(pipe).forward = wrapped_forward

        image = pipe(**kwargs).images[0]
        kwargs['sigmas'] = original_sigmas
        kwargs['num_inference_steps'] = original_steps
        self.get_model(pipe).forward = original_forward
        pipe.scheduler.step = original_step
        return image, return_dict.get('latents', None)

    
    def go_forward(self, pipe, stop_step, **kwargs):
        def wrapped_forward(*args, **kwargs):
            step = self.get_step(pipe)
            if step == stop_step + 1:
                timesteps = pipe.scheduler.timesteps
                sigmas = pipe.scheduler.sigmas
                latents = args[0]
                if pipe.do_classifier_free_guidance:
                    latents = latents[: latents.shape[0] // 2, ...]
                latents = self.descale_latents(pipe, latents)
                raise StopRun((latents, timesteps.cpu().numpy(), sigmas.cpu().numpy()))
            return original_forward(*args, **kwargs)
        
        original_forward = self.get_model(pipe).forward
        self.get_model(pipe).forward = wrapped_forward

        try:
            _ = pipe(**kwargs).images[0]
            self.get_model(pipe).forward = original_forward
        except StopRun as e:
            self.get_model(pipe).forward = original_forward
            return e.args[0]