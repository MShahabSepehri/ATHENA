import os
import time
import functools
import numpy as np
from tqdm import tqdm
from utils import io_tools
from models import judge
from models.diffusion import aux
from transformers import CLIPProcessor, CLIPModel


ROOT = io_tools.get_root(__file__, 2)


def timer_return(func):
    """A decorator that returns the function's value and its execution time."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        run_time = time.perf_counter() - start_time
        return value, run_time
    return wrapper_timer


def _sum_profiler_flops(prof) -> int:
    """Best-effort FLOPs aggregation from a torch profiler session."""
    total_flops = 0
    try:
        for evt in prof.key_averages():
            flops = getattr(evt, "flops", None)
            if flops:
                total_flops += int(flops)
    except Exception:
        return 0
    return int(total_flops)


def flops_return(func):
    """A decorator that returns the function's value and its estimated FLOPs.

    Notes:
    - Uses `torch.profiler` with `with_flops=True` (best-effort; not all ops report FLOPs).
    - Adds CUDA activities when available; synchronizes around the call for more accurate capture.
    """
    @functools.wraps(func)
    def wrapper_flops(*args, **kwargs):
        # Internal controls (won't be forwarded to the wrapped function)
        force_profile = bool(kwargs.pop("_force_profile_flops", False))
        use_cache = bool(kwargs.pop("_cache_flops", True))

        cache = None
        cache_key = None
        if use_cache and len(args) > 0:
            self_obj = args[0]
            # Cache is stored on the instance to avoid global state.
            cache = getattr(self_obj, "_flops_cache", None)
            if cache is None:
                try:
                    cache = {}
                    setattr(self_obj, "_flops_cache", cache)
                except Exception:
                    cache = None

            # Best-effort key: FLOPs generally depend on model + step count + resolution.
            steps = None
            try:
                steps = getattr(self_obj, "generation_args", {}).get("num_inference_steps")
            except Exception:
                steps = None

            cache_key = (func.__qualname__, steps)
            if cache is not None and (not force_profile) and cache_key in cache:
                value = func(*args, **kwargs)
                return value, cache[cache_key]

        try:
            import torch
            from torch.profiler import profile, ProfilerActivity
        except Exception as exc:
            raise ImportError(
                "FLOPs profiling requires PyTorch (`torch`) with `torch.profiler` available."
            ) from exc

        activities = [ProfilerActivity.CPU]
        use_cuda = bool(torch.cuda.is_available())
        if use_cuda:
            activities.append(ProfilerActivity.CUDA)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        with profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            value = func(*args, **kwargs)

        if use_cuda:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        flops = _sum_profiler_flops(prof)

        if cache is not None and cache_key is not None:
            cache[cache_key] = flops
        return value, flops

    return wrapper_flops


class BaseGenModel():
    def __init__(self, model_args_path, dataset_path, judge_name, judge_args_path, seed=23, device='cuda'):
        self.key = None
        self.model_args_path = model_args_path
        self.load_dataset(dataset_path)
        self.device = device
        self.seed = seed
        self.set_judge(judge_name, judge_args_path)
        self.backend = None
        self.set_model_params()
        self.set_name(dataset_path)
        self.sigmas = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        func_list = [f for f in dir(cls) if 'generate' in f and f != 'estimage_generate']

        # decorator = flops_return
        decorator = timer_return

        for func_name in func_list:
            method = getattr(cls, func_name)
            decorated_method = decorator(method)
            setattr(cls, func_name, decorated_method)

    def load_dataset(self, dataset_path):
        if dataset_path is None:
            self.dataset = []
            return
        self.dataset = io_tools.load_json(dataset_path)

    def set_model_params(self):
        args = io_tools.load_json(self.model_args_path)
        self.model_args = args.get('model')
        self.generation_args = args.get('generation')
        self.strategy = args.get('strategy', {})
        self.config_name = args.get('name', 'default_config')

    def generate(self, prompt):
        pass

    def set_judge(self, judge_name, judge_config_path):
        if judge_name is None:
            self.judge = None
            return
        self.judge = judge.JUDGE_DICT.get(judge_name)(judge_name, judge_config_path)

    def check_resume_dict(self, resume_dict, id):
        tmp = False
        if id in resume_dict.keys():
            tmp = True
        if tmp:
            return resume_dict.get(id)
        return None
    
    @staticmethod
    def get_id(sample, num):
        if 'id' in sample.keys():
            return f"{10000 + int(sample.get('id'))}"
        return f"{num}_{sample.get('seed')}"
    
    def set_name(self, dataset_path):
        if dataset_path is None:
            self.name = f"{self.key}_generation"
        else:
            self.name = f"{self.key}_{dataset_path.split('/')[-1].replace('.json', '')}"
    
    def evaluate(self, resume_path, save_dir, precision=3):
        resume = io_tools.load_resume_dict(resume_path)
        save_path, image_save_path = self.check_folder(save_dir)
        results = {}
        counter = 1
        metrics = {'accuracy': 0, 'correct': 0, 'MAE': 0, 'MSE': 0, 'RMSE': 0, 'total': 0, 'average_generation_time': 0}
        for sample in tqdm(self.dataset):
            id = self.get_id(sample, counter)
            tmp = self.check_resume_dict(resume, id)
            image_path = f"{image_save_path}/{id}.jpg"
            result_dict = tmp if tmp is not None else self.sample_eval(sample, image_path)
            results[id] = result_dict
            io_tools.save_json(results, f'{save_path}/results.json')
            self.update_results(metrics, id, result_dict)
            io_tools.save_json(metrics, f'{save_path}/metrics.json')
            counter += 1

        self.print_results(metrics, precision=precision)
        io_tools.save_json(results, f'{save_path}/results.json')
        io_tools.save_json(metrics, f'{save_path}/metrics.json')

            
    def sample_eval(self, sample, save_path):
        prompt = sample.get('prompt')

        obj = sample.get('object')
        image, t = self.image_generation(sample, token=obj)
        image.save(save_path)
        target = sample.get('int_number')

        num_objects = self.judge.judge(image, obj)

        result_dict = {
            'image_path': save_path,
            'prompt': prompt,
            'object': obj,
            'target': target,
            'generated_objects': num_objects,
            'correct': (target == num_objects),
            'MAE': abs(int(target) - int(num_objects)),
            'MSE': (int(target) - int(num_objects)) ** 2,
            'generation_time': t,
        }
        return result_dict
    

    def image_generation(self, sample, token=None):
        prompt = sample.get('prompt')
        strategy = self.strategy.get('type', 'default')
        if strategy == 'default':
            image, t = self.generate(prompt)
        elif strategy == 'static':
            image, t = self.athena_static_generate(prompt,
                                                   target=sample.get('int_number'))
        elif strategy == 'feedback':
            image, t = self.athena_feedback_generate(prompt, 
                                          token=token, 
                                          target=sample.get('int_number'))
        elif strategy == 'adaptive':
            image, t = self.athena_adaptive_generate(prompt, 
                                            token=token, 
                                            target=sample.get('int_number'))
        else:
            raise NotImplementedError(f"Strategy {strategy} not implemented.")
        return image, t


    def check_folder(self, save_dir):
        tmp = self.name.replace(f'{self.key}_', f'{self.key}/{self.config_name}/')
        save_path = f'{save_dir}/{tmp}'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        image_save_path = f'{save_path}/images'
        if not os.path.isdir(image_save_path):
            os.makedirs(image_save_path)
        return save_path, image_save_path
    
    @staticmethod
    def print_results(metrics, precision=3):
        print_format = "{:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}"
        print(print_format.format('Total', 'Accuracy', 'Corrects', 'MAE', 'MSE', 'RMSE', 'Time (s)'))
        print(print_format.format(
                int(metrics.get('total')), 
                f"{metrics.get('accuracy'):.{precision}f}",
                int(metrics.get('correct')),
                f"{metrics.get('MAE'):.{precision}f}",
                f"{metrics.get('MSE'):.{precision}f}",
                f"{metrics.get('RMSE'):.{precision}f}",
                f"{metrics.get('average_generation_time'):.{precision}f}",
                ))

    
    @staticmethod
    def update_results(metrics, id, sample_result):
        tmp = metrics.get('total')
        total = tmp + 1
        metrics['correct'] += int(sample_result.get('correct'))
        metrics['accuracy'] = metrics.get('correct') / total
        metrics['MAE'] = (sample_result.get('MAE') + metrics.get('MAE') * tmp) / total
        metrics['MSE'] = (sample_result.get('MSE') + metrics.get('MSE') * tmp) / total
        metrics['RMSE'] = (metrics.get('MSE')) ** 0.5
        metrics['average_generation_time'] = (metrics.get('average_generation_time') * tmp + sample_result.get('generation_time')) / total
        metrics['total'] = total
    

    def athena_static_generate(self, prompt, target, **kwargs):
        response = self.backend.athena_static(self.model, 
                                         self.strategy.get('factor'), 
                                         self.strategy.get('steering_step'), 
                                         prompt=prompt,
                                         seed=self.seed,
                                         target=target,
                                         beta=self.strategy.get('beta', 1),
                                         replacement=self.strategy.get('replacement', ''),
                                         **self.generation_args,
                                         **kwargs)
        return response
    
    def athena_feedback_generate(self, prompt, token, target, **kwargs):
        count_func = lambda img: self.judge.judge(img, token)
        response = self.backend.athena_feedback(self.model, 
                                            count_func,
                                            target,
                                            self.generation_args.get('num_inference_steps'),
                                            self.strategy.get('factor'), 
                                            self.strategy.get('steering_step'), 
                                            prompt=prompt,
                                            estimate_step=self.strategy.get('estimate_step', 15),
                                            seed=self.seed,
                                            beta=self.strategy.get('beta', 1), 
                                            **self.generation_args,
                                            **kwargs)
        return response
    
    def athena_adaptive_generate(self, prompt, token, target, **kwargs):
        count_func = lambda img: self.judge.judge(img, token)
        response = self.backend.athena_adaptive(self.model, 
                                            count_func,
                                            target,
                                            self.generation_args.get('num_inference_steps'),
                                            self.strategy.get('factor'), 
                                            self.strategy.get('steering_step'), 
                                            prompt=prompt,
                                            estimate_step=self.strategy.get('estimate_step', 15),
                                            seed=self.seed,
                                            beta=self.strategy.get('beta', 1), 
                                            max_try=self.strategy.get('max_try', 2),
                                            r=self.strategy.get('r', 2),
                                            **self.generation_args,
                                            **kwargs)
        return response

    
class FluxGen(BaseGenModel):

    def set_model_params(self):
        global flux
        from models.diffusion import flux
        self.key = 'flux'
        self.backend = aux.FluxBackend()
        super().set_model_params()
        self.model = flux.load_model(**self.model_args)
        self.clip = None
        self.clip_processor = None

    def generate(self, prompt, **kwargs):
        response = flux.generate(self.model, prompt=prompt, 
                                 seed=self.seed, **self.generation_args,
                                 **kwargs)
        return response

class SDGen(BaseGenModel):

    def set_model_params(self):
        global sd
        from models.diffusion import sd
        self.key = 'sd'
        self.backend = aux.SDBackend()
        super().set_model_params()
        self.model = sd.load_model(**self.model_args)

    def generate(self, prompt, **kwargs):
        response = sd.generate(self.model, prompt=prompt, 
                               seed=self.seed, **self.generation_args,
                               **kwargs)
        return response


class SDXLGen(BaseGenModel):
    def __init__(self, model_args_path, dataset_path, judge_name, judge_args_path, seed=23, device='cuda'):
        super().__init__(model_args_path, dataset_path, judge_name, judge_args_path, seed, device)
        self.timesteps = None

    def set_model_params(self):
        global sdxl
        from models.diffusion import sdxl
        self.key = 'sdxl'
        self.backend = aux.SDXLBackend()
        super().set_model_params()
        self.model = sdxl.load_model(**self.model_args)

    def generate(self, prompt, **kwargs):
        response = sdxl.generate(self.model, prompt=prompt, seed=self.seed, **self.generation_args, **kwargs)
        return response
    
    def set_timesteps(self):
        steps = self.generation_args.get('num_inference_steps')
        step_ratio = self.model.scheduler.config.num_train_timesteps // steps
        timesteps = (
            (np.arange(0, steps) * step_ratio).round()[::-1].copy().astype(np.float32)
        )
        timesteps += self.model.scheduler.config.steps_offset
        self.timesteps = timesteps

GENERATION_CLASS_DICT = {
    'flux': FluxGen,
    'sd': SDGen,
    'sdxl': SDXLGen,
}

DEFAULT_MODEL_CONFIGS = {
    'flux': f'{ROOT}/configs/models/flux/1_dev.json',
    'sd': f'{ROOT}/configs/models/sd/3-5_large.json',
    'sdxl': f'{ROOT}/configs/models/sdxl/base.json',
}
