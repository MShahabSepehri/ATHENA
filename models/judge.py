import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from time import time
from tqdm import tqdm
from io import BytesIO
import os, sys, pathlib
from collections import defaultdict
from utils import io_tools


ROOT = io_tools.get_root(__file__, 2)


class Judge():
    NUMBER_LOOKUP_DICT = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
    }

    def __init__(self, key, config_path=None, device='cuda'):
        self.key = key
        self.device = device
        
        if config_path is None:
            config_path = JUDGE_DEFAULT_CONFIGS.get(self.key)
        self.set_params(config_path)

    @staticmethod
    def jpeg_roundtrip_in_memory(img: Image.Image) -> Image.Image:
            buf = BytesIO()
            img.save(buf, format="JPEG")  # subsampling=0 keeps chroma higher quality
            buf.seek(0)
            return Image.open(buf).convert("RGB")

    def set_params(self, config_path):
        config = io_tools.load_json(config_path)
        self.name = config.get('name', 'default_judge')
        self.model_config = config.get('model', {})
        self.inference_config = config.get('inference', {})

    def load_model(self):
        pass

    def run_detection(self, image, tokens):
        pass

    def judge(self, image, tokens, return_prompt=False, cvt_jpeg=True):
        if not isinstance(tokens, list):
            tokens = [tokens]
        if cvt_jpeg and isinstance(image, Image.Image):
            image = self.jpeg_roundtrip_in_memory(image)
        detections = self.run_detection(image, tokens)
        results = {token: detections.get(token, {}).get('count', 0) for token in tokens}
        results = results.get(tokens[0])
        if return_prompt:
            return results, tokens
        return results
    

    def check_folder(self, save_dir):
        if save_dir is None:
            return None
        save_path = f'{save_dir}/{self.name}'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        return save_path


    def evaluate_judge(self, data, resume_path, save_dir, precision=3, max_retries=3):
        resume = io_tools.load_resume_dict(resume_path)
        save_path = self.check_folder(save_dir)
        
        results = {}
        metrics = {'accuracy': 0, 'correct': 0, 'invalid': 0, 'MAE': 0, 'MSE': 0, 'RMSE': 0,'total': 0, 'average_judge_time': 0}

        for id in tqdm(data.keys()):
            tmp = resume.get(id)
            sample = data.get(id)
            result_dict = tmp if tmp is not None else self.sample_eval(sample, max_retries=max_retries)
            results[id] = result_dict
            self.update_results(metrics, id, result_dict)
            io_tools.save_json(results, f'{save_path}/results.json')
            io_tools.save_json(metrics, f'{save_path}/metrics.json')
        self.print_results(metrics, precision=precision)
        io_tools.save_json(results, f'{save_path}/results.json')
        io_tools.save_json(metrics, f'{save_path}/metrics.json')


    def sample_eval(self, sample, max_retries=3):
        counter = max_retries
        while counter > 0:
            st = time()
            response, prompt = self.judge(sample.get('image_path'), sample.get('object'), return_prompt=True)
            total_time = time() - st
            target = sample.get('label')
            results_dict = self.process_answer(response, target, total_time)
            results_dict['prompt'] = prompt
            if not results_dict.get('invalid'):
                return results_dict
            counter -= 1
        return results_dict
    

    @staticmethod
    def process_answer(answer, target, total_time):
        try:
            count = int(answer)
            invalid = False
        except:
            tmp = answer.lower().strip()
            if tmp in Judge.NUMBER_LOOKUP_DICT.keys():
                count = Judge.NUMBER_LOOKUP_DICT.get(tmp)
                invalid = False
            else:
                print(answer)
                count = None
                invalid = True
        if invalid:
            result_dict = {
                'response': answer,
                'invalid': invalid,
                'prediction': count,
                'target': target,
                'correct': 0,
                'MAE': None,
                'MSE': None,
                'judge_time': total_time
            }
        else:
            result_dict = {
                'response': answer,
                'invalid': invalid,
                'prediction': count,
                'target': target,
                'correct': count == target,
                'MAE': abs(count - target),
                'MSE': (count - target) ** 2,
                'judge_time': total_time
            }
        return result_dict

    @staticmethod
    def update_results(metrics, id, sample_result):
        if sample_result.get('invalid'):
            metrics['invalid'] += 1
            return
        tmp = metrics.get('total')
        total = tmp + 1
        metrics['correct'] += int(sample_result.get('correct'))
        metrics['accuracy'] = metrics.get('correct') / total
        metrics['MAE'] = (sample_result.get('MAE') + metrics.get('MAE') * tmp) / total
        metrics['MSE'] = (sample_result.get('MSE') + metrics.get('MSE') * tmp) / total
        metrics['RMSE'] = (metrics.get('MSE')) ** 0.5
        metrics['average_judge_time'] = (metrics.get('average_judge_time') * tmp + sample_result.get('judge_time')) / total
        metrics['total'] = total

    
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
                f"{metrics.get('average_judge_time'):.{precision}f}",
                ))

class GDINOJudge(Judge):

    def set_params(self, config_path):
        super().set_params(config_path)
        self.load_model(self.model_config.get('model_path'), repo_path=self.model_config.get('repo_path'))

        self.fp16_inference = self.inference_config.get('fp16_inference', True)        
        self.box_tr = self.inference_config.get('box_tr', 0.35)
        self.text_tr = self.inference_config.get('text_tr', 0.25)    

    @staticmethod
    def import_grounding_dino(path=None):
        if path is None:
            path = os.path.join(pathlib.Path(__file__).parent.parent.parent, "Grounded-Segment-Anything", "GroundingDINO")
        sys.path.insert(0, path)
        global inference
        from groundingdino.util import inference
        return os.path.join(path, "groundingdino/config/GroundingDINO_SwinT_OGC.py")

    def load_model(self, model_path, repo_path=None):
        config_path = self.import_grounding_dino(path=repo_path)
        self.model = inference.load_model(config_path, model_path)


    def run_detection(self, image, tokens):
        if not isinstance(tokens, list):
            tokens = [tokens]
        text = ". ".join(tokens) + "."
        if isinstance(image, str):
            image = Image.open(image)
        _, image = inference.process_image(image)
        if self.fp16_inference:
            image = image.half()
            self.model = self.model.half()
        boxes, logits, phrases = inference.predict(
            model=self.model,
            image=image,
            caption=text,
            box_threshold=self.box_tr,
            text_threshold=self.text_tr,
            device=self.device,
        )

        results = defaultdict(lambda: {"boxes": [], "logits": [], "count": 0})

        for box, logit, phrase in zip(boxes, logits, phrases):
            phrase = phrase.strip().lower()
            results[phrase]["boxes"].append(box.cpu().numpy())
            results[phrase]["logits"].append(float(logit))
            results[phrase]["count"] += 1
        results = dict(results)
        return results


JUDGE_DICT = {
    'gdino': GDINOJudge,
}

JUDGE_DEFAULT_CONFIGS = {
    'gdino': f'{ROOT}/configs/judges/gdino/vanilla.json',
}