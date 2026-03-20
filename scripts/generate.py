import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import argparse
from shutil import copy
from utils import io_tools
from models.generate import GENERATION_CLASS_DICT, DEFAULT_MODEL_CONFIGS 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='flux')
    parser.add_argument("--judge_name", type=str, default='gdino')
    parser.add_argument("--model_args_path", type=str, default=None)
    parser.add_argument("--judge_args_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--num_objects", type=int, required=True)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--seed", type=str, default=23)
    args = parser.parse_args()


    if args.model_args_path is None:
        args.model_args_path = DEFAULT_MODEL_CONFIGS.get(args.model_name)

    return args

if __name__ == "__main__":
    args = get_args()
    ROOT = io_tools.get_root(__file__, 2)

    if args.save_path is not None:
        save_path = args.save_path
    else:
        save_path = f'{ROOT}/Results/'
    answering_class = GENERATION_CLASS_DICT.get(args.model_name)
    ans_obj = answering_class(model_args_path=args.model_args_path, 
                              dataset_path=None,
                              judge_name=args.judge_name, judge_args_path=args.judge_args_path, seed=args.seed, device=args.device)
    
    sample = {
        'prompt': args.prompt,
        'int_number': args.num_objects
    }
    path, _ = ans_obj.check_folder(save_path)
    image, t = ans_obj.image_generation(sample, token=args.object)
    image.save(os.path.join(path, f'{args.object}_{args.num_objects}.jpg'))
    print(f'Generated image for {args.object} with count {args.num_objects} and prompt "{args.prompt}" at \"{path}\" with time {t:.2f} seconds.')