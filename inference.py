from pvdiffusion import PVDiffusion
import os
from configs.infer_config import get_parser
from decord import VideoReader, cpu
import torch

if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    os.makedirs(opts.save_dir,exist_ok=True)
    pvd = PVDiffusion(opts)
    pvd.nvs_single_view()
