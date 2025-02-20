#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            color_test_dir = Path(scene_dir) / "rgb_test"
            thermal_test_dir = Path(scene_dir) / "thermal_test"

            for method in os.listdir(color_test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = color_test_dir / method 

                print("method_dir is :", method_dir)

                color_gt_dir = method_dir/ "gt"
                color_renders_dir = method_dir / "renders"
                color_renders, color_gts, image_names = readImages(color_renders_dir, color_gt_dir)

                color_ssims = []
                color_psnrs = []
                color_lpipss = []

                for idx in tqdm(range(len(color_renders)), desc="Metric evaluation progress"):
                    color_ssims.append(ssim(color_renders[idx], color_gts[idx]))
                    color_psnrs.append(psnr(color_renders[idx], color_gts[idx]))
                    color_lpipss.append(lpips(color_renders[idx], color_gts[idx], net_type='vgg'))

                print(" color SSIM : {:>12.7f}".format(torch.tensor(color_ssims).mean(), ".5"))
                print(" color PSNR : {:>12.7f}".format(torch.tensor(color_psnrs).mean(), ".5"))
                print(" color LPIPS: {:>12.7f}".format(torch.tensor(color_lpipss).mean(), ".5"))

                print("")

                full_dict[scene_dir][method].update({"color SSIM": torch.tensor(color_ssims).mean().item(),
                                                        "color PSNR": torch.tensor(color_psnrs).mean().item(),
                                                        "color LPIPS": torch.tensor(color_lpipss).mean().item()
                                                        })
                per_view_dict[scene_dir][method].update({"color SSIM": {name: ssim for ssim, name in zip(torch.tensor(color_ssims).tolist(), image_names)},
                                                            "color PSNR": {name: psnr for psnr, name in zip(torch.tensor(color_psnrs).tolist(), image_names)},
                                                            "color LPIPS": {name: lp for lp, name in zip(torch.tensor(color_lpipss).tolist(), image_names)}
                                                            })
                
            for method in os.listdir(thermal_test_dir):
                print("Method:", method)

                method_dir =thermal_test_dir / method 

                print("method_dir is :", method_dir)

                thermal_gt_dir = method_dir/ "gt"
                thermal_renders_dir = method_dir / "renders"
                thermal_renders, thermal_gts, image_names = readImages(thermal_renders_dir, thermal_gt_dir)

                thermal_ssims = []
                thermal_psnrs = []
                thermal_lpipss = []

                for idx in tqdm(range(len(color_renders)), desc="Metric evaluation progress"):

                    thermal_ssims.append(ssim(thermal_renders[idx], thermal_gts[idx]))
                    thermal_psnrs.append(psnr(thermal_renders[idx], thermal_gts[idx]))
                    thermal_lpipss.append(lpips(thermal_renders[idx], thermal_gts[idx], net_type='vgg'))

                print(" thermal SSIM : {:>12.7f}".format(torch.tensor(thermal_ssims).mean(), ".5"))
                print(" thermal PSNR : {:>12.7f}".format(torch.tensor(thermal_psnrs).mean(), ".5"))
                print(" thermal LPIPS: {:>12.7f}".format(torch.tensor(thermal_lpipss).mean(), ".5"))

                print("")

                full_dict[scene_dir][method].update({"thermal SSIM": torch.tensor(thermal_ssims).mean().item(),
                                                        "thermal PSNR": torch.tensor(thermal_psnrs).mean().item(),
                                                        "thermal LPIPS": torch.tensor(thermal_lpipss).mean().item()
                                                        })
                per_view_dict[scene_dir][method].update({"thermal SSIM": {name: ssim for ssim, name in zip(torch.tensor(thermal_ssims).tolist(), image_names)},
                                                            "thermal PSNR": {name: psnr for psnr, name in zip(torch.tensor(thermal_psnrs).tolist(), image_names)},
                                                            "thermal LPIPS": {name: lp for lp, name in zip(torch.tensor(thermal_lpipss).tolist(), image_names)},
                                                            })
                
            

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)