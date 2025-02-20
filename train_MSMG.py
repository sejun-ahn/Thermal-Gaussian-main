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

import os
import torch
import time
from random import randint
from utils.loss_utils import l1_loss, ssim, smoothness_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene_1, Scene_2, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians_1 = GaussianModel(dataset.sh_degree)
    gaussians_2 = GaussianModel(dataset.sh_degree)
    scene_1 = Scene_1(dataset, gaussians_1)
    scene_2 = Scene_2(dataset, gaussians_2)
    gaussians_1.training_setup(opt)
    gaussians_2.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians_1.restore(model_params, opt)
        gaussians_2.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack_1 = None
    viewpoint_stack_2 = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes_1 = None
                net_image_bytes_2 = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image_1 = render(custom_cam, gaussians_1, pipe, background, scaling_modifer)["render"]
                    net_image_bytes_1 = memoryview((torch.clamp(net_image_1, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    net_image_2 = render(custom_cam, gaussians_1, pipe, background, scaling_modifer)["render"]
                    net_image_bytes_2 = memoryview((torch.clamp(net_image_2, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes_1, dataset.source_path)
                network_gui.send(net_image_bytes_2, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians_1.update_learning_rate(iteration)
        gaussians_2.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians_1.oneupSHdegree()
            gaussians_2.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack_1:

            viewpoint_stack_1 = scene_1.getTrainCameras().copy()
        viewpoint_cam_1 = viewpoint_stack_1.pop(randint(0, len(viewpoint_stack_1)-1))    
        if not viewpoint_stack_2:

            viewpoint_stack_2 = scene_2.getTrainCameras().copy()
        viewpoint_cam_2 = viewpoint_stack_2.pop(randint(0, len(viewpoint_stack_2)-1))
        

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg_1 = render(viewpoint_cam_1, gaussians_1, pipe, bg)
        image_1, viewspace_point_tensor_1, visibility_filter_1, radii_1 = render_pkg_1["render"], render_pkg_1["viewspace_points"], render_pkg_1["visibility_filter"], render_pkg_1["radii"]
        render_pkg_2 = render(viewpoint_cam_2, gaussians_2, pipe, bg)
        image_2, viewspace_point_tensor_2, visibility_filter_2, radii_2 = render_pkg_2["render"], render_pkg_2["viewspace_points"], render_pkg_2["visibility_filter"], render_pkg_2["radii"]

        # Loss
        smoothloss_thermal = smoothness_loss(image_2)
        
        gt_image_1 = viewpoint_cam_1.original_image.cuda()
        Ll1_1 = l1_loss(image_1, gt_image_1)
        loss_1 = (1.0 - opt.lambda_dssim) * Ll1_1 + opt.lambda_dssim * (1.0 - ssim(image_1, gt_image_1))
        
        gt_image_2 = viewpoint_cam_2.original_image.cuda()
        Ll1_2 = l1_loss(image_2, gt_image_2)

        loss_2 = (1.0 - opt.lambda_dssim) * Ll1_2 + opt.lambda_dssim * (1.0 - ssim(image_2, gt_image_2)) + 0.6 *smoothloss_thermal
        
        lambda_rgb = gaussians_1.get_xyz.size(0) / ( gaussians_1.get_xyz.size(0) + gaussians_2.get_xyz.size(0))
        lambda_temp = gaussians_2.get_xyz.size(0) / ( gaussians_1.get_xyz.size(0) + gaussians_2.get_xyz.size(0))


        total_loss = 0.3 * lambda_temp *loss_1 + 0.6 * lambda_rgb * loss_2
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1_1 , total_loss ,Ll1_2, total_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene_1,scene_2, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene_1.save(iteration)
                scene_2.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians_1.max_radii2D[visibility_filter_1] = torch.max(gaussians_1.max_radii2D[visibility_filter_1], radii_1[visibility_filter_1])
                gaussians_1.add_densification_stats(viewspace_point_tensor_1, visibility_filter_1)
                gaussians_2.max_radii2D[visibility_filter_2] = torch.max(gaussians_2.max_radii2D[visibility_filter_2], radii_2[visibility_filter_2])
                gaussians_2.add_densification_stats(viewspace_point_tensor_2, visibility_filter_2)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians_1.densify_and_prune(opt.densify_grad_threshold, 0.005, scene_1.cameras_extent, size_threshold)
                    gaussians_2.densify_and_prune(opt.densify_grad_threshold, 0.005, scene_2.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians_1.reset_opacity()
                    gaussians_2.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians_1.optimizer.step()
                gaussians_1.optimizer.zero_grad(set_to_none = True)
                gaussians_2.optimizer.step()
                gaussians_2.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians_1.capture(), iteration), scene_1.model_path + "/chkpnt_1" + str(iteration) + ".pth")
                torch.save((gaussians_2.capture(), iteration), scene_2.model_path + "/chkpnt_2" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1_1,loss_1, Ll1_2, loss_2, l1_loss, elapsed, testing_iterations, scene_1 : Scene_1,scene_2 : Scene_2, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss_1', Ll1_1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss_1', loss_1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_2', Ll1_2.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss_2', loss_2.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        
        torch.cuda.empty_cache()
        validation_configs_1 = ({'name': 'test', 'cameras' : scene_1.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene_1.getTrainCameras()[idx % len(scene_1.getTrainCameras())] for idx in range(5, 30, 5)]})
        validation_configs_2 = ({'name': 'test', 'cameras' : scene_2.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene_2.getTrainCameras()[idx % len(scene_2.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs_1:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene_1.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                
                print("\n[ITER {}] Evaluating color {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/color/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/color/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/color/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/color/loss_viewpoint - ssim', ssim_test, iteration)
                    
        for config in validation_configs_2:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene_2.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating thermal {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/thermal/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/thermal/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/thermal/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/thermal/loss_viewpoint - ssim', ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene_1/opacity_histogram", scene_1.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points_1', scene_1.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram("scene_2/opacity_histogram", scene_2.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points_2', scene_2.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
