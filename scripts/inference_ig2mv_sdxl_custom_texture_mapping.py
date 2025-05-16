import argparse

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
import cv2
import trimesh

import sys
import os
# add the parent directory to successfully import mvadapter as a module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import pdb
import trimesh

from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import get_orthogonal_camera, make_image_grid, tensor_to_image
from mvadapter.utils.render import NVDiffRastContextWrapper, load_mesh, render


def bake_uv_texture_from_views(images_tensor, uv_coords, masks, texres=1024, inpaint=True):
    """
    Args:
        images_tensor: [N, H, W, 3], RGB images from N views
        uv_coords:     [N, H, W, 2], UV coords per view, in [0, 1]
        masks:         [N, H, W, 1], binary mask per view
        texres:        texture resolution (square)
        inpaint:       whether to apply OpenCV inpainting

    Returns:
        tex_img:       np.uint8 texture map of shape [texres, texres, 3]
    """
    device = images_tensor.device
    N, H, W, _ = images_tensor.shape

    # Initialize accumulators
    tex_acc = torch.zeros((texres * texres, 3), dtype=torch.float32, device=device)
    weight = torch.zeros((texres * texres, 1), dtype=torch.float32, device=device)


    # Flatten inputs
    uv = (uv_coords * (texres - 1)).long().view(-1, 2)
    rgb = images_tensor.view(-1, 3)
    mask_flat = masks.view(-1) > 0
    uv = (uv_coords * (texres - 1)).long().view(-1, 2) # shape [3538944, 2]
    rgb = images_tensor.view(-1, 3) # [3538944, 3]
    mask_flat = masks.view(-1) > 0 # [3538944]

    # Only keep valid UVs
    uv = uv[mask_flat]
    rgb = rgb[mask_flat]
    uv = uv[mask_flat] # shape [1795804, 2]
    uv_coords = uv_coords.view(-1, 2)[mask_flat] 
    rgb = rgb[mask_flat] # shape [1795804, 3]

    # Flatten 2D UV to 1D index
    u, v = uv[:, 0], uv[:, 1]
    idx = u + (texres - 1 - v) * texres  # Y-down image space
    idx = idx.view(-1, 1).expand(-1, 3)


    # Initialize accumulators
    tex_acc = torch.zeros((texres * texres, 3), dtype=torch.float32, device=device)
    weight = torch.zeros((texres * texres, 1), dtype=torch.float32, device=device)

    tex_acc = tex_acc.scatter_add(0, idx, rgb)
    weight = weight.scatter_add(0, idx[:, :1], torch.ones_like(idx[:, :1], dtype=torch.float32))

    tex_acc = tex_acc / (weight + 1e-6)
    tex_img = tex_acc.view(texres, texres, 3).clamp(0, 1)

    # Convert to CPU image
    tex_img_np = (tex_img.cpu().numpy() * 255).astype(np.uint8)

    if inpaint:
        inpaint_mask = (weight.view(texres, texres).cpu().numpy() == 0).astype(np.uint8)
        tex_img_np = cv2.inpaint(tex_img_np, inpaint_mask, 3, cv2.INPAINT_TELEA)

    return tex_img_np


def prepare_pipeline(
    base_model,
    vae_model,
    unet_model,
    lora_model,
    adapter_path,
    scheduler,
    num_views,
    device,
    dtype,
):
    # Load vae and unet if provided
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
    if unet_model is not None:
        pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)

    # Prepare pipeline
    pipe: MVAdapterI2MVSDXLPipeline
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

    # Load scheduler if provided
    scheduler_class = None
    if scheduler == "ddpm":
        scheduler_class = DDPMScheduler
    elif scheduler == "lcm":
        scheduler_class = LCMScheduler

    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=scheduler_class,
    )
    pipe.init_custom_adapter(
        num_views=num_views, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0
    )
    pipe.load_custom_adapter(
        adapter_path, weight_name="mvadapter_ig2mv_sdxl.safetensors"
    )

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    # load lora if provided
    if lora_model is not None:
        model_, name_ = lora_model.rsplit("/", 1)
        pipe.load_lora_weights(model_, weight_name=name_)

    return pipe


def remove_bg(image, net, transform, device):
    image_size = image.size
    input_images = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = net(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image


def preprocess_image(image: Image.Image, height, width):
    image = np.array(image)
    alpha = image[..., 3] > 0
    H, W = alpha.shape
    # get the bounding box of alpha
    y, x = np.where(alpha)
    y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
    x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
    image_center = image[y0:y1, x0:x1]
    # resize the longer side to H * 0.9
    H, W, _ = image_center.shape
    if H > W:
        W = int(W * (height * 0.9) / H)
        H = int(height * 0.9)
    else:
        H = int(H * (width * 0.9) / W)
        W = int(width * 0.9)
    image_center = np.array(Image.fromarray(image_center).resize((W, H)))
    # pad to H, W
    start_h = (height - H) // 2
    start_w = (width - W) // 2
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[start_h : start_h + H, start_w : start_w + W] = image_center
    image = image.astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def run_pipeline(
    pipe,
    mesh_path,
    num_views,
    text,
    image,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    seed,
    remove_bg_fn=None,
    reference_conditioning_scale=1.0,
    negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
    lora_scale=1.0,
    device="cuda",
    uv_unwrap=False
):
    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        device=device,
    )
    ctx = NVDiffRastContextWrapper(device=device)

    mesh = load_mesh(mesh_path, rescale=True, device=device, uv_unwrap=True, default_uv_size=512, merge_vertices=False)
    v_tex = mesh.v_tex
    t_tex_idx = mesh.t_tex_idx
    mesh = load_mesh(mesh_path, rescale=True, device=device, uv_unwrap=uv_unwrap, default_uv_size=512, merge_vertices=False)

    render_out = render(
        ctx,
        mesh,
        cameras,
        height=height,
        width=width,
        render_attr=False,
        render_uv=True,     # ← get UVs instead
        render_normal=True,
        render_depth=False,
        normal_background=0.0,
    )
    pos_images = tensor_to_image((render_out.pos + 0.5).clamp(0, 1), batched=True)
    normal_images = tensor_to_image(
        (render_out.normal / 2 + 0.5).clamp(0, 1), batched=True
    )
    control_images = (
        torch.cat(
            [
                (render_out.pos + 0.5).clamp(0, 1),
                (render_out.normal / 2 + 0.5).clamp(0, 1),
            ],
            dim=-1,
        )
        .permute(0, 3, 1, 2)
        .to(device)
    )

    # Prepare image
    reference_image = Image.open(image) if isinstance(image, str) else image
    if remove_bg_fn is not None:
        reference_image = remove_bg_fn(reference_image)
        reference_image = preprocess_image(reference_image, height, width)
    elif reference_image.mode == "RGBA":
        reference_image = preprocess_image(reference_image, height, width)

    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        text,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_views,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=reference_image,
        reference_conditioning_scale=reference_conditioning_scale,
        negative_prompt=negative_prompt,
        cross_attention_kwargs={"scale": lora_scale},
        **pipe_kwargs,
    ).images


    # Save camera extrinsics and intrinsics (MVP components)
    cam_dict = {
        "w2c": cameras.w2c.cpu().numpy(),         # [N, 4, 4]
        "proj_mtx": cameras.proj_mtx.cpu().numpy(),  # [N, 4, 4]
        "c2w": cameras.c2w.cpu().numpy(),         # [N, 4, 4]
    }

    ### Map multi-view images to texture map ###
    # Prepare accumulation buffers
    # convert images to tensor
    images_tensor = torch.stack([torch.tensor(np.array(image)).float() / 255.0 for image in images]).to(device) # [6, 768, 768, 3], range [0, 1]
    uv_coords = render_out.uv  # [6, 768, 768, 2]
    masks = render_out.mask.unsqueeze(-1)  # [6, 768, 768, 1]

    tex_img = bake_uv_texture_from_views(
        images_tensor=images_tensor,     # [6, 768, 768, 3]
        uv_coords=render_out.uv,         # [6, 768, 768, 2]
        masks=render_out.mask.unsqueeze(-1).float(),  # [6, 768, 768, 1]
        texres=1024
    )
    uv_coords = render_out.uv.cpu().numpy()  # [6, 768, 768, 2]
    images_tensor = images_tensor.cpu().numpy()  # [6, 768, 768, 3]
    masks = render_out.mask.unsqueeze(-1).cpu().numpy()  # [6, 768, 768, 1]
    # images and cam_dict are numpy arrays, others should be tensors
    # print(f"check types: images: {type(images)}, pos_images: {type(pos_images)}, normal_images: {type(normal_images)}, reference_image: {type(reference_image)}, cam_dict: {type(cam_dict)}, uv_coords: {type(uv_coords)}, masks: {type(masks)}, mesh.v_pos: {type(mesh.v_pos)}, mesh.v_tex: {type(mesh.v_tex)}, mesh.t_tex_idx: {type(mesh.t_tex_idx)}, mesh.t_pos_idx: {type(mesh.t_pos_idx)}") # before cam_dict: PIL images, after cam_dict: torch tensors, 

    return images, pos_images, normal_images, reference_image, cam_dict, tex_img, uv_coords, images_tensor, masks, v_tex, t_tex_idx
    return images, pos_images, normal_images, reference_image, cam_dict, uv_coords, masks, mesh.v_pos, mesh.v_tex, mesh.t_tex_idx, mesh.t_pos_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Models
    parser.add_argument(
        "--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument(
        "--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix"
    )
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="huanngzh/mv-adapter")
    parser.add_argument("--num_views", type=int, default=6)
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    # Inference
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, required=False, default="high quality")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--reference_conditioning_scale", type=float, default=1.0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="watermark, ugly, deformed, noisy, blurry, low contrast",
    )
    parser.add_argument("--output", type=str, default="output.png")
    # Extra
    parser.add_argument("--remove_bg", action="store_true", help="Remove background")
    parser.add_argument("--save_3D", action="store_true", help="save textured 3D mesh as output")
    args = parser.parse_args()
    output_path = args.output.rsplit(".", 1)[0] 
    tex_path = output_path + "_baked_texture.png"
    device = args.device

    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=args.unet_model,
        lora_model=args.lora_model,
        adapter_path=args.adapter_path,
        scheduler=args.scheduler,
        num_views=args.num_views,
        device=args.device,
        dtype=torch.float16,
    )

    if args.remove_bg:
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        birefnet.to(args.device)
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, args.device)
    else:
        remove_bg_fn = None

    images, pos_images, normal_images, reference_image, cam_dict, tex_img, uv_coords, images_tensor, masks, v_tex, t_tex_idx = run_pipeline(
    images, pos_images, normal_images, reference_image, cam_dict, uv_coords, masks, v_pos, v_tex, t_tex_idx, t_pos_idx = run_pipeline(
        pipe,
        mesh_path=args.mesh,
        num_views=args.num_views,
        text=args.text,
        image=args.image,
        height=768,
        width=768,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        lora_scale=args.lora_scale,
        reference_conditioning_scale=args.reference_conditioning_scale,
        negative_prompt=args.negative_prompt,
        device=args.device,
        remove_bg_fn=remove_bg_fn,
        uv_unwrap=args.save_3D,
    )

    # Save image grid for visual inspection
    make_image_grid(images, rows=1).save(args.output)
    make_image_grid(pos_images, rows=1).save(output_path+ "_pos.png")
    make_image_grid(normal_images, rows=1).save(
        output_path + "_nor.png"
    )
    reference_image.save(output_path + "_reference.png")
    # save camera params
    cam_dict_path = output_path + "_camera_params.npz"
    np.savez(cam_dict_path, **cam_dict)
    Image.fromarray(tex_img).save(tex_path)
    print(f"Saved baked texture to {tex_path}")
    np.savez(output_path + "_uv_coords.npz", uv_coords=uv_coords, images_tensor=images_tensor, masks=masks, v_tex=v_tex.cpu().numpy(), t_tex_idx=t_tex_idx.cpu().numpy())

    # use multi-view images and camera params to bake texture to mesh
    print(f"### Bake Multi-View Texture to Mesh ###")

    # Assign to mesh and export
    textured_mesh = load_mesh(args.mesh, rescale=True, device=args.device, uv_unwrap=False, merge_vertices=False)

    # Get numpy arrays
    verts = textured_mesh.v_pos.cpu().numpy()        # [V, 3]
    uvs = v_tex          # [V, 2]
    
    texture = Image.open(tex_path).convert("RGB")
    texture = np.array(texture) / 255.0  # [1024, 1024, 3], range [0,1]

    H, W, _ = texture.shape

    # Clip UVs to [0,1]
    uvs = np.clip(uvs, 0, 1)

    # Convert UVs to pixel coords
    u_px = (uvs[:, 0] * (W - 1)).astype(np.int32)
    v_px = ((1 - uvs[:, 1]) * (H - 1)).astype(np.int32)  # flip Y for image space

    # Sample texture
    colors = texture[v_px, u_px]  # [V, 3], in [0,1]

    # Clamp and convert to float
    colors = np.clip(colors, 0, 1)

    faces = textured_mesh.t_pos_idx.cpu().numpy()  # [F, 3]

    obj_path = output_path + "_textured.obj"
    with open(obj_path, "w") as f:
        for i in range(verts.shape[0]):
            x, y, z = verts[i]
            r, g, b = colors[i]
            f.write(f"v {x} {y} {z} {r} {g} {b}\n")

        for i in range(faces.shape[0]):
            a, b, c = faces[i] + 1  # .obj is 1-indexed
            f.write(f"f {a} {b} {c}\n")


    if args.save_3D:
        # Save output arrays for debugging and texturing
        cam_dict_path = output_path + "_camera_params.npz"
        np.savez(cam_dict_path, **cam_dict)

        np.savez(output_path + "_uv_coords.npz", images=images, uv_coords=uv_coords.cpu().numpy(), masks=masks.cpu().numpy(), v_pos=v_pos.cpu().numpy(), v_tex=v_tex.cpu().numpy(), t_tex_idx=t_tex_idx.cpu().numpy(), t_pos_idx=t_pos_idx.cpu().numpy())

        ### Load data from saved npz file ###
        uv_param = np.load(output_path + "_uv_coords.npz")
        images = uv_param["images"]  # [6, 768, 768, 3]
        uv_coords = torch.from_numpy(uv_param["uv_coords"]).to(device)  # [6, 768, 768, 2]
        masks = torch.from_numpy(uv_param["masks"]).to(device)  # [6, 768, 768, 1]
        v_pos = torch.from_numpy(uv_param["v_pos"]).to(device)  # [V, 3]
        v_tex = torch.from_numpy(uv_param["v_tex"]).to(device)  # [V, 2]
        t_tex_idx = torch.from_numpy(uv_param["t_tex_idx"]).to(device)  # [F, 3]
        t_pos_idx = torch.from_numpy(uv_param["t_pos_idx"]).to(device)  # [F, 3]
        print(f"check loaded data shape, images: {images.shape}, uv_coords: {uv_coords.shape}, masks: {masks.shape}, v_pos: {v_pos.shape}, v_tex: {v_tex.shape}, t_tex_idx: {t_tex_idx.shape}, t_pos_idx: {t_pos_idx.shape}")


        ### Map multi-view images to texture map ###
        # convert images to tensor
        images_tensor = torch.stack([torch.tensor(np.array(image)).float() / 255.0 for image in images]).to(device) # [6, 768, 768, 3], range [0, 1]

        tex_img = bake_uv_texture_from_views(
            images_tensor=images_tensor,     # [6, 768, 768, 3]
            uv_coords=uv_coords,         # [6, 768, 768, 2]
            masks=masks.float(),  # [6, 768, 768, 1]
            texres=512,
            inpaint=True
        ) #  np.array, shape [512, 512, 3], range [0, 255]
        verts = v_pos.cpu().numpy()  # [V, 3]
        uvs = v_tex.cpu().numpy()  # [V, 2]

        # Export the textured mesh
        texture = tex_img/255.0  # [1024, 1024, 3], range [0,1]
        H, W, _ = texture.shape

        # Clip UVs to [0,1]
        uvs = np.clip(uvs, 0, 1)

        # Convert UVs to pixel coords
        u_px = (uvs[:, 0] * (W - 1)).astype(np.int32)
        v_px = ((1 - uvs[:, 1]) * (H - 1)).astype(np.int32)  # flip Y for image space

        # Sample texture
        colors = texture[v_px, u_px]  # [V, 3], in [0,1]

        # Clamp and convert to float
        colors = np.clip(colors, 0, 1)

        faces = t_pos_idx.cpu().numpy()  # [F, 3]

        obj_path = output_path + "_textured_inpaint.obj"
        with open(obj_path, "w") as f:
            for i in range(verts.shape[0]):
                x, y, z = verts[i]
                r, g, b = colors[i]
                f.write(f"v {x} {y} {z} {r} {g} {b}\n")

            for i in range(faces.shape[0]):
                a, b, c = faces[i] + 1  # .obj is 1-indexed
                f.write(f"f {a} {b} {c}\n")

        print(f"Saved textured mesh to {obj_path}")


        ### Convert textured .obj mesh to .glb format ###
        obj_path = output_path + "_textured_inpaint.obj"
        output_mesh_path = output_path + "_textured.glb"
        # Load .obj
        scene = trimesh.load(obj_path, process=False)
        if isinstance(scene, trimesh.Trimesh):
            mesh = scene
        elif isinstance(scene, trimesh.scene.Scene):
            mesh = trimesh.Trimesh()
            for obj in scene.geometry.values():
                mesh = trimesh.util.concatenate([mesh, obj])
        else:
            print(type(scene))
            raise ValueError(f"Unknown mesh type at {obj_path}.")

        # Ensure RGBA vertex colors
        if mesh.visual.vertex_colors.shape[1] == 3:
            alpha = 255 * np.ones((mesh.visual.vertex_colors.shape[0], 1), dtype=np.uint8)
            mesh.visual.vertex_colors = np.hstack([mesh.visual.vertex_colors, alpha])

        # Export to .glb
        mesh.export(output_mesh_path, file_type='glb')

        print(f"✅ Converted {obj_path} to {output_mesh_path}")