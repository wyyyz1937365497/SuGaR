import os
import glob
import open3d as o3d
import torch
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points
from pytorch3d.io import save_obj
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR, extract_texture_image_and_uv_from_gaussians, load_refined_model
from sugar_utils.spherical_harmonics import SH2RGB

from rich.console import Console

def extract_mesh_and_texture_from_refined_sugar(args):
    CONSOLE = Console(width=120)

    n_skip_images_for_eval_split = 8
            
    # --- Scene data parameters ---
    source_path = args.scene_path
    use_train_test_split = args.eval
    
    # --- Vanilla 3DGS parameters ---
    iteration_to_load = args.iteration_to_load
    gs_checkpoint_path = args.checkpoint_path
    gs_checkpoint_path = os.path.normpath(gs_checkpoint_path) + os.sep
    
    # --- Fine model parameters ---
    refined_model_path = args.refined_model_path
    scene_name = os.path.basename(os.path.normpath(source_path))

    # Resolve refined checkpoint path robustly.
    # Supports direct .pt paths, directory paths containing checkpoints,
    # and short names like "15000.pt" by searching output/refined/<scene>.
    refined_model_path = os.path.normpath(refined_model_path)
    if os.path.isdir(refined_model_path):
        pt_candidates = sorted(glob.glob(os.path.join(refined_model_path, '*.pt')))
        if len(pt_candidates) == 0:
            raise FileNotFoundError(f"No .pt checkpoint found in directory: {refined_model_path}")
        refined_model_path = pt_candidates[-1]
    elif not os.path.exists(refined_model_path):
        refined_name = os.path.basename(refined_model_path)
        refined_scene_dir = os.path.join('output', 'refined', scene_name)
        recursive_candidates = sorted(glob.glob(os.path.join(refined_scene_dir, '**', refined_name), recursive=True))
        if len(recursive_candidates) == 1:
            CONSOLE.print(
                f"[yellow]Warning:[/yellow] Refined checkpoint '{args.refined_model_path}' not found, "
                f"using '{recursive_candidates[0]}' instead."
            )
            refined_model_path = os.path.normpath(recursive_candidates[0])
        elif len(recursive_candidates) > 1:
            raise FileNotFoundError(
                f"Refined checkpoint '{args.refined_model_path}' is ambiguous. "
                f"Multiple matches found under {refined_scene_dir}: {recursive_candidates}"
            )
        else:
            raise FileNotFoundError(
                f"Refined checkpoint not found: {args.refined_model_path}. "
                f"Expected a valid .pt path, or a file under {refined_scene_dir}."
            )

    if args.n_gaussians_per_surface_triangle is None:
        n_gaussians_per_surface_triangle = int(os.path.basename(os.path.normpath(refined_model_path)).split('_gaussperface')[-1])
    else:
        n_gaussians_per_surface_triangle = args.n_gaussians_per_surface_triangle
    
    # --- Output parameters ---
    if args.mesh_output_dir is None:
        scene_name = os.path.basename(os.path.normpath(args.scene_path))
        args.mesh_output_dir = os.path.join("output", "refined_mesh", scene_name)
    mesh_output_dir = args.mesh_output_dir
    os.makedirs(mesh_output_dir, exist_ok=True)
    
    mesh_save_path = os.path.basename(os.path.normpath(refined_model_path))
    if args.postprocess_mesh:
        mesh_save_path = mesh_save_path + '_postprocessed'
    mesh_save_path = mesh_save_path + '.obj'
    mesh_save_path = os.path.join(mesh_output_dir, mesh_save_path)
    
    # Resolve the coarse mesh path robustly.
    # `refined_model_path` can be either a checkpoint file (e.g. .../15000.pt)
    # or a refine directory. For checkpoints, derive from parent directory.
    sugar_mesh_path = getattr(args, 'mesh_path', None)
    if sugar_mesh_path is not None:
        sugar_mesh_path = os.path.normpath(sugar_mesh_path)
    else:
        refined_model_path_norm = os.path.normpath(refined_model_path)
        model_dir_name = os.path.basename(refined_model_path_norm)
        if os.path.isfile(refined_model_path_norm):
            model_dir_name = os.path.basename(os.path.dirname(refined_model_path_norm))

        sugar_mesh_filename = model_dir_name.split('_normalconsistency')[0].replace('sugarfine', 'sugarmesh') + '.ply'
        sugar_mesh_path = os.path.join('output', 'coarse_mesh', scene_name, sugar_mesh_filename)

    if not os.path.exists(sugar_mesh_path):
        coarse_scene_dir = os.path.join('output', 'coarse_mesh', scene_name)
        fallback_candidates = sorted(glob.glob(os.path.join(coarse_scene_dir, '*.ply')))
        if len(fallback_candidates) == 1:
            CONSOLE.print(f"[yellow]Warning:[/yellow] Coarse mesh not found at '{sugar_mesh_path}', using '{fallback_candidates[0]}' instead.")
            sugar_mesh_path = fallback_candidates[0]
        else:
            CONSOLE.print(
                f"[yellow]Warning:[/yellow] Coarse mesh not found: {sugar_mesh_path}. "
                "Will load mesh topology directly from refined checkpoint instead."
            )
    
    if args.square_size is None:
        if n_gaussians_per_surface_triangle == 1:
            # square_size = 5  # Maybe 4 already works
            square_size = 10  # Maybe 4 already works
        if n_gaussians_per_surface_triangle == 6:
            square_size = 10
    else:
        square_size = args.square_size
        
    # Postprocessing
    postprocess_mesh = args.postprocess_mesh
    if postprocess_mesh:
        postprocess_density_threshold = args.postprocess_density_threshold
        postprocess_iterations = args.postprocess_iterations
            
    CONSOLE.print('==================================================')
    CONSOLE.print("Starting extracting texture from refined SuGaR model:")
    CONSOLE.print('Scene path:', source_path)
    CONSOLE.print('Iteration to load:', iteration_to_load)
    CONSOLE.print('Vanilla 3DGS checkpoint path:', gs_checkpoint_path)
    CONSOLE.print('Refined model path:', refined_model_path)
    CONSOLE.print('Coarse mesh path:', sugar_mesh_path)
    CONSOLE.print('Mesh output directory:', mesh_output_dir)
    CONSOLE.print('Mesh save path:', mesh_save_path)
    CONSOLE.print('Number of gaussians per surface triangle:', n_gaussians_per_surface_triangle)
    CONSOLE.print('Square size:', square_size)
    CONSOLE.print('Postprocess mesh:', postprocess_mesh)
    CONSOLE.print('==================================================')
    
    # Set the GPU
    torch.cuda.set_device(args.gpu)
    
    # ==========================    
    
    # --- Loading Vanilla 3DGS model ---
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("Gaussian splatting checkpoint path:", gs_checkpoint_path)    
    CONSOLE.print(f"\nLoading Vanilla 3DGS model config {gs_checkpoint_path}...")
    
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=False,  # TODO: Check
        eval_split=use_train_test_split,
        eval_split_interval=n_skip_images_for_eval_split,
        )
    CONSOLE.print("Vanilla 3DGS Loaded.")
    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')
    CONSOLE.print(len(nerfmodel.gaussians._xyz) / 1e6, "M gaussians detected.")
    
    # --- Loading refined SuGaR model ---
    # Use checkpoint-native loading to reconstruct the exact mesh topology used in refinement.
    # This avoids size mismatches when external coarse meshes differ by simplification or cleaning.
    refined_sugar = load_refined_model(
        refined_sugar_path=refined_model_path,
        nerfmodel=nerfmodel,
        device=nerfmodel.device,
    )
    if (args.n_gaussians_per_surface_triangle is not None) and (
        refined_sugar.n_gaussians_per_surface_triangle != args.n_gaussians_per_surface_triangle
    ):
        CONSOLE.print(
            f"[yellow]Warning:[/yellow] Requested n_gaussians_per_surface_triangle={args.n_gaussians_per_surface_triangle}, "
            f"but checkpoint uses {refined_sugar.n_gaussians_per_surface_triangle}. Using checkpoint value."
        )
    n_gaussians_per_surface_triangle = refined_sugar.n_gaussians_per_surface_triangle
    refined_sugar.eval()
    
    if postprocess_mesh:
        CONSOLE.print("Postprocessing mesh by removing border triangles with low-opacity gaussians...")
        with torch.no_grad():
            new_verts = refined_sugar.surface_mesh.verts_list()[0].detach().clone()
            new_faces = refined_sugar.surface_mesh.faces_list()[0].detach().clone()
            new_normals = refined_sugar.surface_mesh.faces_normals_list()[0].detach().clone()
            
            # For each face, get the 3 edges
            edges0 = new_faces[..., None, (0,1)].sort(dim=-1)[0]
            edges1 = new_faces[..., None, (1,2)].sort(dim=-1)[0]
            edges2 = new_faces[..., None, (2,0)].sort(dim=-1)[0]
            all_edges = torch.cat([edges0, edges1, edges2], dim=-2)
            
            # We start by identifying the inside faces and border faces
            face_mask = refined_sugar.strengths[..., 0] > -1.
            for i in range(postprocess_iterations):
                CONSOLE.print("\nStarting postprocessing iteration", i)
                # We look for edges that appear in the list at least twice (their NN is themselves)
                edges_neighbors = knn_points(all_edges[face_mask].view(1, -1, 2).float(), all_edges[face_mask].view(1, -1, 2).float(), K=2)
                # If all edges of a face appear in the list at least twice, then the face is inside the mesh
                is_inside = (edges_neighbors.dists[0][..., 1].view(-1, 3) < 0.01).all(-1)
                # We update the mask by removing border faces
                face_mask[face_mask.clone()] = is_inside

            # We then add back border faces with high-density
            face_centers = new_verts[new_faces].mean(-2)
            face_densities = refined_sugar.compute_density(face_centers[~face_mask])
            face_mask[~face_mask.clone()] = face_densities > postprocess_density_threshold

            # And we create the new mesh and SuGaR model
            new_faces = new_faces[face_mask]
            new_normals = new_normals[face_mask]

            new_scales = refined_sugar._scales.reshape(len(face_mask), -1, 2)[face_mask].view(-1, 2)
            new_quaternions = refined_sugar._quaternions.reshape(len(face_mask), -1, 2)[face_mask].view(-1, 2)
            new_densities = refined_sugar.all_densities.reshape(len(face_mask), -1, 1)[face_mask].view(-1, 1)
            new_sh_coordinates_dc = refined_sugar._sh_coordinates_dc.reshape(len(face_mask), -1, 1, 3)[face_mask].view(-1, 1, 3)
            new_sh_coordinates_rest = refined_sugar._sh_coordinates_rest.reshape(len(face_mask), -1, 15, 3)[face_mask].view(-1, 15, 3)
            
            new_o3d_mesh = o3d.geometry.TriangleMesh()
            new_o3d_mesh.vertices = o3d.utility.Vector3dVector(new_verts.cpu().numpy())
            new_o3d_mesh.triangles = o3d.utility.Vector3iVector(new_faces.cpu().numpy())
            new_o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(new_normals.cpu().numpy())
            new_o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(torch.ones_like(new_verts).cpu().numpy())
            
            refined_sugar = SuGaR(
                nerfmodel=nerfmodel,
                points=None,
                colors=None,
                initialize=False,
                sh_levels=nerfmodel.gaussians.active_sh_degree+1,
                keep_track_of_knn=False,
                knn_to_track=0,
                beta_mode='average',
                surface_mesh_to_bind=new_o3d_mesh,
                n_gaussians_per_surface_triangle=refined_sugar.n_gaussians_per_surface_triangle,
                )
            refined_sugar._scales[...] = new_scales
            refined_sugar._quaternions[...] = new_quaternions
            refined_sugar.all_densities[...] = new_densities
            refined_sugar._sh_coordinates_dc[...] = new_sh_coordinates_dc
            refined_sugar._sh_coordinates_rest[...] = new_sh_coordinates_rest
        CONSOLE.print("Mesh postprocessed.")
    
    # Compute texture
    with torch.no_grad():
        if True:
            from sugar_extractors.texture import compute_textured_mesh_for_sugar_mesh
            textured_mesh = compute_textured_mesh_for_sugar_mesh(
                sugar=refined_sugar,
                square_size=square_size,
                n_sh=0,
                texture_with_gaussian_renders=True,
                bg_color=[0., 0., 0.]
            )
        else:
            verts_uv, faces_uv, texture_img = extract_texture_image_and_uv_from_gaussians(
                refined_sugar, square_size=square_size, n_sh=1, texture_with_gaussian_renders=True)
            
            textures_uv = TexturesUV(
                maps=texture_img[None], #texture_img[None]),
                verts_uvs=verts_uv[None],
                faces_uvs=faces_uv[None],
                sampling_mode='nearest',
                )
            textured_mesh = Meshes(
                verts=[refined_sugar.surface_mesh.verts_list()[0]],   
                faces=[refined_sugar.surface_mesh.faces_list()[0]],
                textures=textures_uv,
                )
    
    CONSOLE.print("Texture extracted.")
    
    CONSOLE.print("Saving textured mesh...")
    
    with torch.no_grad():
        save_obj(  
            mesh_save_path,
            verts=textured_mesh.verts_list()[0],
            faces=textured_mesh.faces_list()[0],
            verts_uvs=textured_mesh.textures.verts_uvs_list()[0],
            faces_uvs=textured_mesh.textures.faces_uvs_list()[0],
            texture_map=textured_mesh.textures.maps_padded()[0].clamp(0., 1.),
            )
        
    CONSOLE.print("Texture saved at:", mesh_save_path)
    return mesh_save_path