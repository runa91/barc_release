
# part of the code from 
#   https://github.com/benjiebob/SMALify/blob/master/smal_fitter/p3d_renderer.py

import torch
import torch.nn.functional as F
from scipy.io  import loadmat
import numpy as np
# import config

import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    PointLights, HardPhongShader, SoftSilhouetteShader, Materials, Textures, 
    DirectionalLights
)
from pytorch3d.renderer import TexturesVertex, SoftPhongShader
from pytorch3d.io import load_objs_as_meshes

MESH_COLOR_0 = [0, 172, 223]
MESH_COLOR_1 = [172, 223, 0]


'''
Explanation of the shift between projection results from opendr and pytorch3d:
    (0, 0, ?) will be projected to 127.5 (pytorch3d) instead of 128 (opendr)
    imagine you have an image of size 4:
    middle of the first pixel is 0
    middle of the last pixel is 3
    => middle of the imgae would be 1.5 and not 2!
    so in order to go from pytorch3d predictions to opendr we would calculate: p_odr = p_p3d * (128/127.5)
To reproject points (p3d) by hand according to this pytorch3d renderer we would do the following steps:
    1.) build camera matrix
        K = np.array([[flength, 0, c_x],
                    [0, flength, c_y],
                    [0, 0, 1]], np.float)
    2.) we don't need to add extrinsics, as the mesh comes with translation (which is 
        added within smal_pytorch). all 3d points are already in the camera coordinate system.
        -> projection reduces to p2d_proj = K*p3d
    3.) convert to pytorch3d conventions (0 in the middle of the first pixel)
        p2d_proj_pytorch3d = p2d_proj / image_size * (image_size-1.)
renderer.py - project_points_p3d: shows an example of what is described above, but 
    same focal length for the whole batch

'''

class SilhRenderer(torch.nn.Module):
    def __init__(self, image_size, adapt_R_wldo=False):
        super(SilhRenderer, self).__init__()
        # see: https://pytorch3d.org/files/fit_textured_mesh.py, line 315
        # adapt_R=True is True for all my experiments
        # image_size: one number, integer
        # -----
        # set mesh color
        self.register_buffer('mesh_color_0', torch.FloatTensor(MESH_COLOR_0))
        self.register_buffer('mesh_color_1', torch.FloatTensor(MESH_COLOR_1))
        # prepare extrinsics, which in our case don't change
        R = torch.Tensor(np.eye(3)).float()[None, :, :]
        T = torch.Tensor(np.zeros((1, 3))).float()
        if adapt_R_wldo:
            R[0, 0, 0] = -1
        else:       # used for all my own experiments
            R[0, 0, 0] = -1
            R[0, 1, 1] = -1            
        self.register_buffer('R', R)
        self.register_buffer('T', T)
        # prepare that part of the intrinsics which does not change either
        # principal_point_prep = torch.Tensor([self.image_size / 2., self.image_size / 2.]).float()[None, :].float().to(device) 
        # image_size_prep = torch.Tensor([self.image_size, self.image_size]).float()[None, :].float().to(device) 
        self.img_size_scalar = image_size
        self.register_buffer('image_size', torch.Tensor([image_size, image_size]).float()[None, :].float())
        self.register_buffer('principal_point', torch.Tensor([image_size / 2., image_size / 2.]).float()[None, :].float())
        # Rasterization settings for differentiable rendering, where the blur_radius
        # initialization is based on Liu et al, 'Soft Rasterizer: A Differentiable 
        # Renderer for Image-based 3D Reasoning', ICCV 2019
        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        self.raster_settings_soft = RasterizationSettings(
            image_size=image_size,      # 128
            blur_radius=np.log(1. / 1e-4 - 1.)*self.blend_params.sigma,  
            faces_per_pixel=100)     #50, 
        # Renderer for Image-based 3D Reasoning', body part segmentation
        self.blend_params_parts = BlendParams(sigma=2*1e-4, gamma=1e-4)
        self.raster_settings_soft_parts = RasterizationSettings(
            image_size=image_size,      # 128
            blur_radius=np.log(1. / 1e-4 - 1.)*self.blend_params_parts.sigma,  
            faces_per_pixel=60)     #50, 
        # settings for visualization renderer
        self.raster_settings_vis = RasterizationSettings(
                image_size=image_size, 
                blur_radius=0.0, 
                faces_per_pixel=1)

    def _get_cam(self, focal_lengths):
        device = focal_lengths.device
        bs = focal_lengths.shape[0]
        if pytorch3d.__version__ == '0.2.5':
            cameras = PerspectiveCameras(device=device,
                focal_length=focal_lengths.repeat((1, 2)), 
                principal_point=self.principal_point.repeat((bs, 1)), 
                R=self.R.repeat((bs, 1, 1)), T=self.T.repeat((bs, 1)), 
                image_size=self.image_size.repeat((bs, 1)))
        elif pytorch3d.__version__ == '0.6.1':
            cameras = PerspectiveCameras(device=device, in_ndc=False,
            focal_length=focal_lengths.repeat((1, 2)), 
            principal_point=self.principal_point.repeat((bs, 1)), 
            R=self.R.repeat((bs, 1, 1)), T=self.T.repeat((bs, 1)), 
            image_size=self.image_size.repeat((bs, 1)))
        else: 
            print('this part depends on the version of pytorch3d, code was developed with 0.2.5')
            raise ValueError
        return cameras

    def _get_visualization_from_mesh(self, mesh, cameras, lights=None):
        # color renderer for visualization
        with torch.no_grad():
            device = mesh.device
            # renderer for visualization
            if lights is None:
                lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
            vis_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=self.raster_settings_vis),
                shader=HardPhongShader(
                    device=device, 
                    cameras=cameras,
                    lights=lights))
            # render image:
            visualization = vis_renderer(mesh).permute(0, 3, 1, 2)[:, :3, :, :]
        return visualization


    def calculate_vertex_visibility(self, vertices, faces, focal_lengths, soft=False):
        tex = torch.ones_like(vertices) * self.mesh_color_0    # (1, V, 3)
        textures = Textures(verts_rgb=tex)
        mesh = Meshes(verts=vertices, faces=faces, textures=textures)
        cameras = self._get_cam(focal_lengths)
        # NEW: use the rasterizer to check vertex visibility
        #   see: https://github.com/facebookresearch/pytorch3d/issues/126
        # Get a rasterizer
        if soft:
            rasterizer = MeshRasterizer(cameras=cameras, 
                                        raster_settings=self.raster_settings_soft)
        else:
            rasterizer = MeshRasterizer(cameras=cameras, 
                            raster_settings=self.raster_settings_vis)
        # Get the output from rasterization
        fragments = rasterizer(mesh)
        # pix_to_face is of shape (N, H, W, 1)
        pix_to_face = fragments.pix_to_face  
        # (F, 3) where F is the total number of faces across all the meshes in the batch
        packed_faces = mesh.faces_packed() 
        # (V, 3) where V is the total number of verts across all the meshes in the batch
        packed_verts = mesh.verts_packed() 
        vertex_visibility_map = torch.zeros(packed_verts.shape[0])   # (V,)
        # Indices of unique visible faces
        visible_faces = pix_to_face.unique()    # [0]   # (num_visible_faces )
        # Get Indices of unique visible verts using the vertex indices in the faces
        visible_verts_idx = packed_faces[visible_faces]    # (num_visible_faces,  3)
        unique_visible_verts_idx = torch.unique(visible_verts_idx)   # (num_visible_verts, )
        # Update visibility indicator to 1 for all visible vertices 
        vertex_visibility_map[unique_visible_verts_idx] = 1.0
        # since all meshes have the same amount of vertices, we can reshape the result 
        bs = vertices.shape[0]
        vertex_visibility_map_resh = vertex_visibility_map.reshape((bs, -1))
        return pix_to_face, vertex_visibility_map_resh


    def get_torch_meshes(self, vertices, faces, color=0):
        # create pytorch mesh
        if color == 0:
            mesh_color = self.mesh_color_0
        else:
            mesh_color = self.mesh_color_1
        tex = torch.ones_like(vertices) * mesh_color    # (1, V, 3)
        textures = Textures(verts_rgb=tex)
        mesh = Meshes(verts=vertices, faces=faces, textures=textures)  
        return mesh


    def get_visualization_nograd(self, vertices, faces, focal_lengths, color=0):
        # vertices: torch.Size([bs, 3889, 3])
        # faces: torch.Size([bs, 7774, 3]), int
        # focal_lengths: torch.Size([bs, 1])
        device = vertices.device
        # create cameras
        cameras = self._get_cam(focal_lengths)
        # create pytorch mesh
        if color == 0:
            mesh_color = self.mesh_color_0      # blue
        elif color == 1:
            mesh_color = self.mesh_color_1      
        elif color ==  2:   
            MESH_COLOR_2 = [240, 250, 240]      # white
            mesh_color =  torch.FloatTensor(MESH_COLOR_2).to(device)
        elif color ==  3:   
            # MESH_COLOR_3 = [223, 0, 172]         # pink
            # MESH_COLOR_3 = [245, 245, 220]         # beige
            MESH_COLOR_3 = [166, 173, 164] 
            mesh_color =  torch.FloatTensor(MESH_COLOR_3).to(device)        
        else:
            MESH_COLOR_2 = [240, 250, 240]
            mesh_color =  torch.FloatTensor(MESH_COLOR_2).to(device)
        tex = torch.ones_like(vertices) * mesh_color    # (1, V, 3)
        textures = Textures(verts_rgb=tex)
        mesh = Meshes(verts=vertices, faces=faces, textures=textures)  
        # render mesh (no gradients)
        # lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
        # lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])
        lights = DirectionalLights(device=device, direction=[[0.0, -5.0, -10.0]])
        visualization = self._get_visualization_from_mesh(mesh, cameras, lights=lights)
        return visualization
              
    def project_points(self, points, focal_lengths=None, cameras=None):
        # points: torch.Size([bs, n_points, 3])
        # either focal_lengths or cameras is needed:
        #   focal_lenghts: torch.Size([bs, 1])
        #   cameras: pytorch camera, for example PerspectiveCameras()
        bs = points.shape[0]
        device = points.device
        screen_size = self.image_size.repeat((bs, 1))
        if cameras is None:
            cameras = self._get_cam(focal_lengths)
        if pytorch3d.__version__ == '0.2.5':
            proj_points_orig = cameras.transform_points_screen(points, screen_size)[:, :, [1, 0]]       # used in the original virtuel environment (for cvpr BARC submission) 
        elif pytorch3d.__version__ == '0.6.1':
            proj_points_orig = cameras.transform_points_screen(points)[:, :, [1, 0]]        
        else: 
            print('this part depends on the version of pytorch3d, code was developed with 0.2.5')
            raise ValueError
        # flip, otherwise the 1st and 2nd row are exchanged compared to the ground truth
        proj_points = torch.flip(proj_points_orig, [2])   
        # --- project points 'manually'
        # j_proj = project_points_p3d(image_size, focal_length, points, device)
        return proj_points      

    def forward(self, vertices, points, faces, focal_lengths, color=None):
        # vertices: torch.Size([bs, 3889, 3])
        # points: torch.Size([bs, n_points, 3]) (or None)
        # faces: torch.Size([bs, 7774, 3]), int
        # focal_lengths: torch.Size([bs, 1])
        # color: if None we don't render a visualization, else it should
        #   either be 0 or 1
        # ---> important: results are around 0.5 pixels off compared to chumpy!
        #   have a look at renderer.py for an explanation
        # create cameras
        cameras = self._get_cam(focal_lengths)
        # create pytorch mesh
        if color is None or color == 0:
            mesh_color = self.mesh_color_0
        else:
            mesh_color = self.mesh_color_1           
        tex = torch.ones_like(vertices) * mesh_color    # (1, V, 3)
        textures = Textures(verts_rgb=tex)
        mesh = Meshes(verts=vertices, faces=faces, textures=textures)
        # silhouette renderer 
        renderer_silh = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=self.raster_settings_soft),
            shader=SoftSilhouetteShader(blend_params=self.blend_params))
        # project silhouette
        silh_images = renderer_silh(mesh)[..., -1].unsqueeze(1)
        # project points
        if points is None:
            proj_points = None
        else:
            proj_points = self.project_points(points=points, cameras=cameras)
        if color is not None:
            # color renderer for visualization (no gradients)
            visualization = self._get_visualization_from_mesh(mesh, cameras)    
            return silh_images, proj_points, visualization
        else:
            return silh_images, proj_points



  
