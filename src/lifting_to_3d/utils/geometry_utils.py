
import torch
from torch.nn import functional as F
import numpy as np
from torch import nn


def geodesic_loss(R, Rgt):
    # see: Silvia tiger pose model 3d code
    num_joints = R.shape[1]
    RT = R.permute(0,1,3,2)
    A = torch.matmul(RT.view(-1,3,3),Rgt.view(-1,3,3))
    # torch.trace works only for 2D tensors
    n = A.shape[0]
    po_loss =  0
    eps = 1e-7
    T = torch.sum(A[:,torch.eye(3).bool()],1)
    theta = torch.clamp(0.5*(T-1), -1+eps, 1-eps)
    angles =  torch.acos(theta)
    loss = torch.sum(angles)/(n*num_joints)
    return loss

class geodesic_loss_R(nn.Module):
    def __init__(self,reduction='mean'):
        super(geodesic_loss_R, self).__init__()
        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,bRgts,bRps):
        #return((bRgts - bRps)**2.).mean()
        return geodesic_loss(bRgts, bRps)

    def forward(self, ypred, ytrue):
        theta = geodesic_loss(ypred,ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        else:
            return theta

def batch_rodrigues_numpy(theta):
    """ Code adapted from spin
    Convert axis-angle representation to rotation matrix.
    Remark: 
        this leads to the same result as kornia.angle_axis_to_rotation_matrix(theta)
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = np.linalg.norm(theta + 1e-8, ord = 2, axis = 1)
    # angle = np.unsqueeze(l1norm, -1)
    angle = l1norm.reshape((-1, 1))
    # normalized = np.div(theta, angle)
    normalized = theta / angle
    angle = angle * 0.5
    v_cos = np.cos(angle)
    v_sin = np.sin(angle)
    # quat = np.cat([v_cos, v_sin * normalized], dim = 1)
    quat = np.concatenate([v_cos, v_sin * normalized], axis = 1)
    return quat_to_rotmat_numpy(quat)

def quat_to_rotmat_numpy(quat):
    """Code from: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    # norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    norm_quat = norm_quat/np.linalg.norm(norm_quat, ord=2, axis=1, keepdims=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]
    B = quat.shape[0]
    # w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    w2, x2, y2, z2 = w**2, x**2, y**2, z**2
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    rotMat = np.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], axis=1).reshape(B, 3, 3)
    return rotMat  


def batch_rodrigues(theta):
    """Code from: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    Convert axis-angle representation to rotation matrix.
    Remark: 
        this leads to the same result as kornia.angle_axis_to_rotation_matrix(theta)
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def batch_rot2aa(Rs, epsilon=1e-7):
    """ Code from: https://github.com/vchoutas/expose/blob/dffc38d62ad3817481d15fe509a93c2bb606cb8b/expose/utils/rotation_utils.py#L55 
    Rs is B x 3 x 3
    void cMathUtil::RotMatToAxisAngle(const tMatrix& mat, tVector& out_axis,
                                      double& out_theta)
    {
        double c = 0.5 * (mat(0, 0) + mat(1, 1) + mat(2, 2) - 1);
        c = cMathUtil::Clamp(c, -1.0, 1.0);
        out_theta = std::acos(c);
        if (std::abs(out_theta) < 0.00001)
        {
            out_axis = tVector(0, 0, 1, 0);
        }
        else
        {
            double m21 = mat(2, 1) - mat(1, 2);
            double m02 = mat(0, 2) - mat(2, 0);
            double m10 = mat(1, 0) - mat(0, 1);
            double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
            out_axis[0] = m21 / denom;
            out_axis[1] = m02 / denom;
            out_axis[2] = m10 / denom;
            out_axis[3] = 0;
        }
    }
    """
    cos = 0.5 * (torch.einsum('bii->b', [Rs]) - 1)
    cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)
    theta = torch.acos(cos)
    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10 + epsilon)
    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)
    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)

def quat_to_rotmat(quat):
    """Code from: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat   

def rot6d_to_rotmat(rot6d):
    """ Code from: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    rot6d = rot6d.view(-1,3,2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rotmat = torch.stack((b1, b2, b3), dim=-1)
    return rotmat

def rotmat_to_rot6d(rotmat):
    """ Convert 3x3 rotation matrix to 6D rotation representation.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,3,3) Batch of corresponding rotation matrices
    Output:
        (B,6) Batch of 6-D rotation representations
    """ 
    rot6d = rotmat[:, :, :2].reshape((-1, 6))
    return rot6d


def main():
    # rotation matrix and 6d representation
    # see "On the Continuity of Rotation Representations in Neural Networks"
    from pyquaternion import Quaternion
    batch_size = 5
    rotmat = np.zeros((batch_size, 3, 3))
    for ind in range(0, batch_size):
        rotmat[ind, :, :] = Quaternion.random().rotation_matrix
    rotmat_torch = torch.Tensor(rotmat)
    rot6d = rotmat_to_rot6d(rotmat_torch)
    rotmat_rec = rot6d_to_rotmat(rot6d)
    print('..................... 1 ....................')
    print(rotmat_torch[0, :, :])
    print(rotmat_rec[0, :, :])
    print('Conversion from rotmat to rot6d and inverse are ok!')
    # rotation matrix and axis angle representation
    import kornia
    input = torch.rand(1, 3)
    output = kornia.angle_axis_to_rotation_matrix(input)
    input_rec = kornia.rotation_matrix_to_angle_axis(output)
    print('..................... 2 ....................')
    print(input)
    print(input_rec)
    print('Kornia implementation for rotation_matrix_to_angle_axis is wrong!!!!')
    # For non-differential conversions use scipy:
    #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(rotmat[0, :, :])
    print('..................... 3 ....................')
    print(r.as_matrix())
    print(r.as_rotvec())
    print(r.as_quaternion)
    # one might furthermore have a look at:
    #   https://github.com/silviazuffi/smalst/blob/master/utils/transformations.py



if __name__ == "__main__":
    main()


