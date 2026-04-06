import torch
import math

# ----- Helper functions -------

# Get a window of frame indices around the bottom position of the rep
def get_bottom_window(bottom_idx, num_frames, window_size=5, device=None):
    half = window_size // 2
    offsets = torch.arange(-half, half + 1, device=device)
    indices = offsets + bottom_idx
    # Shift window to stay within bounds
    if indices[0] < 0:
        indices = indices - indices[0]
    elif indices[-1] >= num_frames:
        indices = indices - (indices[-1] - num_frames + 1)
    return indices

# compute mean of values at given indices
def get_mean(joints, frames):
    return torch.stack([joints[f] for f in frames]).mean()

# compute angle at p2 formed by p1-p2-p3 in degrees
def angle_from_positions(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-6)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    return torch.acos(cos_angle) * 180.0 / math.pi

# Get the (x, y) position of a specific joint at a specific frame
def joint_pos(joints, frame, joint_idx):
    return torch.stack([
        joints[0, 0, frame, joint_idx, 0],
        joints[0, 1, frame, joint_idx, 0]
    ])

# Get the mean (x, y) position of a joint over multiple frames with gaussian weighted mean
def mean_joint_pos(joints, joint_idx, frame_indices):
    x = get_mean(joints[0, 0, :, joint_idx, 0], frame_indices)
    y = get_mean(joints[0, 1, :, joint_idx, 0], frame_indices)
    return x, y


# Joint index maps (Penn Action Dataset)
# 0: head, 1: L_shoulder, 2: R_shoulder, 3: L_elbow, 4: R_elbow,
# 5: L_wrist, 6: R_wrist, 7: L_hip, 8: R_hip, 9: L_knee, 10: R_knee,
# 11: L_ankle, 12: R_ankle

JOINT_MAP = {
    "left": {
        "shoulder": 1, "elbow": 3, "wrist": 5,
        "hip": 7, "knee": 9, "ankle": 11,
    },
    "right": {
        "shoulder": 2, "elbow": 4, "wrist": 6,
        "hip": 8, "knee": 10, "ankle": 12,
    },
}

# ----- Main functions -------

# analyse form for pushups and squats based on joint positions and rep phase
def form_analysis(exercise, phase, joints, vis):
    if exercise == "pushup":
        return analyse_pushup(phase, joints, vis)
    elif exercise == "squat":
        return analyse_squat(phase, joints, vis)
    else:
        return 0, 0, {}


def analyse_pushup(phase, joints, vis):
    num_frames = joints.shape[2]
    device = joints.device
    jm = JOINT_MAP[vis] # get joints on more visible side

    feedback = {}

    # 1. Depth check
    deep_enough = 0
    depth_angle = None

    if phase == "down":
        # Find bottom by smallest knee angle (deepest point)
        angles_all = []
        for f in range(num_frames):
            shoulder = joint_pos(joints, f, jm["shoulder"])
            elbow = joint_pos(joints, f, jm["elbow"])
            wrist = joint_pos(joints, f, jm["wrist"])
            angles_all.append(angle_from_positions(shoulder, elbow, wrist))
        
        angles_all_t = torch.stack(angles_all)
        bottom = torch.argmin(angles_all_t).item()
        window = get_bottom_window(bottom, num_frames, device=device)
        
        depth_angle = torch.stack([angles_all_t[i] for i in window]).mean()

        # Elbow angle < 90 degrees at bottom then its deep enough
        if depth_angle < 90.0:
            deep_enough = 1

        feedback["bottom_elbow_angle"] = depth_angle.item()

    # 2. Back straightness
    last_frames = get_bottom_window(num_frames - 3, num_frames, device=device)

    sh_x, sh_y = mean_joint_pos(joints, jm["shoulder"], last_frames)
    hp_x, hp_y = mean_joint_pos(joints, jm["hip"], last_frames)
    kn_x, kn_y = mean_joint_pos(joints, jm["knee"], last_frames)

    # Perpendicular distance of hip from shoulder-knee line
    dist_num = ((kn_y - sh_y) * hp_x - (kn_x - sh_x) * hp_y + kn_x * sh_y - kn_y * sh_x)
    dist_den = ((kn_y - sh_y) ** 2 + (kn_x - sh_x) ** 2) ** 0.5 + 1e-8
    hip_deviation = dist_num / dist_den

    # Use angle-based threshold: compute shoulder-hip-knee angle
    sh_pt = torch.stack([sh_x, sh_y])
    hp_pt = torch.stack([hp_x, hp_y])
    kn_pt = torch.stack([kn_x, kn_y])
    body_angle = angle_from_positions(sh_pt, hp_pt, kn_pt)

    if body_angle < 160:
        # Hip deviates significantly from the straight line
        if hip_deviation > 0.04:
            back_form = 1   # rounded back
        elif hip_deviation < -0.04:
            back_form = -1  # hollow back
        else:
            back_form = 0
    else:
        back_form = 0  # body is straight enough

    feedback["hip_deviation"] = hip_deviation.item()
    feedback["body_angle"] = body_angle.item()

    return deep_enough, back_form, feedback


def analyse_squat(phase, joints, vis):
    num_frames = joints.shape[2]
    device = joints.device
    jm = JOINT_MAP[vis] # get joints on more visible side

    feedback = {}

    # 1. Depth check
    deep_enough = 0
    depth_angle = None

    if phase == "down":
        # Find bottom by smallest knee angle (deepest point)
        angles_all = []
        for f in range(num_frames):
            hip = joint_pos(joints, f, jm["hip"])
            knee = joint_pos(joints, f, jm["knee"])
            ankle = joint_pos(joints, f, jm["ankle"])
            angles_all.append(angle_from_positions(hip, knee, ankle))
        
        angles_all_t = torch.stack(angles_all)
        bottom = torch.argmin(angles_all_t).item()
        window = get_bottom_window(bottom, num_frames, device=device)
        
        depth_angle = torch.stack([angles_all_t[i] for i in window]).mean()

        # Knee angle < 90 degrees at bottom then its deep enough
        if depth_angle < 90.0:
            deep_enough = 1

        feedback["bottom_knee_angle"] = depth_angle.item()

    # 2. Forward lean
    last_frames = get_bottom_window(num_frames - 3, num_frames, device=device)

    sh_x, sh_y = mean_joint_pos(joints, jm["shoulder"], last_frames)
    hp_x, hp_y = mean_joint_pos(joints, jm["hip"], last_frames)
    ak_x, ak_y = mean_joint_pos(joints, jm["ankle"], last_frames)

    # Torso angle: angle between vertical and shoulder-hip line
    # Compute shoulder-hip vector angle relative to vertical (0, -1)
    torso_dx = sh_x - hp_x
    torso_dy = sh_y - hp_y
    torso_len = (torso_dx ** 2 + torso_dy ** 2) ** 0.5 + 1e-8
    # Vertical is (0, -1) in image coords (y increases downward)
    cos_lean = (-torso_dy) / torso_len  # dot with (0, -1)
    lean_angle = torch.acos(torch.clamp(cos_lean, -1.0, 1.0)) * 180.0 / math.pi

    if lean_angle > 45:
        forward_lean = 1  # excessive
    else:
        forward_lean = 0

    feedback["torso_lean_angle"] = lean_angle.item()

    # 3. Knee valgus
    l_knee_x = get_mean(joints[0, 0, :, 9, 0], last_frames)
    r_knee_x = get_mean(joints[0, 0, :, 10, 0], last_frames)
    l_hip_x = get_mean(joints[0, 0, :, 7, 0], last_frames)
    r_hip_x = get_mean(joints[0, 0, :, 8, 0], last_frames)

    hip_width = abs(l_hip_x - r_hip_x)
    knee_width = abs(l_knee_x - r_knee_x)

    if hip_width > 1e-4:
        valgus_ratio = knee_width / hip_width
        if valgus_ratio < 0.6:
            feedback["knee_valgus"] = "severe"
        elif valgus_ratio < 0.8:
            feedback["knee_valgus"] = "mild"
        else:
            feedback["knee_valgus"] = "good"
    else:
        feedback["knee_valgus"] = "unknown"  # pure side view

    return deep_enough, forward_lean, feedback