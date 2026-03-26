import torch

dummy = torch.randn(1, 2, 16, 13, 1)
vis = "left"

def form_analysis(exercise, phase, joints, vis):
    if exercise == 'pushup':
        deep_enough = 0
        if phase == "down":
            if vis == "left":
                bottom = torch.argmax(joints[0, 1, :, 1, 0])
                shoulder_pos = joints[0, 1, :, 1, 0]
                elbow_pos = joints[0, 1, :, 3, 0]
                print("shoulder:", shoulder_pos)
                print("elbow:", elbow_pos)
            else:
                bottom = torch.argmax(joints[0, 1, :, 2, 0])
                shoulder_pos = joints[0, 1, :, 2, 0]
                elbow_pos = joints[0, 1, :, 4, 0]
            bottom_tensor = torch.tensor([-2, -1, 0, 1, 2], device=joints.device) + bottom
            if bottom_tensor[0] == -2:
                bottom_tensor = bottom_tensor + 2
            elif bottom_tensor[0] == -1:
                bottom_tensor = bottom_tensor + 1 
            elif bottom_tensor[4] == 17:
                bottom_tensor = bottom_tensor - 2
            elif bottom_tensor[4] == 16:
                bottom_tensor = bottom_tensor - 1
            #print(bottom_tensor)
            shoulder = 0
            elbow = 0
            for i in bottom_tensor:
                shoulder += shoulder_pos[i]
                elbow += elbow_pos[i]
            print("shoulder2:", shoulder)
            print("elbow2:", elbow)
            shoulder_mean = shoulder/5
            elbow_mean = elbow/5
            if shoulder_mean * 1 > elbow_mean:
                deep_enough = 1
        if vis == "left":
            shoulder_pos_x = torch.sum(joints[0, 0, :, 1, 0][-5:])/5
            hip_pos_x = torch.sum(joints[0, 0, :, 7, 0][-5:])/5
            knee_pos_x = torch.sum(joints[0, 0, :, 9, 0][-5:])/5
            shoulder_pos_y = torch.sum(joints[0, 1, :, 1, 0][-5:])/5
            hip_pos_y = torch.sum(joints[0, 1, :, 7, 0][-5:])/5
            knee_pos_y = torch.sum(joints[0, 1, :, 9, 0][-5:])/5
        else:
            shoulder_pos_x = torch.sum(joints[0, 0, :, 2, 0][-5:])/5
            hip_pos_x = torch.sum(joints[0, 0, :, 8, 0][-5:])/5
            knee_pos_x = torch.sum(joints[0, 0, :, 10, 0][-5:])/5
            shoulder_pos_y = torch.sum(joints[0, 1, :, 2, 0][-5:])/5
            hip_pos_y = torch.sum(joints[0, 1, :, 8, 0][-5:])/5
            knee_pos_y = torch.sum(joints[0, 1, :, 10, 0][-5:])/5
        
        dist_num = ((knee_pos_y - shoulder_pos_y) * hip_pos_x - (knee_pos_x - shoulder_pos_x) * hip_pos_y + knee_pos_x * shoulder_pos_y - knee_pos_y * shoulder_pos_x) 
        dist_den = ((knee_pos_y - shoulder_pos_y)**2 + (knee_pos_x - shoulder_pos_x)**2)**0.5
        dist = dist_num/dist_den
        if dist > 0.06:
            back_form = 1
        elif dist < -0.06:
            back_form = -1
        else:
            back_form = 0
        return deep_enough, back_form
    if exercise == 'squat':
        deep_enough = 0
        if phase == "down":
            if vis == "left":
                bottom = torch.argmax(joints[0, 1, :, 7, 0])
                hip_pos = joints[0, 1, :, 7, 0]
                knee_pos = joints[0, 1, :, 9, 0]
            else:
                bottom = torch.argmax(joints[0, 1, :, 8, 0])
                hip_pos = joints[0, 1, :, 8, 0]
                knee_pos = joints[0, 1, :, 10, 0]
            bottom_tensor = torch.tensor([-2, -1, 0, 1, 2], device=joints.device) + bottom
            if bottom_tensor[0] == -2:
                bottom_tensor = bottom_tensor + 2
            elif bottom_tensor[0] == -1:
                bottom_tensor = bottom_tensor + 1 
            elif bottom_tensor[4] == 17:
                bottom_tensor = bottom_tensor - 2
            elif bottom_tensor[4] == 16:
                bottom_tensor = bottom_tensor - 1
            #print(bottom_tensor)
            hip = 0
            knee = 0
            for i in bottom_tensor:
                hip += hip_pos[i]
                knee += knee_pos[i]
            hip_mean = hip/5
            knee_mean = knee/5
            if hip_mean * 1 > knee_mean:
                deep_enough = 1
        if vis == "left":
            shoulder_pos_x = torch.sum(joints[0, 0, :, 1, 0][-5:])/5
            ankle_pos_x = torch.sum(joints[0, 0, :, 11, 0][-5:])/5
            hip_pos_x = torch.sum(joints[0, 0, :, 7, 0][-5:])/5
            shoulder_pos_y = torch.sum(joints[0, 1, :, 1, 0][-5:])/5
            hip_pos_y = torch.sum(joints[0, 1, :, 7, 0][-5:])/5
        else:
            shoulder_pos_x = torch.sum(joints[0, 0, :, 2, 0][-5:])/5
            ankle_pos_x = torch.sum(joints[0, 0, :, 12, 0][-5:])/5
            hip_pos_x = torch.sum(joints[0, 0, :, 8, 0][-5:])/5
            shoulder_pos_y = torch.sum(joints[0, 1, :, 2, 0][-5:])/5
            hip_pos_y = torch.sum(joints[0, 1, :, 8, 0][-5:])/5
        dist_ankle_shoulder = abs(shoulder_pos_x - ankle_pos_x) / ((hip_pos_y - shoulder_pos_y)**2 + (hip_pos_x - shoulder_pos_x)**2)**0.5
        if dist_ankle_shoulder > 0.8:
            forward_bend = 1
        else:
            forward_bend = 0 
        return deep_enough, forward_bend
    return 0
                
form_analysis('pushup', "down", dummy, vis)    