

import torch

#https://github.com/foamliu/TwinsOrNot 

# model params
threshold = 73.18799151798612
mu_0 = 89.6058
sigma_0 = 4.5451
mu_1 = 43.5357
sigma_1 = 8.83

def compare(fn_0, fn_1):
    print('fn_0: ' + fn_0)
    print('fn_1: ' + fn_1)
    img0 = get_image(fn_0)
    img1 = get_image(fn_1)
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float)
    imgs[0] = img0
    imgs[1] = img1

    with torch.no_grad():
        output = model(imgs)

        feature0 = output[0].cpu().numpy()
        feature1 = output[1].cpu().numpy()
        x0 = feature0 / np.linalg.norm(feature0)
        x1 = feature1 / np.linalg.norm(feature1)
        cosine = np.dot(x0, x1)
        theta = math.acos(cosine)
        theta = theta * 180 / math.pi

    print('theta: ' + str(theta))
    prob = get_prob(theta)
    print('prob: ' + str(prob))
    return prob, theta < threshold


def get_prob(theta):
    prob_0 = norm.pdf(theta, mu_0, sigma_0)
    prob_1 = norm.pdf(theta, mu_1, sigma_1)
    total = prob_0 + prob_1
    return prob_1 / total

def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)
        
        
def resize(filename):
    img = cv.imread(filename)
    h, w = img.shape[:2]
    ratio_w = w / 1280
    ratio_h = h / 720
    if ratio_w > 1 or ratio_h > 1:
        ratio = max(ratio_w, ratio_h)
        new_w = int(w / ratio)
        new_h = int(h / ratio)
        img = cv.resize(img, (new_w, new_h))
        cv.imwrite(filename, img)
        
