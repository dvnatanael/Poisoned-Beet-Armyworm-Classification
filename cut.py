import matplotlib.pyplot as plt

W = 4864
H = 3648

L = 1300

def cut(image):
    rs = [500, H//2]
    cs = [500, W//3+250, W//3*2]

    cut_images = []
    for r in rs:
        for c in cs:
            cut_images.append(image[r:r+L, c:c+L])
            
    return cut_images

def show(cut_images):
    fig, axs = plt.subplots(2, 3)
    for a in axs.ravel():
        a.axis('off')
    idx = 0
    for i in range(2):
        for j in range(3):
             axs[i, j].imshow(cut_images[idx])
             idx += 1
    plt.tight_layout()
    plt.show()

def main():
    image = plt.imread("AI_Bug_0524/P1030042.jpg")
    plt.imshow(image)
    cut_images = cut(image)
    show(cut_images)

if __name__ == '__main__':
    main()
    