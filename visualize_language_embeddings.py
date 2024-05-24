# Matt Strong 2024
# This script is used to visualize the embeddings of the language model with the requested words

import os

import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # get predicted images in specified path
    output_path = 'output/real-world/lab-scene/0/open_new_eval_softmax_s10.0_a05'
    predicted_images_path = 'pred_images'
    pred_seg_path = 'pred_segs'
    
    full_pred_images_path = os.path.join(output_path, predicted_images_path)
    full_pred_seg_path = os.path.join(output_path, pred_seg_path)
    
    all_predicted_seg_images = [f for f in os.listdir(full_pred_seg_path) if not os.path.isfile(os.path.join(full_pred_seg_path, f))]
    
    all_predicted_seg_images.sort()
    
    # get all predicted images
    all_predicted_images = [f for f in os.listdir(full_pred_images_path) if os.path.isfile(os.path.join(full_pred_images_path, f))]
    
    all_predicted_images.sort()
    
    for pred_image in all_predicted_images:
        # remove .png extension
        pred_image_name = pred_image.split('.')[0]
        
        # read image
        full_image_path = os.path.join(full_pred_images_path, pred_image)
        
        # read the image
        image = cv2.imread(full_image_path)
        
        # get stuff stored in pred_segs
        full_seg_path= os.path.join(full_pred_seg_path, pred_image_name)
        contents = os.listdir(full_seg_path)
        for item in contents:
            if item.endswith('.png'):
                # get the image, which is a mask
                # get object name
                object_name = item.split('.')[0]
                full_image_path = os.path.join(full_seg_path, item)
                
                # read the image
                mask = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
                
                color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

                alpha = 0.5  
                overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                plt.imshow(overlay_rgb)
                plt.title(f"{object_name} in scene", fontsize=20)
                plt.axis('off')

                plt.show()
        
        