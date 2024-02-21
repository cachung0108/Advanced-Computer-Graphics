import os
import cv2
import numpy as np

def main():

    source_path = "source"
    target_path = "target"
    
    sourcefiles = os.listdir(source_path)
    targetfiles = os.listdir(target_path)

    for file in sourcefiles:

        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):

            prefix = file[:2]
            image_path = os.path.join(source_path, file)
            
            source_rgb_values = cv2.imread(image_path)
            source_blue_channel = source_rgb_values[:, :, 0]
            source_green_channel = source_rgb_values[:, :, 1]
            source_red_channel = source_rgb_values[:, :, 2]
            source_blue_mean = np.mean(source_blue_channel)
            source_green_mean = np.mean(source_green_channel)
            source_red_mean = np.mean(source_red_channel)

            source_blue_std_dev = np.std(source_blue_channel)
            source_green_std_dev = np.std(source_green_channel)
            source_red_std_dev = np.std(source_red_channel)

            for file in targetfiles:

                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):

                    if file[:2] == prefix:

                        image_path = os.path.join(target_path, file)
                        
                        target_rgb_values = cv2.imread(image_path)
                        target_blue_channel = target_rgb_values[:, :, 0]
                        target_green_channel = target_rgb_values[:, :, 1]
                        target_red_channel = target_rgb_values[:, :, 2]
                        target_blue_mean = np.mean(target_blue_channel)
                        target_green_mean = np.mean(target_green_channel)
                        target_red_mean = np.mean(target_red_channel)

                        target_blue_std_dev = np.std(target_blue_channel)
                        target_green_std_dev = np.std(target_green_channel)
                        target_red_std_dev = np.std(target_red_channel)

                        blue_weight = 0.49
                        green_weight = 0.54
                        red_weight = 0.62
                        
                        result_blue_channel = ((blue_weight * target_blue_std_dev + (1 - blue_weight) * source_blue_std_dev)/source_blue_std_dev) * (source_blue_channel - source_blue_mean) + blue_weight * target_blue_mean + (1 - blue_weight) * source_blue_mean
                        result_green_channel = ((green_weight * target_green_std_dev + (1 - green_weight) * source_green_std_dev)/source_green_std_dev) * (source_green_channel - source_green_mean) + green_weight * target_green_mean + (1 - green_weight) * source_green_mean
                        result_red_channel = ((red_weight * target_red_std_dev + (1 - red_weight) * source_red_std_dev)/source_red_std_dev) * (source_red_channel - source_red_mean) + red_weight * target_red_mean + (1 - red_weight) * source_red_mean

                        result_rgb_values = np.stack((result_blue_channel, result_green_channel, result_red_channel), axis=-1)
                        result_rgb_values = np.clip(result_rgb_values, 0, 255)
                        result_rgb_values = np.floor(result_rgb_values)

                        output_image_path = "result/"+ prefix +"_output_image.jpg"
                        cv2.imwrite(output_image_path, result_rgb_values)

if __name__ == "__main__":
    main()