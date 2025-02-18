from pathlib import Path
import shutil

main_dir = Path("/home/lixion/rgbd/2011_09_26")
data = Path("/home/lixion/rgbd/combined")

data.mkdir(parents=True, exist_ok=True)


for subdir in main_dir.iterdir():
    print(f"Processing {subdir}")
    
    # Construct paths to images, labels, and lidar directories
    images = Path(subdir, "processed", "images")
    labels = Path(subdir, "processed", "labels")
    lidar = Path(subdir, "processed", "lidar")

    # Ensure that the paths are valid directories
    if not images.is_dir():
        print(f"Warning: {images} is not a valid directory. Skipping.")
        continue
    if not labels.is_dir():
        print(f"Warning: {labels} is not a valid directory. Skipping.")
        continue
    if not lidar.is_dir():
        print(f"Warning: {lidar} is not a valid directory. Skipping.")
        continue
    
    # Proceed with file operations
    for im in images.iterdir():
        try:
            # Get lidar and label paths corresponding to the image
            lid_og = Path(lidar, im.stem + '.npy')  # Use im.stem to remove the extension
            la_og = Path(labels, im.stem + '.txt')  # Use im.stem to remove the extension

            # Destination paths for image, lidar, and label
            im_dest = Path(data, "images", im.name)
            lid_dest = Path(data, "lidar", lid_og.name)
            la_dest = Path(data, "labels", la_og.name)

            # Ensure source and destination exist before copying
            if im.exists() and lid_og.exists() and la_og.exists():
                # Create the destination directories if they do not exist
                im_dest.parent.mkdir(parents=True, exist_ok=True)
                lid_dest.parent.mkdir(parents=True, exist_ok=True)
                la_dest.parent.mkdir(parents=True, exist_ok=True)

                # Copy files
                shutil.copy(im, im_dest)
                shutil.copy(lid_og, lid_dest)
                shutil.copy(la_og, la_dest)
        
        except Exception as e:
            print(f"Error processing {im.name}: {e}")
            continue


    

    