from pathlib import Path
import shutil

main_dir = Path("/home/lixion/rgbd/2011_09_26")
data = Path("/home/lixion/rgbd/combined")

data.mkdir(parents=True, exist_ok=True)

for subdir in main_dir.iterdir():
    print(subdir)
    images = Path(subdir, "processed", "images")
    labels = Path(subdir, "processed", "labels")
    lidar = Path(subdir, "processed", "lidar")
    
    (Path(data, "images")).mkdir(parents=True, exist_ok=True)
    (Path(data, "labels")).mkdir(parents=True, exist_ok=True)
    (Path(data, "lidar")).mkdir(parents=True, exist_ok=True)
    
    for im in images.iterdir():
        lid_og = Path(lidar, (im.with_suffix('.npy')).name)
        la_og = Path(labels, (im.with_suffix('.txt')).name)

        im_dest = Path(data, "images", im.name)
        lid_dest = Path(data, "lidar", (im.with_suffix('.npy')).name)
        la_dest = Path(data, "labels", (im.with_suffix('.txt')).name)

        if im.exists() and lid_og.exists() and la_og.exists():
            shutil.copy(im, im_dest)
            shutil.copy(lid_og, lid_dest)
            shutil.copy(la_og, la_dest)

    