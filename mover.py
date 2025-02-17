from pathlib import Path
import shutil

main_dir = Path("/home/lixion/rgbd/2011_09_26")
data = Path("/home/lixion/rgbd/combined")


for subdir in main_dir.iterdir():
    print(subdir)
    images = Path(subdir, "processed", "images")
    labels = Path(subdir, "processed", "labels")
    lidar = Path(subdir, "processed", "lidar")

    
    for im in images.iterdir():
        print(im)
        lid_og = Path(lidar, (im.with_suffix('.npy')).name)
        print(lid_og)
        la_og = Path(labels, (im.with_suffix('.txt')).name)

        im_dest = Path(data, "images", im.name)
        lid_dest = Path(data, "lidar", (im.with_suffix('.npy')).name)
        la_dest = Path(data, "labels", (im.with_suffix('.txt')).name)

        if im.exists() and lid_og.exists() and la_og.exists():
            shutil.copy(im, im_dest)
            shutil.copy(lid_og, lid_dest)
            shutil.copy(la_og, la_dest)

    