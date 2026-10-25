import os
import torchvision.datasets as datasets

def extract_stl10_images(output_dir='./data/images'):
    print("Downloading/Loading STL10 dataset...")
    # We use the same split 'train' as in the notebook
    dataset = datasets.STL10(root='./data', split='train', download=True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Extracting {len(dataset)} images to {output_dir}...")
    
    for i in range(len(dataset)):
        image, label = dataset[i]
        # Save as JPEG
        image_path = os.path.join(output_dir, f"Image_{i}.jpg")
        image.save(image_path)
        
        if i % 500 == 0:
            print(f"Extracted {i} images...")
        
    print("Extraction complete.")

if __name__ == "__main__":
    extract_stl10_images()
