import os
import argparse
import logging
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from custom_sam3 import SAM3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_prompt_for_class(class_name):
    class_name = class_name.lower().strip()
    
    prompts = {
        "virus": [
            "crinkled leaf surface",   
            "mosaic pattern",         
            "curled leaf",              
            "mottled leaf",             
            "necrotic spot",            
            "stunted leaf",             
            "rugose leaf surface"       
        ],
        
        "phytopthora": [
            "dark brown lesion",        
            "black lesion",             
            "circular necrotic patch",  
            "irregular dark spot",      
            "wet spot"                  
        ],
        
        "nematode": [
            "yellowish leaf patch",   
            "yellowing leaf",         
            "discolored leaf area",   
        ],
        
        "fungi": [
            "circular spot",                     
            "concentric ring spot",              
            "sunken spot with yellow border",    
            "light brown spot",                  
            "powdery patch"                      
        ],
        
        "bacteria": [   
            "wilted leaf",             
            "drooping leaf",           
            "shriveled leaf",          
            "soft rot spot"            
        ],
        
        "pest": [
            "coiled leaf tip",       
            "leaf hole",             
            "mined leaf route",      
            "silver dots",           
            "distorted leaf tissue", 
            "chewed leaf edge"       
        ],

        "healthy": ["uniform green potato leaf"],
    }
    
    return prompts.get(class_name, ["disease lesions on potato leaf"])

def process_dataset(source_dir, target_dir, device='cuda'):
    if not os.path.exists(source_dir):
        logging.error(f"Source directory not found: {source_dir}")
        return

    try:
        sam3 = SAM3(device=device)
    except Exception as e:
        logging.error(f"Failed to initialize SAM3: {e}")
        return
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    classes = os.listdir(source_dir)
    total_processed = 0
    total_skipped = 0
    
    for class_name in classes:
        class_src_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_src_path):
            continue
        
        # Create target class directory
        class_tgt_path = os.path.join(target_dir, class_name)
        os.makedirs(class_tgt_path, exist_ok=True)
        
        specific_prompt = get_prompt_for_class(class_name)
        logging.info(f"Processing Class: {class_name} | Prompt: {specific_prompt}")
        
        image_files = os.listdir(class_src_path)
        n_skipped = 0
        
        for img_name in tqdm(image_files, desc=f"Class {class_name}"):
            img_path = os.path.join(class_src_path, img_name)
            save_path = os.path.join(class_tgt_path, img_name)
            
            if os.path.exists(save_path): 
                continue

            try:
                # 1. Load image
                image_pil = Image.open(img_path).convert("RGB")

                # 2. Remove Background using generic prompt (Layer 1)
                leaf_mask = sam3.remove_background(image_pil, text_prompt="potato leaf")
                
                # 3. Disease prompt (Layer 2)
                # Note: Layer 2 (Focus in Disease) is optional/commented out in notebook 
                # and can be added here if needed using `sam3.process_mask(image_pil, specific_prompt)`
                
                # 4. Apply mask to image
                img_tensor = transforms.ToTensor()(image_pil).to(sam3.device)
                focused_tensor = sam3.apply_mask_to_image(img_tensor, leaf_mask)
                
                # Convert back to numpy/PIL and save
                focused_img_np = (focused_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(focused_img_np).save(save_path)
                
                total_processed += 1
                
            except Exception as e:
                n_skipped += 1
                logging.warning(f"Error processing {img_name}: {e}")
                
        total_skipped += n_skipped
        logging.info(f"Class '{class_name}' done. Skipped: {n_skipped}")

    logging.info("=== PROCESSING COMPLETED ===")
    logging.info(f"Total processed: {total_processed}")
    logging.info(f"Total skipped: {total_skipped}")

def main():
    parser = argparse.ArgumentParser(description="SAM3 Data Preprocessing Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--target", type=str, required=True, help="Path to save processed dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run SAM3 (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, switching to CPU")
        args.device = 'cpu'

    process_dataset(args.source, args.target, args.device)

if __name__ == "__main__":
    main()
