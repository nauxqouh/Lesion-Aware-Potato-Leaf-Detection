import torch
import os
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
import torchvision
from torchvision import transforms
import logging

class SAM3:
    def __init__(self, device='cuda'):
        self.device = device
        logging.info(f"Loading SAM 3 ...")
        self.model = build_sam3_image_model(device=device)
        self.model.eval()
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
        logging.info("SAM 3 model loaded.")
        
    def segment_by_text_prompts(self, image_input, text_prompts):
        """Segment focused target area based on text prompt"""
        
        inference_state = self.processor.set_image(image_input)
        
        accumulated_masks = []
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        for prompt in text_prompts:
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            if masks.numel() > 0:
                masks = masks.to(self.device)
                if masks.dim() == 4:
                    masks = masks.squeeze(1) # [N, H, W]
                current_prompt_mask, _ = torch.max(masks, dim=0)
                accumulated_masks.append(current_prompt_mask)
            else:
                logging.warning(f"Found 0 object from prompt: {prompt}!")
                
        if len(accumulated_masks) > 0:
            all_masks_tensor = torch.stack(accumulated_masks)
            final_mask, _ = torch.max(all_masks_tensor, dim=0) # [H, W]
            return final_mask
        else:
            logging.warning(f"FOUND NOTHING HERE.")
            return None
        
    def process_mask(self, image_input, text_prompts):
        masks = self.segment_by_text_prompts(image_input, text_prompts)
        if masks is None or masks.sum() == 0:
            sim_prompt = ["disease lesions on potato leaf"]
            masks = self.segment_by_text_prompts(image_input, sim_prompt)
            # Fallback
            if masks is None or masks.sum() == 0:
                w, h = image_input.size
                return torch.ones((1, h, w), device=self.device)
                
        if masks.dim() == 2:
            final_masks = masks.unsqueeze(0)
        final_masks = (masks > 0.0).float()
        
        return final_masks

    def remove_background(self, image_input, text_prompt):
        w, h = image_input.size
        
        inference_state = self.processor.set_image(image_input)
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

        if masks.numel() > 0:
            masks = masks.to(self.device)
            if masks.dim() == 4:
                masks = masks.squeeze(1) # [N, H, W]
            mask, _ = torch.max(masks, dim=0) 
            return mask.unsqueeze(0) # [1, H, W]
        else:
            logging.warning(f"Found 0 object from prompt: {text_prompt}!")
            return torch.ones((1, h, w), device=self.device)
        
    def apply_mask_to_image(self, image_tensor, mask):
        if mask.shape[-2:] != image_tensor.shape[-2:]:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0), 
                size=image_tensor.shape[-2:], 
                mode='nearest'
            ).squeeze(0)
        focused_image = image_tensor * mask
        
        return focused_image
