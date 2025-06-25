import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM 



def draw_bbox(image, detection_results):
    """
    Draw bounding boxes on an image.
    
    Args:
        image: PIL Image object
        detection_results: Dictionary with 'bboxes' and 'labels' keys
    """
    draw = ImageDraw.Draw(image)
    bboxes = detection_results.get('bboxes', [])
    labels = detection_results.get('labels', [])
    
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # Draw label
        draw.text((x1, y1 - 10), label, fill="red")
    
    return image

def draw_polygons(image, segmentation_results, fill_mask=False):
    """
    Draw polygons on an image.
    
    Args:
        image: PIL Image object
        segmentation_results: Dictionary with 'polygons' and 'labels' keys
        fill_mask: Whether to fill the polygons or just draw outlines
    """
    draw = ImageDraw.Draw(image)
    polygons = segmentation_results.get('polygons', [])
    labels = segmentation_results.get('polygons_labels', [])
    
    for polygon, label in zip(polygons, labels):
        # Flatten polygon coordinates
        coords = []
        for point in polygon:
            coords.extend(point)
        
        if fill_mask:
            draw.polygon(coords, fill=(255, 0, 0, 128), outline="red", width=2)
        else:
            draw.polygon(coords, outline="red", width=2)
        
        # Draw label at first point
        if polygon:
            draw.text((polygon[0][0], polygon[0][1] - 10), label, fill="red")
    
    return image

class Florence2:
    def __init__(self, gpu: bool = True):
        self.device = "cuda:0" if torch.cuda.is_available() and gpu else "mps" if torch.backends.mps.is_available() and gpu else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    def text_segmentation(self, prompt: str, image: Image, draw: bool = False):
        prompt = f"<REFERRING_EXPRESSION_SEGMENTATION>{prompt}"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(generated_text, task="<REFERRING_EXPRESSION_SEGMENTATION>", image_size=(image.width, image.height))

        if draw:
            draw_polygons(image, parsed_answer['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True)

        return parsed_answer
    
    def text_bbox(self, prompt: str, image: Image, draw: bool = False):
        def convert_to_od_format(data):  
              """  
              Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.  
            
              Parameters:  
              - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.  
            
              Returns:  
              - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.  
              """  
              # Extract bounding boxes and labels  
              bboxes = data.get('bboxes', [])  
              labels = data.get('bboxes_labels', [])  
                
              # Construct the output format  
              od_results = {  
                  'bboxes': bboxes,  
                  'labels': labels  
              }  
                
              return od_results
      
        prompt = f"<OPEN_VOCABULARY_DETECTION>{prompt}"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.processor.post_process_generation(generated_text, task="<OPEN_VOCABULARY_DETECTION>", image_size=(image.width, image.height))

        if draw:
            draw_bbox(image, convert_to_od_format(parsed_answer['<OPEN_VOCABULARY_DETECTION>']))

        return parsed_answer
    

