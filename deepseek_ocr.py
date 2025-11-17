import torch
from transformers import AutoModel, AutoTokenizer
import os
import glob
import re
from io import StringIO
import sys

def extract_text_from_image(
    image_path, 
    output_path="./output",
    model=None,
    tokenizer=None,
    prompt_type='free',  # 'free' or 'markdown'
    base_size=1024,
    image_size=640,
    crop_mode=True
):
    """
    Extract text from an image using DeepSeek-OCR model
    
    Args:
        image_path: Path to the image file
        output_path: Directory to save output files
        model: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer instance
        prompt_type: 'free' for free OCR or 'markdown' for markdown conversion
        base_size: Base size for model (512, 640, 1024, 1280)
        image_size: Image processing size
        crop_mode: Whether to use crop mode (True for Gundam config)
    
    Returns:
        Extracted text from the image
    """
    
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Prepare the prompt
    if prompt_type == 'free':
        prompt = "<image>\nFree OCR. "
    elif prompt_type == 'markdown':
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    else:
        prompt = f"<image>\n{prompt_type}"
    
    print(f"\nProcessing: {image_path}")
    print(f"Configuration: base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}")
    
    # Capture stdout to get the extracted text
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=True,
            test_compress=True
        )
    finally:
        # Restore stdout
        sys.stdout = old_stdout
    
    output_text = captured_output.getvalue()
    
    # Extract the actual text from the output (remove detection tags)
    # Find all text content between <|ref|>text<|/ref|> tags
    text_matches = re.findall(r'<\|ref\|>text<\|/ref\|><\|det\|>.*?<\|/det\|>\n(.*?)(?=\n\n|<\|ref\|>|$)', output_text, re.DOTALL)
    
    # Clean and join the extracted text
    extracted_text = '\n'.join([match.strip() for match in text_matches if match.strip()])
    
    # If no text was extracted using the pattern, try to get the raw model output
    if not extracted_text:
        # Look for text files in output directory
        txt_files = glob.glob(os.path.join(output_path, "*.txt"))
        if txt_files:
            with open(txt_files[0], 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        else:
            extracted_text = output_text
    
    return extracted_text


def load_model(model_name='deepseek-ai/DeepSeek-OCR', gpu_id='0'):
    """
    Load the DeepSeek-OCR model and tokenizer
    
    Args:
        model_name: HuggingFace model path or local path
        gpu_id: GPU device ID to use
    
    Returns:
        model, tokenizer
    """
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    
    # Set memory management for better allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This model requires GPU.")
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load the tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    
    model = model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer


def process_directory(
    root_dir,
    output_base_dir="./ocr_results",
    model=None,
    tokenizer=None,
    prompt_type='free',
    base_size=1024,
    image_size=640,
    crop_mode=True
):
    """
    Process all PNG images in a directory recursively
    
    Args:
        root_dir: Root directory to search for PNG images
        output_base_dir: Base directory to save OCR results
        model: Pre-loaded model instance
        tokenizer: Pre-loaded tokenizer instance
        prompt_type: 'free' or 'markdown'
        base_size: Model base size
        image_size: Image processing size
        crop_mode: Whether to use crop mode
    
    Returns:
        Dictionary with image paths as keys and extracted text as values
    """
    
    # Find all PNG files recursively
    png_files = glob.glob(os.path.join(root_dir, "**/*.png"), recursive=True)
    
    print(f"\nFound {len(png_files)} PNG files in {root_dir}")
    
    if len(png_files) == 0:
        print("No PNG files found!")
        return {}
    
    results = {}
    
    for idx, image_path in enumerate(png_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing image {idx}/{len(png_files)}")
        print(f"{'='*80}")
        
        try:
            # Create output directory maintaining the same structure
            rel_path = os.path.relpath(os.path.dirname(image_path), root_dir)
            output_dir = os.path.join(output_base_dir, rel_path)
            
            # Extract text
            extracted_text = extract_text_from_image(
                image_path=image_path,
                output_path=output_dir,
                model=model,
                tokenizer=tokenizer,
                prompt_type=prompt_type,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode
            )
            
            # Save the extracted text
            if extracted_text:
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                txt_filename = f"{image_name}_ocr.txt"
                txt_path = os.path.join(output_dir, txt_filename)
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                
                print(f"\n✓ Text saved to: {txt_path}")
                results[image_path] = extracted_text
            else:
                print(f"\n⚠ No text extracted from: {image_path}")
                results[image_path] = ""
                
        except Exception as e:
            print(f"\n✗ Error processing {image_path}: {e}")
            results[image_path] = f"ERROR: {e}"
    
    return results


if __name__ == "__main__":
    # Configuration
    root_directory = "/home/darshan/darshan/darshan/Placement_Data/Placements/Placements"
    output_directory = "./ocr_results_full"
    
    # GPU Configuration
    gpu_id = '1'  # Use GPU 1 (has more free memory)
    
    # OCR Configuration
    prompt_type = 'free'  # Use 'free' OCR (not markdown)
    
    # Model configuration (Gundam - recommended)
    config = {
        'base_size': 1024,
        'image_size': 640,
        'crop_mode': True
    }
    
    try:
        print("="*80)
        print("DeepSeek-OCR Batch Processing")
        print("="*80)
        print(f"Root directory: {root_directory}")
        print(f"Output directory: {output_directory}")
        print(f"GPU: {gpu_id}")
        print(f"OCR Type: {prompt_type}")
        
        # Load model once (reuse for all images)
        model, tokenizer = load_model(gpu_id=gpu_id)
        
        # Process all PNG files
        results = process_directory(
            root_dir=root_directory,
            output_base_dir=output_directory,
            model=model,
            tokenizer=tokenizer,
            prompt_type=prompt_type,
            **config
        )
        
        # Print summary
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"Total images processed: {len(results)}")
        print(f"Successful extractions: {sum(1 for v in results.values() if v and not v.startswith('ERROR'))}")
        print(f"Failed extractions: {sum(1 for v in results.values() if not v or v.startswith('ERROR'))}")
        print(f"\nResults saved in: {output_directory}")
        
        # Save summary report
        summary_file = os.path.join(output_directory, "processing_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("OCR Processing Summary\n")
            f.write("="*80 + "\n\n")
            for img_path, text in results.items():
                status = "✓ SUCCESS" if text and not text.startswith('ERROR') else "✗ FAILED"
                f.write(f"{status}: {img_path}\n")
                if text.startswith('ERROR'):
                    f.write(f"  Error: {text}\n")
                f.write("\n")
        
        print(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"\nFatal Error: {e}")
        import traceback
        traceback.print_exc()