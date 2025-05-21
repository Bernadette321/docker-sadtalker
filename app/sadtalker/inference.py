import os
import shutil
import torch
from time import strftime
import tempfile # Keep this if needed for other parts of your app

# SadTalker specific imports
# We need to adjust import paths based on SadTalker_source being a subdirectory
from .SadTalker_source.src.utils.preprocess import CropAndExtract
from .SadTalker_source.src.test_audio2coeff import Audio2Coeff  
from .SadTalker_source.src.facerender.animate import AnimateFromCoeff
from .SadTalker_source.src.generate_batch import get_data
from .SadTalker_source.src.generate_facerender_batch import get_facerender_data
from .SadTalker_source.src.utils.init_path import init_path

# Helper class to mimic argparse.Namespace for SadTalker functions
class SadTalkerArgs:
    def __init__(self, source_image, driven_audio, result_dir):
        self.source_image = source_image
        self.driven_audio = driven_audio
        self.result_dir = result_dir
        
        # Default values from original SadTalker inference.py, adjust as needed
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), 'SadTalker_source', 'checkpoints')
        self.config_dir = os.path.join(os.path.dirname(__file__), 'SadTalker_source', 'src', 'config')
        self.pose_style = 0
        self.batch_size = 2
        self.size = 256  # Output video size
        self.expression_scale = 1.0
        self.input_yaw = None
        self.input_pitch = None
        self.input_roll = None
        self.ref_eyeblink = None # Path to reference video for eye blinking
        self.ref_pose = None     # Path to reference video for pose
        self.enhancer = 'gfpgan' # Options: None, 'gfpgan', 'RestoreFormer'
        self.background_enhancer = None # Options: None, 'realesrgan'
        self.cpu = False
        self.face3dvis = False
        self.still = True # To generate a still mode video (image talking)
        self.preprocess = 'crop' # Options: 'crop', 'extcrop', 'resize', 'full', 'extfull'
        self.verbose = False
        self.old_version = False # Use safetensors by default

        # Determine device
        if torch.cuda.is_available() and not self.cpu:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # These are also from the original script, might not all be used directly in our simplified case
        self.net_recon='resnet50'
        self.init_path=None
        self.use_last_fc=False
        self.bfm_folder=os.path.join(os.path.dirname(__file__), 'SadTalker_source', 'checkpoints', 'BFM_Fitting')
        self.bfm_model='BFM_model_front.mat'
        self.focal=1015.
        self.center=112.
        self.camera_d=10.
        self.z_near=5.
        self.z_far=15.


def run_sadtalker(image_path: str, audio_path: str) -> str:
    # Create a temporary directory for results, as SadTalker creates subdirectories
    # The main FastAPI app will handle the final result file and cleanup of the parent temp_dir
    # We use a timestamped subdirectory for SadTalker's own output structure.
    
    # The main app creates a temp_dir, we'll create our own subdir within it for SadTalker
    # Let's use a fixed name for now, or derive from input if needed.
    # The `main.py` already creates a temp_dir which is cleaned up.
    # We need a *subdirectory* within that for SadTalker's specific outputs.
    
    # It's better if run_sadtalker itself defines its *own* temporary output space if it needs one,
    # rather than relying on the caller's temp_dir structure directly, for better encapsulation.
    # However, the original script uses args.result_dir.
    # Let's make result_dir a subdirectory of the input image_path's directory,
    # assuming image_path and audio_path are in a shared temp directory managed by main.py.
    
    # The main.py already creates a general temp_dir. We need to ensure SadTalker's outputs
    # go into a predictable place *within* that temp_dir or a new one.
    # The original script puts results in args.result_dir / timestamp.
    # We need to return a single file path.
    
    sadtalker_temp_output_dir = tempfile.mkdtemp() # SadTalker will create its timestamped folder inside this.

    args = SadTalkerArgs(source_image=image_path, driven_audio=audio_path, result_dir=sadtalker_temp_output_dir)

    # Ensure the root for SadTalker (where it expects 'src' and 'checkpoints') is correct.
    # current_root_path in the original script is os.path.split(sys.argv[0])[0]
    # In our case, it should be the directory of this inference.py file.
    current_script_dir = os.path.dirname(__file__)
    sadtalker_root_path = os.path.join(current_script_dir, 'SadTalker_source')


    sadtalker_paths = init_path(args.checkpoint_dir, args.config_dir, args.size, args.old_version, args.preprocess)

    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, args.device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths, args.device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, args.device)

    # Create the specific output directory SadTalker expects (timestamped)
    # The original main() uses save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    # We'll use a fixed name or derive one to make it easier to retrieve the final .mp4
    # For now, let SadTalker create its timestamped directory inside sadtalker_temp_output_dir
    
    # The official script generates a save_dir like: args.result_dir/YYYY_MM_DD_HH.MM.SS
    # And then moves the final result to args.result_dir.mp4 (this seems odd, might be save_dir + '.mp4')
    # Let's clarify this: shutil.move(result, save_dir+'.mp4') suggests the final file is save_dir.mp4 in the parent of save_dir
    
    # Let's define our save_dir which is where intermediate files and the final video (before move) will be
    # We will make it *inside* our sadtalker_temp_output_dir to keep things contained.
    
    # SadTalker's main function creates its own timestamped save_dir.
    # We need to adapt this. Let's use a predictable name for the output directory within sadtalker_temp_output_dir.
    # Or, let it create its timestamped dir, then find the .mp4 file.

    # The main function in original inference.py:
    # save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    # os.makedirs(save_dir, exist_ok=True)
    # ...
    # shutil.move(result, save_dir+'.mp4') -> this means result is moved to args.result_dir/timestamp.mp4
    # This is good, we can predict this.

    #crop image and extract 3dmm from image
    # The original script puts first_frame_dir inside its save_dir.
    # We need to ensure paths are correctly resolved.
    
    # SadTalker's main() will create a timestamped subdirectory in args.result_dir.
    # Let's call the original main function's logic, adapted.

    # === Replicating SadTalker main() logic starts here ===
    
    # 1. Define save_dir (timestamped, inside args.result_dir which is our sadtalker_temp_output_dir)
    # This will be created by the SadTalker internal logic if we pass args correctly or replicate it.
    # For simplicity, let's directly replicate the save_dir creation.
    effective_save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(effective_save_dir, exist_ok=True)

    first_frame_dir = os.path.join(effective_save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        args.source_image, 
        first_frame_dir, 
        args.preprocess,
        source_image_flag=True, 
        pic_size=args.size
    )
    
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        # Clean up and raise an error or return a specific value
        shutil.rmtree(sadtalker_temp_output_dir)
        raise RuntimeError("SadTalker: 3DMM extraction failed for source image.")

    ref_eyeblink_coeff_path = None
    if args.ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(args.ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(effective_save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(
            args.ref_eyeblink, 
            ref_eyeblink_frame_dir, 
            args.preprocess, 
            source_image_flag=False
        )

    ref_pose_coeff_path = None
    if args.ref_pose is not None:
        if args.ref_pose == args.ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(args.ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(effective_save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ = preprocess_model.generate(
                args.ref_pose, 
                ref_pose_frame_dir, 
                args.preprocess, 
                source_image_flag=False
            )

    #audio2coeff
    batch = get_data(first_coeff_path, args.driven_audio, args.device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, effective_save_dir, args.pose_style, ref_pose_coeff_path)

    # 3dface render (optional, controlled by args.face3dvis)
    if args.face3dvis:
        from .SadTalker_source.src.face3d.visualize import gen_composed_video
        gen_composed_video(args, args.device, first_coeff_path, coeff_path, args.driven_audio, os.path.join(effective_save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(
        coeff_path, 
        crop_pic_path, 
        first_coeff_path, 
        args.driven_audio, 
        args.batch_size, 
        args.input_yaw, 
        args.input_pitch, 
        args.input_roll,
        expression_scale=args.expression_scale, 
        still_mode=args.still, 
        preprocess=args.preprocess, 
        size=args.size
    )
    
    result_video_path = animate_from_coeff.generate(
        data, 
        effective_save_dir, 
        args.source_image, # pic_path in original
        crop_info, 
        enhancer=args.enhancer, 
        background_enhancer=args.background_enhancer, 
        preprocess=args.preprocess, 
        img_size=args.size
    )
    
    # The `result_video_path` from animate_from_coeff.generate is the path to the final video.
    # The original script then does: shutil.move(result, save_dir+'.mp4')
    # This means the video `result` (which is `result_video_path` here) is moved to
    # `effective_save_dir.mp4` (i.e., a file named like the timestamped folder, but with .mp4, in `args.result_dir`).
    
    final_output_path_sadtalker = effective_save_dir + '.mp4'
    shutil.move(result_video_path, final_output_path_sadtalker)
    
    print(f'The generated video is named: {final_output_path_sadtalker}')

    if not args.verbose:
        # Clean up the timestamped subdirectory if not verbose
        shutil.rmtree(effective_save_dir)
        # The final_output_path_sadtalker is now in sadtalker_temp_output_dir (args.result_dir)
        # The main FastAPI app should be responsible for cleaning up sadtalker_temp_output_dir later
        # after it has copied the file.

    # We need to return the path to this final_output_path_sadtalker
    # This path is inside sadtalker_temp_output_dir.
    return final_output_path_sadtalker

    # Placeholder from original template, to be removed
    # output_video = os.path.splitext(audio_path)[0] + "_sadtalker.mp4"
    # with open(output_video, "wb") as f:
    #     f.write(b"") # Create a dummy file
    # return output_video