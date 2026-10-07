#!/usr/bin/env python3
"""
Script to filter video pairs using Gemini API.
Filters out videos where:
1. Stimuli video includes pure human conversation
2. Face videos include interaction with other humans
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "common"))
import rvb_config as cfg  # noqa: E402  (loads .env, puts every stage dir on sys.path)

import argparse
import pandas as pd
import os
import time
import json
from pathlib import Path
import google.generativeai as genai
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThreadSafeResultsManager:
    """Thread-safe manager for collecting and saving results."""
    
    def __init__(self, output_file: str, save_every: bool = True):
        self.output_file = output_file
        self.save_every = save_every
        self.results = []
        self.lock = threading.Lock()
        self.count = 0
        self.processed_ids = set()
        
    def add_result(self, result: Dict):
        """Add a result in a thread-safe manner and save immediately."""
        with self.lock:
            self.results.append(result)
            self.processed_ids.add(result['id'])
            self.count += 1
            
            # Save after every result if enabled
            if self.save_every:
                self._save_results()
                if self.count % 10 == 0:
                    logger.info(f"Progress: {self.count} videos processed")
    
    def _save_results(self):
        """Save results to file (must be called with lock held)."""
        # Write to temporary file first, then rename for atomicity
        temp_file = self.output_file + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        os.replace(temp_file, self.output_file)
    
    def save_final_results(self):
        """Save final results."""
        with self.lock:
            self._save_results()
            logger.info(f"Final results saved: {self.count} total videos processed")
    
    def load_existing_results(self, filepath: str):
        """Load existing results from a file."""
        with self.lock:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.results = json.load(f)
                self.count = len(self.results)
                self.processed_ids = {r['id'] for r in self.results}
                logger.info(f"Loaded {self.count} previous results from {filepath}")
                return True
            return False
    
    def get_processed_ids(self) -> set:
        """Get set of already processed IDs."""
        with self.lock:
            return self.processed_ids.copy()
    
    def get_results(self) -> List[Dict]:
        """Get copy of all results."""
        with self.lock:
            return self.results.copy()


class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    def __init__(self, delay_seconds: float = 1.0):
        self.delay_seconds = delay_seconds
        self.lock = threading.Lock()
        self.last_call_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.delay_seconds:
                sleep_time = self.delay_seconds - time_since_last_call
                time.sleep(sleep_time)
            
            self.last_call_time = time.time()


def load_clean_ids(clean_ids_path: str) -> set:
    """Load the clean IDs from the text file."""
    with open(clean_ids_path, 'r') as f:
        clean_ids = set(line.strip() for line in f if line.strip())
    logger.info(f"Loaded {len(clean_ids)} clean IDs")
    return clean_ids


def load_and_filter_mapping(mapping_path: str, clean_ids: set) -> pd.DataFrame:
    """Load mapping CSV and filter based on criteria."""
    df = pd.read_csv(mapping_path)
    
    # Convert new_id to 5-digit string format
    df['id_str'] = df['new_id'].apply(lambda x: f"{x:05d}")
    
    logger.info(f"Total rows in mapping: {len(df)}")
    
    # Filter for patreon entries
    df_patreon = df[df['original_face_path'].str.contains('patreon', case=False, na=False)]
    logger.info(f"Rows with 'patreon' in original_face_path: {len(df_patreon)}")
    
    # Filter for clean IDs
    df_filtered = df_patreon[df_patreon['id_str'].isin(clean_ids)]
    logger.info(f"Rows after filtering for clean IDs: {len(df_filtered)}")
    
    return df_filtered


def analyze_video_with_gemini(video_path: str, prompt: str, model, rate_limiter: RateLimiter = None) -> Dict:
    """
    Analyze a video using Gemini API.
    
    Returns:
        Dict with 'answer' (bool) and 'explanation' (str)
    """
    try:
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return {'answer': None, 'explanation': 'Video file not found'}
        
        # Wait for rate limiter if provided
        if rate_limiter:
            rate_limiter.wait_if_needed()
        
        # Upload the video file
        logger.debug(f"Uploading video: {video_path}")
        video_file = genai.upload_file(path=video_path)
        
        # Wait for the file to be processed
        while video_file.state.name == "PROCESSING":
            logger.debug("Waiting for video processing...")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            logger.error(f"Video processing failed for {video_path}")
            return {'answer': None, 'explanation': 'Video processing failed'}
        
        # Generate content with the video
        response = model.generate_content(
            [video_file, prompt],
            request_options={"timeout": 120}
        )
        
        # Clean up the uploaded file
        genai.delete_file(video_file.name)
        
        # Parse the response
        response_text = response.text.strip()
        logger.debug(f"Gemini response: {response_text}")
        
        # Try to parse as JSON first
        try:
            result = json.loads(response_text)
            if isinstance(result, dict) and 'answer' in result:
                return result
        except json.JSONDecodeError:
            pass
        
        # Parse Yes/No answer
        answer_lower = response_text.lower()
        if 'yes' in answer_lower[:50]:
            answer = True
        elif 'no' in answer_lower[:50]:
            answer = False
        else:
            answer = None
        
        return {
            'answer': answer,
            'explanation': response_text
        }
        
    except Exception as e:
        logger.error(f"Error analyzing video {video_path}: {str(e)}")
        return {'answer': None, 'explanation': f'Error: {str(e)}'}


def check_stimuli_video(video_path: str, model, rate_limiter: RateLimiter = None) -> Dict:
    """
    Check if stimuli video includes pure human conversation.
    Returns True if it DOES include pure human conversation (should be filtered out).
    """
    prompt = """Analyze this video and answer the following question:

Does this video primarily contain pure human conversation (people talking to each other, interviews, dialogue scenes)?

Please respond with:
- "Yes" if the video is primarily human conversation/dialogue
- "No" if the video contains other content (music, gaming, reactions, visual content, etc.)

Then provide a brief explanation (1-2 sentences) of what you see in the video.

Format your response as:
Answer: [Yes/No]
Explanation: [Your explanation]
"""
    return analyze_video_with_gemini(video_path, prompt, model, rate_limiter)


def check_face_video(video_path: str, model, rate_limiter: RateLimiter = None) -> Dict:
    """
    Check if face video includes interaction with other humans.
    Returns True if it DOES include interaction (should be filtered out).
    """
    prompt = """Analyze this video and answer the following question:

Does this video show a person interacting with other humans (talking to someone, conversing, having a discussion with another person)?

Please respond with:
- "Yes" if the person is actively interacting with other people
- "No" if the person is alone, reacting to content, or not interacting with others

Then provide a brief explanation (1-2 sentences) of what you see in the video.

Format your response as:
Answer: [Yes/No]
Explanation: [Your explanation]
"""
    return analyze_video_with_gemini(video_path, prompt, model, rate_limiter)


def process_video_pair(row: pd.Series, model, rate_limiter: RateLimiter = None) -> Dict:
    """
    Process a video pair and determine if it should be filtered out.
    
    Returns:
        Dict with filtering decision and explanations
    """
    video_id = row['id_str']
    stimuli_path = row['new_stimuli_path']
    face_path = row['new_face_origin_path']
    
    logger.info(f"Processing video pair {video_id}")
    
    # Check stimuli video
    logger.info(f"  Checking stimuli video: {stimuli_path}")
    stimuli_result = check_stimuli_video(stimuli_path, model, rate_limiter)
    
    # Check face video
    logger.info(f"  Checking face video: {face_path}")
    face_result = check_face_video(face_path, model, rate_limiter)
    
    # Determine if should be filtered out
    # Filter out if EITHER condition is met
    should_filter = (
        (stimuli_result['answer'] is True) or  # Has pure human conversation
        (face_result['answer'] is True)  # Has interaction with others
    )
    
    result = {
        'id': video_id,
        'new_id': row['new_id'],
        'should_filter': should_filter,
        'stimuli_has_conversation': stimuli_result['answer'],
        'stimuli_explanation': stimuli_result['explanation'],
        'face_has_interaction': face_result['answer'],
        'face_explanation': face_result['explanation'],
        'original_face_path': row['original_face_path'],
        'original_stimuli_path': row['original_stimuli_path'],
        'new_face_path': face_path,
        'new_stimuli_path': stimuli_path
    }
    
    logger.info(f"Completed video pair {video_id}: should_filter={should_filter}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Filter video pairs using Gemini API'
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='Gemini API key (or set GEMINI_API_KEY environment variable)'
    )
    parser.add_argument(
        '--mapping-csv',
        default=str(cfg.meta_file('mapping.csv')),
        help='Path to mapping.csv file'
    )
    parser.add_argument(
        '--clean-ids',
        default=str(cfg.work('clean_ids', 'clean_id_list_1min_filtered.txt')),
        help='Path to clean ID list file'
    )
    parser.add_argument(
        '--output-dir',
        default=str(cfg.work('gemini_filtering')),
        help='Output directory for results'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of videos to process (for testing)'
    )
    parser.add_argument(
        '--model-name',
        default='gemini-1.5-flash',
        help='Gemini model name to use (default: gemini-1.5-flash)'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume from a previous results file'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for processing videos (default: 1)'
    )
    parser.add_argument(
        '--rate-limit-delay',
        type=float,
        default=1.0,
        help='Delay in seconds between API calls per worker (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Get API key from CLI argument or environment variable
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        logger.error("Gemini API key not provided!")
        logger.error("Please provide API key via:")
        logger.error("  1. Command line: --api-key YOUR_API_KEY")
        logger.error("  2. Environment variable: export GEMINI_API_KEY=YOUR_API_KEY")
        return
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model_name)
    
    logger.info("Starting video filtering process")
    logger.info(f"Using model: {args.model_name}")
    logger.info(f"Using {args.workers} worker(s) for parallel processing")
    
    # Load clean IDs
    clean_ids = load_clean_ids(args.clean_ids)
    
    # Load and filter mapping
    df_filtered = load_and_filter_mapping(args.mapping_csv, clean_ids)
    
    if len(df_filtered) == 0:
        logger.error("No videos to process after filtering!")
        return
    
    # Apply limit if specified
    if args.limit:
        df_filtered = df_filtered.head(args.limit)
        logger.info(f"Limited to {args.limit} videos for processing")
    
    # Setup results manager and rate limiter
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'gemini_filtering_results.json')
    results_manager = ThreadSafeResultsManager(output_file, save_every=True)
    rate_limiter = RateLimiter(delay_seconds=args.rate_limit_delay)
    
    # Automatically load previous results if they exist
    if args.resume_from:
        # Explicit resume file takes precedence
        results_manager.load_existing_results(args.resume_from)
    elif os.path.exists(output_file):
        # Auto-resume from output file if it exists
        logger.info(f"Found existing results file, resuming from {output_file}")
        results_manager.load_existing_results(output_file)
    
    # Get list of videos to process
    processed_ids = results_manager.get_processed_ids()
    videos_to_process = []
    for idx, row in df_filtered.iterrows():
        if row['id_str'] not in processed_ids:
            videos_to_process.append(row)
        else:
            logger.info(f"Skipping already processed video {row['id_str']}")
    
    total_to_process = len(videos_to_process)
    logger.info(f"Processing {total_to_process} video pairs...")
    
    # Process videos in parallel
    if args.workers == 1:
        # Sequential processing
        for row in videos_to_process:
            try:
                result = process_video_pair(row, model, rate_limiter)
                results_manager.add_result(result)
            except Exception as e:
                logger.error(f"Error processing video {row['id_str']}: {str(e)}")
                continue
    else:
        # Parallel processing with ThreadPoolExecutor
        def process_wrapper(row):
            """Wrapper to catch exceptions in threads."""
            try:
                return process_video_pair(row, model, rate_limiter)
            except Exception as e:
                logger.error(f"Error processing video {row['id_str']}: {str(e)}")
                return None
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_row = {
                executor.submit(process_wrapper, row): row 
                for row in videos_to_process
            }
            
            # Process completed tasks as they finish
            completed = 0
            for future in as_completed(future_to_row):
                result = future.result()
                if result is not None:
                    results_manager.add_result(result)
                    completed += 1
                else:
                    # Still count failed videos
                    completed += 1
                
                # Progress is logged automatically in add_result every 10 videos
                # Just log the final completion
                if completed == total_to_process:
                    logger.info(f"Completed all {total_to_process} videos")
    
    # Save final results
    results_manager.save_final_results()
    logger.info(f"Saved results to {output_file}")
    
    # Get all results for summary
    results = results_manager.get_results()
    
    # Create summary
    total_videos = len(results)
    filtered_out = sum(1 for r in results if r['should_filter'])
    kept_videos = total_videos - filtered_out
    
    # Save filtered IDs (videos to keep)
    kept_ids = [r['id'] for r in results if not r['should_filter']]
    kept_ids_file = os.path.join(args.output_dir, 'gemini_kept_ids.txt')
    with open(kept_ids_file, 'w') as f:
        for vid_id in sorted(kept_ids):
            f.write(f"{vid_id}\n")
    logger.info(f"Saved kept IDs to {kept_ids_file}")
    
    # Save filtered out IDs
    filtered_ids = [r['id'] for r in results if r['should_filter']]
    filtered_ids_file = os.path.join(args.output_dir, 'gemini_filtered_out_ids.txt')
    with open(filtered_ids_file, 'w') as f:
        for vid_id in sorted(filtered_ids):
            f.write(f"{vid_id}\n")
    logger.info(f"Saved filtered out IDs to {filtered_ids_file}")
    
    # Save summary
    summary = {
        'total_videos_processed': total_videos,
        'videos_filtered_out': filtered_out,
        'videos_kept': kept_videos,
        'filter_rate': f"{filtered_out/total_videos*100:.2f}%" if total_videos > 0 else "N/A",
        'workers_used': args.workers,
        'breakdown': {
            'stimuli_with_conversation': sum(1 for r in results if r['stimuli_has_conversation']),
            'face_with_interaction': sum(1 for r in results if r['face_has_interaction']),
        }
    }
    
    summary_file = os.path.join(args.output_dir, 'gemini_filtering_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("FILTERING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total videos processed: {total_videos}")
    logger.info(f"Videos filtered out: {filtered_out}")
    logger.info(f"Videos kept: {kept_videos}")
    logger.info(f"Filter rate: {summary['filter_rate']}")
    logger.info(f"  - Stimuli with conversation: {summary['breakdown']['stimuli_with_conversation']}")
    logger.info(f"  - Face with interaction: {summary['breakdown']['face_with_interaction']}")
    logger.info("="*60)
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"Kept IDs saved to: {kept_ids_file}")
    logger.info(f"Filtered IDs saved to: {filtered_ids_file}")


if __name__ == '__main__':
    main()

