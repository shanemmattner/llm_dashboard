import os
import logging
from typing import Dict, List, Optional, Tuple
import base64
from datetime import datetime
import json
import shutil

logger = logging.getLogger(__name__)

class DocumentManager:
    """Manage document storage and metadata"""
    
    def __init__(self, base_dir: str = "."):
        """Initialize document manager with improved structure"""
        self.base_dir = os.path.abspath(base_dir)
        self.prompt_files: Dict[str, str] = {}
        self.metadata: Dict[str, Dict] = {}
        
        # Setup directories with improved organization
        self.dirs = {
            'uploads': os.path.join(base_dir, "uploads"),
            'prompt_files': os.path.join(base_dir, "prompt_files"),
            'metadata': os.path.join(base_dir, "metadata"),
            'archive': os.path.join(base_dir, "archive")
        }
        
        self._setup_directories()
        self._load_metadata()

    def _setup_directories(self):
        """Create necessary directories with proper permissions"""
        for name, directory in self.dirs.items():
            try:
                os.makedirs(directory, exist_ok=True)
                os.chmod(directory, 0o755)
                logger.debug(f"Setup directory {name}: {directory}")
            except Exception as e:
                logger.error(f"Error setting up directory {directory}: {e}")
                raise

    def _load_metadata(self):
        """Load metadata from disk with error recovery"""
        metadata_file = os.path.join(self.dirs['metadata'], 'files.json')
        backup_file = os.path.join(self.dirs['metadata'], 'files.json.backup')
        
        try:
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                # Create backup after successful load
                shutil.copy2(metadata_file, backup_file)
            else:
                # Try to recover from backup
                if os.path.exists(backup_file):
                    logger.warning("Metadata file missing, recovering from backup")
                    with open(backup_file, 'r') as f:
                        self.metadata = json.load(f)
                else:
                    self.metadata = {}
                    
        except json.JSONDecodeError:
            logger.error("Corrupted metadata file, attempting recovery from backup")
            try:
                if os.path.exists(backup_file):
                    with open(backup_file, 'r') as f:
                        self.metadata = json.load(f)
                else:
                    self.metadata = {}
            except Exception as e:
                logger.error(f"Failed to recover metadata: {e}")
                self.metadata = {}
                
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.metadata = {}

    def _save_metadata(self):
        """Save metadata to disk with backup"""
        metadata_file = os.path.join(self.dirs['metadata'], 'files.json')
        backup_file = os.path.join(self.dirs['metadata'], 'files.json.backup')
        
        try:
            # Create new metadata file
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Update backup
            shutil.copy2(metadata_file, backup_file)
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            # Attempt to restore from backup if save fails
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, metadata_file)

    def save_file(self, content: str, filename: str, file_type: str) -> Tuple[bool, str]:
        """Save a file with improved handling and validation"""
        try:
            # Validate inputs
            if not content or not filename:
                raise ValueError("Missing required content or filename")
            
            if file_type not in ['prompt', 'rag']:
                raise ValueError(f"Invalid file type: {file_type}")
            
            # Parse content
            try:
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
            except Exception as e:
                raise ValueError(f"Invalid content format: {e}")
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            
            # Determine directory and create filepath
            directory = self.dirs['prompt_files'] if file_type == 'prompt' else self.dirs['uploads']
            filepath = os.path.join(directory, unique_filename)
            
            # Save file with error handling
            try:
                with open(filepath, 'wb') as f:
                    f.write(decoded)
                os.chmod(filepath, 0o644)
            except Exception as e:
                raise IOError(f"Failed to write file: {e}")
            
            # Update metadata
            self.metadata[unique_filename] = {
                'original_name': filename,
                'type': file_type,
                'timestamp': timestamp,
                'filepath': filepath,
                'size': len(decoded),
                'content_type': content_type
            }
            self._save_metadata()
            
            logger.info(f"Successfully saved file: {filename} ({file_type})")
            return True, filepath
            
        except Exception as e:
            logger.error(f"Error saving file {filename}: {e}")
            return False, ""

    def load_prompt_file(self, content: str, filename: str) -> bool:
        """Load a file for direct prompt use with validation"""
        try:
            # Save file first
            success, filepath = self.save_file(content, filename, 'prompt')
            if not success:
                return False
            
            # Try multiple encodings for text loading
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                        break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise UnicodeDecodeError(
                    f"Failed to decode {filename} with any supported encoding"
                )
            
            # Store content in memory
            self.prompt_files[filename] = content
            
            logger.info(f"Successfully loaded prompt file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading prompt file {filename}: {e}")
            return False

    def get_prompt_context(self, filenames: Optional[List[str]] = None) -> str:
        """Get combined content of selected prompt files."""
        if not filenames:
            filenames = list(self.prompt_files.keys())

        contexts = []
        for filename in filenames:
            if filename in self.prompt_files:
                contexts.append(f"Content from {filename}:\n{self.prompt_files[filename]}\n")
        return "\n".join(contexts)

    def get_file_list(self, file_type: Optional[str] = None) -> List[Dict]:
        """Get list of files with enhanced metadata"""
        try:
            files = []
            for filename, info in self.metadata.items():
                # Filter by type if specified
                if file_type and info['type'] != file_type:
                    continue
                
                # Enhanced file info
                file_info = {
                    'filename': info['original_name'],
                    'filepath': info['filepath'],
                    'timestamp': info['timestamp'],
                    'type': info['type'],
                    'size': info.get('size', 0),
                    'content_type': info.get('content_type', ''),
                    'exists': os.path.exists(info['filepath'])
                }
                
                files.append(file_info)
            
            # Sort by timestamp descending
            files.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting file list: {e}")
            return []

    def clear_prompt_files(self):
            """Clear loaded prompt files with cleanup and archiving"""
            try:
                # Clear memory cache
                self.prompt_files.clear()
                
                # Remove prompt files from disk
                prompt_files = [
                    f for f, info in self.metadata.items()
                    if info['type'] == 'prompt'
                ]
                
                # Create archive subdirectory with timestamp
                archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_dir = os.path.join(
                    self.dirs['archive'],
                    f"prompt_files_{archive_timestamp}"
                )
                os.makedirs(archive_dir, exist_ok=True)
                
                for filename in prompt_files:
                    try:
                        filepath = self.metadata[filename]['filepath']
                        if os.path.exists(filepath):
                            # Archive the file content
                            archive_path = os.path.join(archive_dir, filename)
                            shutil.copy2(filepath, archive_path)
                            
                            # Remove original file
                            os.remove(filepath)
                            logger.debug(f"Archived and removed {filename}")
                            
                    except Exception as e:
                        logger.error(f"Error processing file {filename} during cleanup: {e}")
                
                # Update metadata
                self.metadata = {
                    k: v for k, v in self.metadata.items()
                    if v['type'] != 'prompt'
                }
                self._save_metadata()
                
                logger.info(f"Cleared prompt files. Archived to: {archive_dir}")
                
            except Exception as e:
                logger.error(f"Error during prompt files cleanup: {e}")
                raise

    def delete_file(self, filename: str) -> bool:
        """Delete a file with archiving and metadata cleanup"""
        try:
            if filename not in self.metadata:
                logger.warning(f"File not found in metadata: {filename}")
                return False
            
            file_info = self.metadata[filename]
            filepath = file_info['filepath']
            
            # Create archive entry
            archive_info = {
                **file_info,
                'deletion_timestamp': datetime.now().isoformat(),
                'deletion_reason': 'user_requested'
            }
            
            # Archive the file if it exists
            if os.path.exists(filepath):
                try:
                    # Create archive path
                    archive_path = os.path.join(
                        self.dirs['archive'],
                        f"deleted_{filename}"
                    )
                    
                    # Copy file to archive
                    shutil.copy2(filepath, archive_path)
                    
                    # Remove original file
                    os.remove(filepath)
                    logger.debug(f"Archived and removed file: {filepath}")
                    
                except Exception as e:
                    logger.error(f"Error archiving file {filename}: {e}")
                    return False
            
            # Save archive metadata
            archive_metadata_path = os.path.join(
                self.dirs['archive'],
                f"deleted_{filename}_metadata.json"
            )
            try:
                with open(archive_metadata_path, 'w') as f:
                    json.dump(archive_info, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving archive metadata for {filename}: {e}")
            
            # Remove from current metadata
            del self.metadata[filename]
            self._save_metadata()
            
            # Remove from prompt files if present
            if filename in self.prompt_files:
                del self.prompt_files[filename]
            
            logger.info(f"Successfully deleted and archived file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {e}")
            return False

    def get_file_stats(self) -> Dict[str, any]:
        """Get statistical information about managed files"""
        try:
            stats = {
                'total_files': len(self.metadata),
                'prompt_files': 0,
                'rag_files': 0,
                'total_size': 0,
                'file_types': {},
                'missing_files': 0,
                'processing_dates': set(),
                'storage_usage': {
                    'uploads': 0,
                    'prompt_files': 0,
                    'archive': 0
                }
            }
            
            # Process metadata
            for info in self.metadata.values():
                # Count by type
                file_type = info['type']
                if file_type == 'prompt':
                    stats['prompt_files'] += 1
                else:
                    stats['rag_files'] += 1
                
                # Track file types
                ext = os.path.splitext(info['original_name'])[1].lower()
                stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
                
                # Track size
                stats['total_size'] += info.get('size', 0)
                
                # Check file existence
                if not os.path.exists(info['filepath']):
                    stats['missing_files'] += 1
                
                # Track processing dates
                date = info['timestamp'].split('_')[0]
                stats['processing_dates'].add(date)
            
            # Calculate storage usage
            for dir_name, dir_path in self.dirs.items():
                if os.path.exists(dir_path):
                    total_size = sum(
                        os.path.getsize(os.path.join(dirpath, f))
                        for dirpath, dirnames, filenames in os.walk(dir_path)
                        for f in filenames
                    )
                    stats['storage_usage'][dir_name] = total_size
            
            # Convert dates to sorted list
            stats['processing_dates'] = sorted(list(stats['processing_dates']))
            
            # Add summary statistics
            stats['summary'] = {
                'total_files': stats['total_files'],
                'active_files': stats['total_files'] - stats['missing_files'],
                'total_size_mb': round(stats['total_size'] / (1024 * 1024), 2),
                'total_storage_mb': round(sum(stats['storage_usage'].values()) / (1024 * 1024), 2)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating file stats: {e}")
            return {}